"""Rebuild every result, figure input and Power BI file.

    python experiments/run_all.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csc import backtests, classify, data, evaluate, inventory  # noqa: E402
from csc.recommend import recommend  # noqa: E402

RES, PBI = ROOT / "results", ROOT / "powerbi"
LEAD = 3


def main() -> None:
    RES.mkdir(exist_ok=True)
    PBI.mkdir(exist_ok=True)
    wide = data.load_wide()
    q = data.quality(wide)
    comp = data.complete_parts(wide)
    cls_train = classify.sbc(comp.iloc[:evaluate.TEST_START]).set_index("part")["demand_class"]
    cls_full = classify.sbc(wide)

    acc = evaluate.accuracy(comp, cls_train)
    acc.to_csv(RES / "forecast_accuracy.csv", index=False)
    stock = inventory.compare(comp, cls_train, leads=(1, 3), quantiles=(0.8, 0.9, 0.95, 0.98))
    stock.to_csv(RES / "stock_policy_backtest.csv", index=False)
    dormant = [backtests.dormant_rule(comp, 24, g) for g in (6, 12)]
    pd.DataFrame(dormant).to_csv(RES / "dormant_rule_backtest.csv", index=False)
    yearly = backtests.yearly_demand(comp)
    yearly.to_csv(RES / "demand_by_year.csv", index=False)
    cls_full.to_csv(RES / "part_classes.csv", index=False)

    # demonstration: simulated stock at the end of the history under the recommended policy
    y = comp.to_numpy(dtype=float)
    sim = inventory.simulate(y, inventory.order_up_to(y, "ses_normal", LEAD, 0.95), LEAD)
    demo_stock = pd.DataFrame({"part": comp.columns, "on_hand": sim["final_on_hand"],
                               "on_order": sim["final_on_order"]})
    demo_stock.to_csv(RES / "demo_stock_simulated.csv", index=False)
    rec = recommend(comp, LEAD, 0.95, demo_stock)
    rec.to_csv(RES / "recommendations_demo.csv", index=False)

    # SQL KPIs on the same tables
    con = duckdb.connect()
    con.register("demand", data.to_long(comp))
    con.register("recommendations", rec)
    con.execute((ROOT / "sql" / "kpis.sql").read_text())
    for view in ["demand_by_year", "class_summary", "dormant_with_stock", "orders_by_abc"]:
        con.execute(f"SELECT * FROM {view}").df().to_csv(RES / f"sql_{view}.csv", index=False)
    con.close()

    # Power BI
    data.to_long(comp).to_csv(PBI / "fact_demand.csv", index=False)
    cls_full.merge(classify.abc(wide).rename_axis("part").reset_index(), on="part").to_csv(
        PBI / "dim_part.csv", index=False)
    rec.to_csv(PBI / "fact_recommendation.csv", index=False)
    stock.to_csv(PBI / "stock_policy_backtest.csv", index=False)

    a_all = acc[acc["demand_class"] == "all"].set_index("method")
    a_int = acc[acc["demand_class"] == "intermittent"].set_index("method")
    s95 = stock[(stock["target"] == 0.95) & (stock["lead_time_months"] == LEAD)]
    summary = {
        "data": {**q, "sha256": data.sha256(data.RAW), "complete_parts": int(comp.shape[1])},
        "classes_full_history": cls_full["demand_class"].value_counts().to_dict(),
        "classes_first_24_months": cls_train.value_counts().to_dict(),
        "no_demand_last_12_months": int((np.nan_to_num(wide.to_numpy())[-12:].sum(axis=0) == 0).sum()),
        "demand_by_year": yearly.round(1).to_dict("records"),
        "accuracy_all": a_all[["scaled_mae", "relative_mse", "bias"]].round(3).to_dict("index"),
        "accuracy_intermittent": a_int[["scaled_mae", "relative_mse", "bias"]].round(3).to_dict("index"),
        "stock_95_lead3": s95.round(3).to_dict("records"),
        "dormant_rule": dormant,
        "demo_orders": {"parts_to_order": int((rec["proposed_order"] > 0).sum()),
                        "units": float(rec["proposed_order"].sum()),
                        "flagged": rec["flags"].str.split(";").explode().replace("", np.nan).dropna()
                        .value_counts().to_dict()},
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
