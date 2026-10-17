import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from csc import backtests, classify, data, evaluate, forecast, inventory
from csc.recommend import recommend

ROOT = Path(__file__).resolve().parent.parent


def test_data_file_and_shape():
    w = data.load_wide()
    assert w.shape == (51, 2674) and data.complete_parts(w).shape[1] == 2509
    assert data.quality(w)["history_stops_early"] == 165


def test_long_round_trip():
    w = data.complete_parts(data.load_wide()).iloc[:, :20]
    back = data.from_long(data.to_long(w))
    pd.testing.assert_frame_equal(back[w.columns].astype(float), w, check_freq=False, check_names=False)


def test_sbc_classes():
    w = pd.DataFrame({"smooth": [5, 6, 5, 6, 5, 6], "lumpy": [0, 0, 9, 0, 1, 0], "rare": [0, 0, 0, 0, 0, 3]},
                     index=pd.period_range("2020-01", periods=6, freq="M"), dtype=float)
    c = classify.sbc(w).set_index("part")["demand_class"]
    assert c["smooth"] == "smooth" and c["lumpy"] == "lumpy" and c["rare"] == "too few demands"


def test_forecasts_use_only_the_past():
    rng = np.random.default_rng(0)
    y = rng.poisson(0.5, (40, 30)).astype(float)
    for name, fn in forecast.METHODS.items():
        f1 = fn(y)
        y2 = y.copy()
        y2[30:] = 99  # changing the future must not change forecasts up to month 30
        assert np.allclose(np.nan_to_num(f1[:31]), np.nan_to_num(fn(y2)[:31])), name


def test_sba_is_croston_times_correction():
    y = np.random.default_rng(1).poisson(0.4, (36, 10)).astype(float)
    assert np.allclose(np.nan_to_num(forecast.sba(y)), np.nan_to_num(forecast.croston(y)) * 0.95)


def test_simulation_conserves_units():
    y = np.random.default_rng(2).poisson(1.0, (40, 5)).astype(float)
    r = inventory.simulate(y, inventory.order_up_to(y, "ses_normal", 2), 2)
    assert np.all(r["served"] <= r["demanded"]) and r["demanded"].sum() == y[evaluate.TEST_START:].sum()


def test_more_safety_stock_never_lowers_fill_rate():
    w = data.complete_parts(data.load_wide()).iloc[:, :300]
    cls = classify.sbc(w.iloc[:24]).set_index("part")["demand_class"]
    s = inventory.compare(w, cls, leads=(3,), methods=("ses_normal",), quantiles=(0.8, 0.95))
    a = s[s["demand_class"] == "all"].set_index("target")
    assert a.loc[0.95, "fill_rate"] >= a.loc[0.8, "fill_rate"]


def test_recommend_with_stock_and_flags():
    idx = pd.period_range("2020-01", periods=24, freq="M")
    w = pd.DataFrame({"busy": [2.0] * 24, "dead": [1.0] * 6 + [0.0] * 18, "once": [0.0] * 23 + [4.0]}, index=idx)
    st = pd.DataFrame({"part": ["busy", "dead", "once"], "on_hand": [0, 5, 0], "on_order": [0, 0, 0]})
    r = recommend(w, lead=2, target=0.95, stock=st).set_index("part")
    assert r.loc["busy", "proposed_order"] > 0
    assert "no_demand_12m" in r.loc["dead", "flags"] and r.loc["dead", "proposed_order"] == 0
    assert "too_few_demands" in r.loc["once", "flags"]


def test_cli(tmp_path):
    from csc.__main__ import main
    d = pd.DataFrame({"part": ["A"] * 24, "month": [str(p) for p in pd.period_range("2021-01", periods=24,
                                                                                     freq="M")], "qty": [1] * 24})
    d.to_csv(tmp_path / "d.csv", index=False)
    assert main([str(tmp_path / "d.csv"), "--out", str(tmp_path / "r.csv")]) == 0
    assert pd.read_csv(tmp_path / "r.csv").loc[0, "order_up_to_level"] > 0
    pd.DataFrame({"part": ["A"]}).to_csv(tmp_path / "bad.csv", index=False)
    assert main([str(tmp_path / "bad.csv")]) == 2
    d.head(6).to_csv(tmp_path / "short.csv", index=False)
    assert main([str(tmp_path / "short.csv")]) == 2


def test_sql_matches_pandas():
    rec = pd.read_csv(ROOT / "results" / "recommendations_demo.csv", dtype={"part": str})
    con = duckdb.connect()
    con.register("recommendations", rec)
    con.register("demand", pd.DataFrame({"part": ["x"], "month": ["2000-01"], "qty": [1.0]}))
    con.execute((ROOT / "sql" / "kpis.sql").read_text())
    sql = con.execute("SELECT demand_class, parts FROM class_summary").df().set_index("demand_class")["parts"]
    assert sql.to_dict() == rec["demand_class"].value_counts().to_dict()


def test_summary_is_current():
    s = json.loads((ROOT / "results" / "summary.json").read_text())
    w = data.complete_parts(data.load_wide())
    assert s["dormant_rule"][1] == backtests.dormant_rule(w, 24, 12)
