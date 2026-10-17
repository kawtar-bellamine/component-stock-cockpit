"""Purchase recommendations for each part, at the last month of the history.

The order-up-to level uses simple exponential smoothing with normally distributed errors (``ses_normal``), the
method that gave the best fill rate for a given stock in the backtest (see README). Parts whose history cannot
support a statistical level are flagged for a manual decision instead of being given a number that looks precise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .classify import abc, sbc
from .forecast import ses
from .inventory import order_up_to

MIN_MONTHS = 13  # 12 months to initialise the forecast, plus one month of errors

FLAG_TEXT = {
    "no_demand_12m": "no demand for 12 months: check with engineering before buying (in the backtest, most such "
                     "parts sold again)",
    "too_few_demands": "fewer than two demands: decide manually (last-time buy, on-demand purchase, or minimum stock)",
    "excess": "stock above twice the order-up-to level: stop orders and review",
}


def recommend(wide: pd.DataFrame, lead: int = 3, target: float = 0.95,
              stock: pd.DataFrame | None = None) -> pd.DataFrame:
    w = wide.fillna(0.0)
    if len(w) < MIN_MONTHS:
        raise ValueError(f"at least {MIN_MONTHS} months of history are needed, got {len(w)}")
    y = w.to_numpy(dtype=float)
    ext = np.vstack([y, np.full((1, y.shape[1]), np.nan)])  # one empty month: forecasts made with all history
    fc = ses(ext)[-1]
    level = np.ceil(order_up_to(ext, "ses_normal", lead, target)[-1])
    cls = sbc(w).set_index("part")
    out = pd.DataFrame({"part": w.columns, "demand_class": cls.loc[w.columns, "demand_class"].to_numpy(),
                        "abc_units": abc(w).reindex(w.columns).to_numpy(),
                        "demand_last_12m": y[-12:].sum(axis=0),
                        "months_since_last_demand": [int(len(c) - 1 - np.flatnonzero(c > 0)[-1])
                                                     if (c > 0).any() else len(c) for c in y.T],
                        "forecast_per_month": fc.round(3), "order_up_to_level": level})
    if stock is not None:
        s = stock.set_index("part").reindex(out["part"]).fillna(0)
        out["on_hand"] = s["on_hand"].to_numpy()
        out["on_order"] = s.get("on_order", pd.Series(0, index=s.index)).to_numpy()
        out["proposed_order"] = np.maximum(out["order_up_to_level"] - out["on_hand"] - out["on_order"], 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["coverage_months"] = np.where(fc > 0, out["on_hand"] / fc, np.nan).round(1)
    flags = []
    for r in out.itertuples():
        f = []
        if r.demand_class == "too few demands":
            f.append("too_few_demands")
        if r.demand_last_12m == 0:
            f.append("no_demand_12m")
        if stock is not None and r.on_hand > 2 * max(r.order_up_to_level, 1):
            f.append("excess")
        flags.append(";".join(f))
    out["flags"] = flags
    if stock is not None:
        manual = out["flags"].str.contains("too_few_demands|no_demand_12m")
        out.loc[manual, "proposed_order"] = 0
    return out
