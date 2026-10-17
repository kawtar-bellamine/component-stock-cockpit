"""Stock policy simulation on the real demand history.

Periodic review every month, order-up-to level S, lead time L months, unmet demand back-ordered. At the start of
each test month the policy only knows the past. Three ways of setting S for a service target (95% by default):

* ``sba_normal``: SBA forecast x (L + 1) + z x RMSE x sqrt(L + 1), the textbook formula that assumes
  normally distributed forecast errors (z = 1.645 for 95%);
* ``ses_normal``: the same with simple exponential smoothing;
* ``empirical``: the 95th percentile of all past (L + 1)-month demand totals of the part, with no distribution
  assumed.

Fill rate = units served from stock / units demanded. Demand months served in full = share of the months with
demand where all of it was served from stock (months without demand are left out, they would flatter the result).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import TEST_START
from .forecast import WARMUP, sba, ses


def rolling_rmse(y: np.ndarray, f: np.ndarray, window: int = 12) -> np.ndarray:
    """RMSE of the one-step errors over the ``window`` months before t (known at time t)."""
    e2 = (f - y) ** 2
    out = np.full_like(y, np.nan, dtype=float)
    for t in range(WARMUP + 3, len(y)):
        out[t] = np.sqrt(np.nanmean(e2[max(WARMUP, t - window):t], axis=0))
    return out


def order_up_to(y: np.ndarray, method: str, lead: int, quantile: float = 0.95) -> np.ndarray:
    h = lead + 1
    if method in ("sba_normal", "ses_normal"):
        f = sba(y) if method == "sba_normal" else ses(y)
        from statistics import NormalDist
        z = NormalDist().inv_cdf(quantile)
        return f * h + z * rolling_rmse(y, f) * np.sqrt(h)
    if method == "empirical":
        out = np.full_like(y, np.nan, dtype=float)
        csum = np.vstack([np.zeros(y.shape[1]), np.cumsum(np.nan_to_num(y), axis=0)])
        for t in range(WARMUP, len(y)):
            windows = csum[h:t + 1] - csum[:t + 1 - h]  # every complete past (L+1)-month total
            out[t] = np.quantile(windows, quantile, axis=0, method="higher")
        return out
    raise ValueError(method)


def simulate(y: np.ndarray, levels: np.ndarray, lead: int, start: int = TEST_START) -> dict[str, np.ndarray]:
    """Monthly review from ``start``; orders placed at the end of month t arrive at the start of month t + lead + 1.
    The pipeline starts empty and the net stock starts at the first order-up-to level."""
    t_len, n = y.shape
    s = np.ceil(np.nan_to_num(levels[start:], nan=0.0))
    net = s[0].copy()
    pipeline = np.zeros((lead + 1, n))
    served_p = np.zeros(n)
    demanded_p = np.zeros(n)
    short_months = np.zeros(n)
    demand_months = np.zeros(n)
    onhand_p = np.zeros(n)
    for k, t in enumerate(range(start, t_len)):
        net += pipeline[0]
        pipeline = np.vstack([pipeline[1:], np.zeros(n)])
        d = np.nan_to_num(y[t])
        avail = np.maximum(net, 0)
        sv = np.minimum(avail, d)
        served_p += sv
        demanded_p += d
        short_months += d > avail
        demand_months += d > 0
        net -= d
        onhand_p += np.maximum(net, 0)
        position = net + pipeline.sum(axis=0)
        level = s[k + 1] if k + 1 < len(s) else s[k]
        pipeline[-1] += np.maximum(level - position, 0)
    months = t_len - start
    return {"served": served_p, "demanded": demanded_p, "short_months": short_months,
            "demand_months": demand_months, "avg_onhand": onhand_p / months, "months": months,
            "final_on_hand": np.maximum(net, 0), "final_on_order": pipeline.sum(axis=0)}


def _summary(r: dict, i: np.ndarray) -> dict:
    return {"parts": len(i), "fill_rate": r["served"][i].sum() / r["demanded"][i].sum(),
            "demand_months_served_in_full": 1 - r["short_months"][i].sum() / r["demand_months"][i].sum(),
            "avg_units_on_hand_per_part": r["avg_onhand"][i].mean(),
            "stock_in_months_of_demand": r["avg_onhand"][i].sum() / (r["demanded"][i].sum() / r["months"])}


def compare(wide: pd.DataFrame, classes: pd.Series, leads=(1, 3), methods=("sba_normal", "ses_normal", "empirical"),
            quantiles=(0.95,), start: int = TEST_START) -> pd.DataFrame:
    y = wide.to_numpy(dtype=float)
    groups = list(pd.Series(range(y.shape[1]), index=classes.to_numpy()).groupby(level=0))
    groups.append(("all", pd.Series(range(y.shape[1]))))
    rows = []
    for lead in leads:
        for m in methods:
            for q in quantiles:
                r = simulate(y, order_up_to(y, m, lead, q), lead, start)
                for cls, idx in groups:
                    rows.append({"lead_time_months": lead, "method": m, "target": q, "demand_class": cls,
                                 **_summary(r, idx.to_numpy())})
    return pd.DataFrame(rows)
