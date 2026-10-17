"""Backtests that answer purchasing questions directly from the history."""
from __future__ import annotations

import numpy as np
import pandas as pd


def dormant_rule(wide: pd.DataFrame, at: int = 24, gap: int = 12) -> dict:
    """Parts with no demand in the ``gap`` months before month ``at``: how many sold again afterwards?"""
    y = np.nan_to_num(wide.to_numpy(dtype=float))
    silent = y[at - gap:at].sum(axis=0) == 0
    later = y[at:].sum(axis=0) > 0
    return {"at_month": str(wide.index[at]), "gap_months": gap, "parts_flagged": int(silent.sum()),
            "sold_again": int((silent & later).sum()),
            "share_sold_again": float((silent & later).sum() / max(silent.sum(), 1)),
            "units_sold_later_by_flagged": float(y[at:, silent].sum()), "units_sold_later_total": float(y[at:].sum())}


def yearly_demand(wide: pd.DataFrame) -> pd.DataFrame:
    tot = wide.fillna(0).sum(axis=1)
    g = tot.groupby(tot.index.year).agg(units="sum", months="count")
    g["units_per_month"] = g["units"] / g["months"]
    return g.reset_index(names="year")
