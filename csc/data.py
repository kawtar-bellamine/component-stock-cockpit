"""Demand data.

The reference data set is ``carparts``: monthly sales of 2,674 slow-moving car spare parts, January 1998 to
March 2002 (51 months), from Hyndman, Koehler, Ord and Snyder, *Forecasting with exponential smoothing* (Springer,
2008), distributed in the R package ``expsmooth``. It is used here because it is a real, public history of
intermittent demand, the same pattern as most electronic components in a low-volume, high-mix industry.

Any other history can be used in the same format: one row per part and month (``part``, ``month``, ``qty``).
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "carparts.rda"
RAW_SHA256 = "81f1d1f1785278e376651ff447b6b0e0a2cf763373f01d61fe04fad0b7f20b0c"
FIRST_MONTH = "1998-01"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def load_wide() -> pd.DataFrame:
    """Months in rows, parts in columns; NaN where the history has stopped."""
    import pyreadr

    if sha256(RAW) != RAW_SHA256:
        raise ValueError("data/raw/carparts.rda is not the expected file")
    df = next(iter(pyreadr.read_r(str(RAW)).values())).astype(float)
    df.index = pd.period_range(FIRST_MONTH, periods=len(df), freq="M")
    df.columns = df.columns.astype(str)
    return df


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    long = wide.stack(future_stack=True).rename("qty").reset_index()
    long.columns = ["month", "part", "qty"]
    long["month"] = long["month"].astype(str)
    return long.dropna(subset=["qty"])


def from_long(long: pd.DataFrame) -> pd.DataFrame:
    """Back to a month x part matrix; months with no row inside a part's history count as zero demand."""
    w = long.pivot_table(index="month", columns="part", values="qty", aggfunc="sum")
    w.index = pd.PeriodIndex(w.index, freq="M")
    full = pd.period_range(w.index.min(), w.index.max(), freq="M")
    return w.reindex(full).fillna(0.0)


def complete_parts(wide: pd.DataFrame) -> pd.DataFrame:
    """Parts observed in every month (the evaluation set)."""
    return wide.loc[:, wide.notna().all()]


def quality(wide: pd.DataFrame) -> dict:
    z = wide.fillna(0).to_numpy()
    stopped = wide.isna().any()
    return {"parts": int(wide.shape[1]), "months": int(wide.shape[0]),
            "first_month": str(wide.index[0]), "last_month": str(wide.index[-1]),
            "history_stops_early": int(stopped.sum()),
            "units": float(z.sum()), "max_monthly_qty": float(np.nanmax(wide.to_numpy())),
            "share_of_zero_months": float((wide.to_numpy()[wide.notna().to_numpy()] == 0).mean())}
