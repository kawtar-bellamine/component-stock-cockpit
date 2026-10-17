"""Forecast accuracy over a rolling origin: every month of the test window is forecast with the past only.

Three measures, each answering a different question:

* scaled MAE: typical size of the error, relative to the part's average demand. For intermittent demand it
  rewards forecasting zero, because most months are zero.
* scaled MSE: penalises large misses; it is the one that matters for stock, since stock has to cover the months
  with demand.
* bias: total forecast minus total demand, over total demand. A positive bias builds stock, a negative one
  creates shortages.
* relative MSE: each part's MSE divided by the MSE of the 12-month moving average, then the geometric mean over
  parts. Below 1, the method beats the moving average on large misses. It avoids letting a few parts with a tiny
  average dominate the result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast import METHODS

TEST_START = 24  # first test month (index): two years of history before the first scored forecast


def accuracy(wide: pd.DataFrame, classes: pd.Series, start: int = TEST_START) -> pd.DataFrame:
    y = wide.to_numpy(dtype=float)
    scale = np.nanmean(y[:start], axis=0)
    ok = scale > 0
    rows = []
    ref = np.nanmean((METHODS["moving_average"](y)[start:] - y[start:]) ** 2, axis=0)
    for name, fn in METHODS.items():
        f = fn(y)[start:]
        e = f - y[start:]
        mae = np.nanmean(np.abs(e), axis=0) / np.where(ok, scale, np.nan)
        mse = np.nanmean(e ** 2, axis=0) / np.where(ok, scale ** 2, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.log(np.nanmean(e ** 2, axis=0) / ref)
        rel = np.where(np.isfinite(rel), rel, np.nan)
        for cls, idx in pd.Series(range(y.shape[1]), index=classes.to_numpy()).groupby(level=0):
            i = idx.to_numpy()
            rows.append({"method": name, "demand_class": cls, "parts": int(ok[i].sum()),
                         "scaled_mae": float(np.nanmean(mae[i])), "scaled_mse": float(np.nanmean(mse[i])),
                         "relative_mse": float(np.exp(np.nanmean(rel[i]))),
                         "bias": float(np.nansum(e[:, i]) / np.nansum(y[start:, i]))})
        rows.append({"method": name, "demand_class": "all", "parts": int(ok.sum()),
                     "scaled_mae": float(np.nanmean(mae)), "scaled_mse": float(np.nanmean(mse)),
                     "relative_mse": float(np.exp(np.nanmean(rel))),
                     "bias": float(np.nansum(e) / np.nansum(y[start:]))})
    return pd.DataFrame(rows)
