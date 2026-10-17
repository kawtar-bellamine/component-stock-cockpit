"""Forecasting methods for slow-moving parts, all one step ahead and without look-ahead.

Each function takes a month x part matrix and returns a matrix of the same shape where row t is the forecast for
month t made with months 0 to t-1 only (NaN where there is not enough history yet).

* ``zero``: always 0. A benchmark that point-accuracy measures such as MAE reward for intermittent demand.
* ``naive``: last month's demand.
* ``moving_average``: mean of the last 12 months.
* ``ses``: simple exponential smoothing (alpha 0.1).
* ``croston``: Croston (1972): sizes and intervals smoothed separately, forecast = size / interval.
* ``sba``: Syntetos-Boylan approximation, Croston times (1 - alpha / 2), which removes Croston's bias.
* ``tsb``: Teunter, Syntetos and Babai (2011): demand probability and size smoothed separately; the probability
  keeps decaying when demand stops, which suits parts going obsolete.
"""
from __future__ import annotations

import numpy as np

WARMUP = 12  # months of history before the first forecast


def _init_mean(y: np.ndarray) -> np.ndarray:
    return np.nanmean(y[:WARMUP], axis=0)


def zero(y: np.ndarray) -> np.ndarray:
    f = np.zeros_like(y, dtype=float)
    f[:WARMUP] = np.nan
    return f


def naive(y: np.ndarray) -> np.ndarray:
    f = np.full_like(y, np.nan, dtype=float)
    f[1:] = y[:-1]
    f[:WARMUP] = np.nan
    return f


def moving_average(y: np.ndarray, window: int = 12) -> np.ndarray:
    f = np.full_like(y, np.nan, dtype=float)
    for t in range(WARMUP, len(y)):
        f[t] = np.nanmean(y[max(0, t - window):t], axis=0)
    return f


def ses(y: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    f = np.full_like(y, np.nan, dtype=float)
    level = _init_mean(y)
    for t in range(WARMUP, len(y)):
        f[t] = level
        level = level + alpha * (np.nan_to_num(y[t], nan=level) - level)
    # levels before the warm-up end are updated from month WARMUP onwards only (initialised on the warm-up mean)
    return f


def _croston_core(y: np.ndarray, alpha: float):
    t_len, n = y.shape
    size = np.full(n, np.nan)
    interval = np.full(n, np.nan)
    since = np.ones(n)
    out = np.full((t_len, n), np.nan)
    for t in range(t_len):
        if t >= WARMUP:
            out[t] = np.where(np.isnan(size), 0.0, size / interval)
        d = np.nan_to_num(y[t])
        hit = d > 0
        first = hit & np.isnan(size)
        size = np.where(first, d, size)
        interval = np.where(first, since, interval)
        upd = hit & ~first
        size = np.where(upd, size + alpha * (d - size), size)
        interval = np.where(upd, interval + alpha * (since - interval), interval)
        since = np.where(hit, 1, since + 1)
    return out


def croston(y: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    return _croston_core(y, alpha)


def sba(y: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    return _croston_core(y, alpha) * (1 - alpha / 2)


def tsb(y: np.ndarray, alpha: float = 0.1, beta: float = 0.1) -> np.ndarray:
    t_len, n = y.shape
    first = np.nan_to_num(y[:WARMUP])
    prob = (first > 0).mean(axis=0)
    with np.errstate(invalid="ignore"):
        size = np.where((first > 0).any(axis=0), first.sum(axis=0) / np.maximum((first > 0).sum(axis=0), 1), 0.0)
    out = np.full((t_len, n), np.nan)
    for t in range(WARMUP, t_len):
        out[t] = prob * size
        d = np.nan_to_num(y[t])
        hit = d > 0
        prob = prob + beta * (hit.astype(float) - prob)
        size = np.where(hit, np.where(size > 0, size + alpha * (d - size), d), size)
    return out


METHODS = {"zero": zero, "naive": naive, "moving_average": moving_average, "ses": ses, "croston": croston,
           "sba": sba, "tsb": tsb}
