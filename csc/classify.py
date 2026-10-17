"""Demand classes.

Syntetos, Boylan and Croston (2005) classify a demand history with two numbers:

* ADI, the average interval between two demands (in months);
* CV², the squared coefficient of variation of the demand sizes, when there is demand.

With the usual cut-offs (ADI 1.32, CV² 0.49): smooth (frequent, regular), erratic (frequent, irregular sizes),
intermittent (rare, regular sizes), lumpy (rare and irregular). Parts with fewer than two demands cannot be
classified and are reported apart.

ABC is computed on units, because the data has no prices; with prices it should be computed on value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ADI_CUT, CV2_CUT = 1.32, 0.49
CLASSES = ["smooth", "erratic", "intermittent", "lumpy", "too few demands"]


def sbc(wide: pd.DataFrame) -> pd.DataFrame:
    y = wide.to_numpy(dtype=float)
    observed = ~np.isnan(y)
    pos = np.nan_to_num(y) > 0
    n_obs = observed.sum(axis=0)
    n_dem = pos.sum(axis=0)
    adi = np.where(n_dem > 0, n_obs / np.maximum(n_dem, 1), np.inf)
    cnt = pos.sum(axis=0)
    total = np.where(pos, y, 0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / cnt
        var = (np.where(pos, (np.nan_to_num(y) - mean) ** 2, 0)).sum(axis=0) / cnt
        cv2 = var / mean ** 2
    cls = np.select([n_dem < 2, (adi < ADI_CUT) & (cv2 < CV2_CUT), adi < ADI_CUT, cv2 < CV2_CUT],
                    ["too few demands", "smooth", "erratic", "intermittent"], default="lumpy")
    return pd.DataFrame({"part": wide.columns, "months_observed": n_obs, "months_with_demand": n_dem,
                         "adi": adi, "cv2": cv2, "demand_class": cls})


def abc(wide: pd.DataFrame, a: float = 0.80, b: float = 0.95) -> pd.Series:
    units = wide.fillna(0).sum().sort_values(ascending=False)
    share = units.cumsum() / units.sum()
    prev = share.shift(fill_value=0)
    return pd.Series(np.select([prev < a, prev < b], ["A", "B"], default="C"), index=units.index, name="abc")
