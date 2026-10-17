# Data

`raw/carparts.rda`: monthly sales of 2,674 slow-moving car spare parts, January 1998 to March 2002 (51 months).
Source: Hyndman, Koehler, Ord and Snyder, *Forecasting with exponential smoothing: the state space approach*
(Springer, 2008), distributed in the R package `expsmooth` (GPL, from github.com/cran/expsmooth).
SHA-256: `81f1d1f1785278e376651ff447b6b0e0a2cf763373f01d61fe04fad0b7f20b0c` (checked when loading).

165 parts have no values after a certain month (their history stops early); they are kept in the descriptive
figures and left out of the backtests, which need the full 51 months.
