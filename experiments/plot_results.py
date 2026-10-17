"""Figures for the README, from results/."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
COL = {"sba_normal": "#b23a3a", "ses_normal": "#1f3b5c", "empirical": "#b08d3c"}
LABEL = {"sba_normal": "SBA + normal safety stock", "ses_normal": "SES + normal safety stock",
         "empirical": "empirical 95th percentile"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def frontier():
    s = pd.read_csv(RES / "stock_policy_backtest.csv")
    s = s[(s["demand_class"] == "all") & (s["lead_time_months"] == 3)]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m, g in s.groupby("method"):
        g = g.sort_values("target")
        ax.plot(g["stock_in_months_of_demand"], g["fill_rate"] * 100, "o-", color=COL[m], label=LABEL[m])
        for _, r in g.iterrows():
            ax.annotate(f"{r['target']:.0%}", (r["stock_in_months_of_demand"], r["fill_rate"] * 100),
                        textcoords="offset points", xytext=(4, -10), fontsize=7, color=COL[m])
    ax.set_xlabel("average stock, in months of demand")
    ax.set_ylabel("fill rate (%)")
    ax.set_title("2,509 parts, lead time 3 months: service bought per unit of stock")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_stock_vs_fill_rate.png", dpi=150)


def accuracy():
    a = pd.read_csv(RES / "forecast_accuracy.csv")
    a = a[a["demand_class"] == "intermittent"].set_index("method").loc[
        ["zero", "naive", "moving_average", "ses", "croston", "sba", "tsb"]]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].barh(a.index, a["relative_mse"], color="#1f3b5c")
    ax[0].axvline(1, color="grey", lw=1, ls="--")
    ax[0].set_title("Large misses: MSE relative to the moving average")
    ax[1].barh(a.index, a["bias"] * 100, color=["#b23a3a" if b > 0.3 else "#9aa5b1" for b in a["bias"]])
    ax[1].axvline(0, color="grey", lw=1)
    ax[1].set_title("Bias: forecast minus demand (%)")
    for x in ax:
        x.invert_yaxis()
    fig.suptitle("Intermittent parts (1,579), 27 months forecast one month ahead", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_forecast_accuracy.png", dpi=150)


def demand_trend():
    d = pd.read_csv(RES / "demand_by_year.csv")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(d["year"].astype(str), d["units_per_month"], color="#1f3b5c")
    for i, v in enumerate(d["units_per_month"]):
        ax.text(i, v + 20, f"{v:,.0f}", ha="center", fontsize=8)
    ax.set_ylabel("units per month, all parts")
    ax.set_title("Demand fell every year (2002: January to March)")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_demand_trend.png", dpi=150)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    frontier(), accuracy(), demand_trend()
