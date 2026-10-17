"""Component stock cockpit. Run with: streamlit run app/dashboard.py"""
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
st.set_page_config(page_title="Component stock cockpit", layout="wide")


@st.cache_data
def csv(name):
    return pd.read_csv(RES / name)


S = json.loads((RES / "summary.json").read_text())
PAGES = ["Overview", "Forecast accuracy", "Stock policy backtest", "Dormant parts", "Purchase recommendations",
         "Use your own data"]
page = st.sidebar.radio("Page", PAGES)
st.sidebar.caption(f"Real demand history: {S['data']['parts']:,} parts, {S['data']['first_month']} to "
                   f"{S['data']['last_month']} (carparts, Hyndman et al. 2008).")

if page == "Overview":
    st.title("Slow-moving parts: what the history says")
    d = S["data"]
    c = st.columns(4)
    c[0].metric("Parts", f"{d['parts']:,}")
    c[1].metric("Months with zero demand", f"{d['share_of_zero_months']:.0%}")
    c[2].metric("No demand in the last 12 months", S["no_demand_last_12_months"])
    c[3].metric("History stops early", d["history_stops_early"])
    cls = pd.Series(S["classes_full_history"]).rename_axis("class").reset_index(name="parts")
    st.altair_chart(alt.Chart(cls).mark_bar().encode(x="parts:Q", y=alt.Y("class:N", sort="-x"),
                                                     tooltip=["class", "parts"]), width="stretch")
    st.dataframe(csv("demand_by_year.csv"), width="stretch", hide_index=True)

elif page == "Forecast accuracy":
    st.title("Forecast accuracy, one month ahead, months 25 to 51")
    a = csv("forecast_accuracy.csv")
    cls = st.selectbox("Demand class", sorted(a["demand_class"].unique()), index=list(
        sorted(a["demand_class"].unique())).index("intermittent"))
    st.dataframe(a[a["demand_class"] == cls], width="stretch", hide_index=True)
    st.caption("Relative MSE below 1 beats the 12-month moving average. Bias above 0 builds stock.")

elif page == "Stock policy backtest":
    st.title("Fill rate bought per unit of stock")
    s = csv("stock_policy_backtest.csv")
    c1, c2 = st.columns(2)
    lead = c1.selectbox("Lead time (months)", sorted(s["lead_time_months"].unique()), index=1)
    cls = c2.selectbox("Demand class", ["all"] + sorted(set(s["demand_class"]) - {"all"}))
    v = s[(s["lead_time_months"] == lead) & (s["demand_class"] == cls)]
    st.altair_chart(alt.Chart(v).mark_line(point=True).encode(
        x=alt.X("stock_in_months_of_demand:Q", title="stock in months of demand"),
        y=alt.Y("fill_rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)), color="method:N",
        tooltip=["method", "target", alt.Tooltip("fill_rate:Q", format=".1%"), "stock_in_months_of_demand"]),
        width="stretch")
    st.dataframe(v, width="stretch", hide_index=True)

elif page == "Dormant parts":
    st.title("Is '12 months without demand' a good obsolescence rule?")
    st.dataframe(csv("dormant_rule_backtest.csv"), width="stretch", hide_index=True)
    st.caption("Parts flagged in January 2000, then followed for the next 27 months.")

elif page == "Purchase recommendations":
    st.title("Purchase recommendations")
    st.warning("Stock levels on this page are simulated (the data set has no stock). With real stock, use the "
               "last page or the command line.")
    r = csv("recommendations_demo.csv")
    flag = st.selectbox("Show", ["parts to order", "all", "flagged for review"])
    if flag == "parts to order":
        r = r[r["proposed_order"] > 0]
    elif flag == "flagged for review":
        r = r[r["flags"].notna() & (r["flags"] != "")]
    st.dataframe(r, width="stretch", hide_index=True)

else:
    st.title("Recommendations on your own data")
    st.write("Demand file: columns `part`, `month` (YYYY-MM), `qty`. Stock file (optional): `part`, `on_hand`, "
             "`on_order`.")
    dem = st.file_uploader("Demand history (CSV)", type="csv")
    sto = st.file_uploader("Stock (CSV, optional)", type="csv")
    lead = st.number_input("Lead time (months)", 1, 24, 3)
    target = st.slider("Service target", 0.80, 0.99, 0.95)
    if dem is not None:
        from csc.data import from_long
        from csc.recommend import recommend
        d = pd.read_csv(dem, dtype={"part": str, "month": str})
        if not {"part", "month", "qty"} <= set(d.columns):
            st.error("The demand file needs the columns part, month and qty.")
            st.stop()
        stock = pd.read_csv(sto, dtype={"part": str}) if sto is not None else None
        try:
            rec = recommend(from_long(d.assign(qty=d["qty"].clip(lower=0))), int(lead), float(target), stock)
        except ValueError as err:
            st.error(str(err))
            st.stop()
        st.dataframe(rec, width="stretch", hide_index=True)
        st.download_button("Download", rec.to_csv(index=False), "recommendations.csv")
