"""Interactive business pricing / profitability demo (run: streamlit run app.py)."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from pricing_engine import (
    SQL_SUMMARY, add_financials, fit_demand_model, make_scenario,
    sql_product_summary, validate_sales,
)

st.set_page_config(page_title="Pricing & Profitability Engine", page_icon="📈", layout="wide")
st.title("Pricing & Profitability Decision Engine")
st.caption("Portfolio demo | Fictional sample data | All scenario results are estimates, not guaranteed outcomes")

sample_path = Path(__file__).resolve().parent / "data" / "sample_sales.csv"
upload = st.file_uploader("Optional: upload product-level monthly sales CSV", type="csv")

try:
    raw = pd.read_csv(upload if upload is not None else sample_path)
    df = validate_sales(raw)
except (ValueError, FileNotFoundError, pd.errors.ParserError) as exc:
    st.error(f"Could not load sales data: {exc}")
    st.info("Run `python generate_data.py` to create sample data. See README.md for required CSV columns.")
    st.stop()

financials = add_financials(df)
summary = sql_product_summary(df)

st.subheader("1. Business overview (SQL analytics)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Revenue", f"₹{summary['revenue'].sum():,.0f}")
col2.metric("Contribution profit*", f"₹{summary['contribution_profit'].sum():,.0f}")
col3.metric("Units sold", f"{int(summary['units_sold'].sum()):,}")
all_revenue = summary["revenue"].sum()
all_profit = summary["contribution_profit"].sum()
col4.metric("Contribution margin", f"{100 * all_profit / all_revenue:.1f}%" if all_revenue else "0%")
st.caption("*Contribution profit = revenue − cost of goods sold − marketing spend; other overheads and taxes are excluded.")
st.dataframe(
    summary.rename(columns={
        "product": "Product", "units_sold": "Units", "revenue": "Revenue (₹)",
        "contribution_profit": "Contribution profit (₹)",
        "contribution_margin_pct": "Contribution margin (%)",
    }).round(2),
    hide_index=True, use_container_width=True,
)
with st.expander("Show the actual SQL aggregation query"):
    st.code(SQL_SUMMARY, language="sql")

st.subheader("2. Historical performance")
selected = st.selectbox("Choose a product", sorted(df["product"].unique()))
product_data = financials.loc[financials["product"] == selected].sort_values("date")
plot_data = product_data.melt(
    id_vars="date", value_vars=["revenue", "contribution_profit"],
    var_name="Metric", value_name="INR",
)
fig = px.line(plot_data, x="date", y="INR", color="Metric", markers=True,
              title=f"{selected}: monthly revenue and contribution profit")
st.plotly_chart(fig, use_container_width=True)

st.subheader("3. Pricing what-if simulator")
if len(product_data) < 12:
    st.warning("At least 12 months of product data are required for a demand model.")
    st.stop()

fit = fit_demand_model(product_data)
latest = product_data.iloc[-1]
slider_col, info_col = st.columns([2, 1])
with slider_col:
    pct = st.slider("Simulated price change (%)", min_value=-10, max_value=10, value=0, step=1)
with info_col:
    st.metric("Holdout MAE (units)", f"{fit.holdout_mae_units:,.1f}")
    st.caption(f"Measured on the last {fit.test_months} months before refitting on full history. Naive baseline MAE: {fit.baseline_mae_units:,.1f} units.")

if fit.price_elasticity >= 0:
    st.warning("The fitted price coefficient is non-negative. This may reflect confounding or weak data; interpret scenarios cautiously.")

base = make_scenario(
    fit, float(latest["selling_price"]), float(latest["unit_cost"]),
    float(latest["marketing_spend"]), latest["date"], 0,
)
scenario = make_scenario(
    fit, float(latest["selling_price"]), float(latest["unit_cost"]),
    float(latest["marketing_spend"]), latest["date"], pct,
)
left, mid, right = st.columns(3)
left.metric("Simulated selling price", f"₹{scenario['selling_price']:,.0f}",
            f"{pct:+d}% vs latest")
mid.metric("Estimated units", f"{scenario['estimated_units']:,.0f}",
           f"{scenario['estimated_units'] - base['estimated_units']:+,.0f} vs baseline")
right.metric("Estimated contribution profit", f"₹{scenario['estimated_contribution_profit']:,.0f}",
             f"₹{scenario['estimated_contribution_profit'] - base['estimated_contribution_profit']:+,.0f} vs baseline")

options = pd.DataFrame([
    make_scenario(
        fit, float(latest["selling_price"]), float(latest["unit_cost"]),
        float(latest["marketing_spend"]), latest["date"], change,
    )
    for change in range(-10, 11, 2)
])
profit_fig = px.line(
    options, x="price_change_pct", y="estimated_contribution_profit", markers=True,
    title="Estimated monthly contribution profit at different prices",
    labels={"price_change_pct": "Price change (%)", "estimated_contribution_profit": "Contribution profit (₹)"},
)
st.plotly_chart(profit_fig, use_container_width=True)

best = options.loc[options["estimated_contribution_profit"].idxmax()]
st.write(
    f"**Illustrative scenario comparison:** Among the tested price changes from −10% to +10%, "
    f"the model estimates its highest contribution profit at **{best['price_change_pct']:+.0f}%** "
    f"(₹{best['estimated_contribution_profit']:,.0f}). This is not a verified optimal price."
)
with st.expander("Model limitations and explanation"):
    st.markdown(
        "- Demand is estimated with an interpretable **log-linear regression** using selling price, "
        "marketing spend, and month-of-year seasonality.\n"
        "- Price coefficient approximates *price elasticity*: "
        f"**{fit.price_elasticity:.2f}** for this product.\n"
        "- The newest months are held out for validation; predictions are not causal proof.\n"
        "- Scenario analysis keeps marketing spend and unit cost constant and only varies price.\n"
        "- Tested changes are limited to ±10%; results beyond the historical price range are unreliable.\n"
        "- Sample data is fictional, so no real-world revenue improvement is claimed."
    )
