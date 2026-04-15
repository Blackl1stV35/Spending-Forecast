"""
Home page — Spending Pattern & Forecast Dashboard

Entry point for Streamlit Cloud deployment.
Run locally with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATA_DIR, PEOPLE, PERSON_COLORS
from src.parsers import load_person_data
from src.categoriser import get_spending_df
from src.forecaster import prepare_monthly_series, PROPHET_AVAILABLE
from src.charts import comparison_chart

st.set_page_config(
    page_title="Spending forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading data…")
def load_all(data_dir_str: str) -> dict:
    result = {}
    for person in PEOPLE:
        bank_df, cc_df = load_person_data(person, Path(data_dir_str))
        spending = get_spending_df(bank_df, cc_df)
        result[person] = {"bank": bank_df, "cc": cc_df, "spending": spending}
    return result


data = load_all(str(DATA_DIR))

st.title("📊 Spending pattern & forecast")
st.caption("KBank statement analysis — Kanokphan & Yensa")

if not PROPHET_AVAILABLE:
    st.info("Prophet is not installed — the Prophet model will be skipped. Add `prophet` to requirements.txt to enable it.", icon="ℹ️")

st.markdown("---")

cols = st.columns(2)
any_data = False

for person, col in zip(PEOPLE, cols):
    df = data[person]["spending"]
    color = PERSON_COLORS[person]

    with col:
        st.markdown(f"### {person}")

        if df.empty:
            st.warning(
                f"No data found. Add CSV files to:\n"
                f"`data/{person}/BankAccount/*.csv`\n"
                f"`data/{person}/CreditCard/*.csv`",
            )
            continue

        any_data = True

        # Exclude investment flows for lifestyle summary
        lifestyle = df[~df["Category"].isin(["Investment", "Incoming Transfer"])]
        total = lifestyle["Amount"].sum()
        n_months = lifestyle["YearMonth"].nunique()
        avg = total / n_months if n_months else 0
        top_cat = lifestyle.groupby("Category")["Amount"].sum().idxmax() if not lifestyle.empty else "—"
        n_tx = len(lifestyle)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", f"฿{total:,.0f}")
        m2.metric("Avg / mo", f"฿{avg:,.0f}")
        m3.metric("Tx count", f"{n_tx:,}")
        m4.metric("Top cat.", top_cat)

        monthly = prepare_monthly_series(lifestyle)
        if not monthly.empty:
            st.line_chart(
                monthly.rename("Monthly spend (฿)"),
                color=color,
                height=160,
                use_container_width=True,
            )

st.markdown("---")

if any_data:
    st.subheader("Side-by-side monthly comparison")
    series_map = {}
    for person in PEOPLE:
        df = data[person]["spending"]
        if not df.empty:
            lifestyle = df[~df["Category"].isin(["Investment", "Incoming Transfer"])]
            series_map[person] = prepare_monthly_series(lifestyle)

    if series_map:
        st.plotly_chart(comparison_chart(series_map), use_container_width=True)

st.markdown("---")
st.markdown(
    """
    **Navigate** using the sidebar:

    | Page | What you'll find |
    |------|-----------------|
    | Kanokphan | Bank + CC analysis, heatmap, transaction table |
    | Yensa | Bank + CC analysis, heatmap, transaction table |
    | Comparison | Side-by-side metrics and category table |
    | Forecasting | ETS · ARIMA · Ridge · Prophet with CV metrics |
    """
)
