from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, PEOPLE, EXCLUDE_FROM_LIFESTYLE
from src.parsers import load_person_data
from src.categoriser import get_spending_df
from src.forecaster import prepare_monthly_series
from src.charts import comparison_chart, category_bar, category_donut

st.set_page_config(page_title="Comparison", page_icon="🔍", layout="wide")
st.title("🔍 Kanokphan vs Yensa — comparison")


@st.cache_data(show_spinner="Loading…")
def load_all(d: str) -> dict:
    result = {}
    for p in PEOPLE:
        bank, cc = load_person_data(p, Path(d))
        result[p] = get_spending_df(bank, cc)
    return result


all_data = load_all(str(DATA_DIR))

with st.sidebar:
    st.header("Options")
    all_avail = sorted(set(
        cat for df in all_data.values() if not df.empty
        for cat in df["Category"].unique()
    ))
    # BUG-1 fix: filter defaults to options present
    safe_exc = [c for c in EXCLUDE_FROM_LIFESTYLE if c in all_avail]
    exclude_cats = st.multiselect("Exclude categories (both)", all_avail, default=safe_exc)

data = {
    p: (df[~df["Category"].isin(exclude_cats)] if not df.empty else df)
    for p, df in all_data.items()
}

st.subheader("Summary")
cols = st.columns(len(PEOPLE))
for person, col in zip(PEOPLE, cols):
    with col:
        df = data[person]
        st.markdown(f"**{person}**")
        if df.empty:
            st.warning("No data.")
            continue
        total   = df["Amount"].sum()
        n_mo    = df["YearMonth"].nunique()
        avg     = total / n_mo if n_mo else 0
        top_cat = df.groupby("Category")["Amount"].sum().idxmax()
        st.metric("Total spend",    f"฿{total:,.0f}")
        st.metric("Avg / month",    f"฿{avg:,.0f}")
        st.metric("Transactions",   f"{len(df):,}")
        st.metric("Months covered", n_mo)
        st.metric("Top category",   top_cat)

st.markdown("---")
st.subheader("Monthly spend — both people")
series_map = {
    p: prepare_monthly_series(df)
    for p, df in data.items() if not df.empty
}
if series_map:
    st.plotly_chart(comparison_chart(series_map), use_container_width=True)

st.markdown("---")
st.subheader("Category breakdown")
col1, col2 = st.columns(2)
for person, col in zip(PEOPLE, [col1, col2]):
    with col:
        st.markdown(f"**{person}**")
        df = data[person]
        if not df.empty:
            t1, t2 = st.tabs(["Donut", "Bar"])
            with t1:
                st.plotly_chart(category_donut(df), use_container_width=True)
            with t2:
                st.plotly_chart(category_bar(df, top_n=8), use_container_width=True)

st.markdown("---")
st.subheader("Category comparison table")
all_cats = sorted(set(
    cat for df in data.values() if not df.empty
    for cat in df["Category"].unique()
))
rows = []
for cat in all_cats:
    row: dict = {"Category": cat}
    for p in PEOPLE:
        df = data[p]
        val = df[df["Category"] == cat]["Amount"].sum() if not df.empty else 0
        row[p] = f"฿{val:,.0f}" if val > 0 else "—"
    rows.append(row)
if rows:
    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇ Download comparison table (CSV)",
        comp_df.to_csv(index=False).encode("utf-8"),
        "comparison_table.csv", "text/csv",
    )
