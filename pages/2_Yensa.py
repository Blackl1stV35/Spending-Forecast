"""
Yensa — individual spending analysis page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, EXCLUDE_FROM_LIFESTYLE
from src.parsers import load_person_data, load_from_uploads
from src.categoriser import get_spending_df
from src.charts import (
    category_bar,
    category_donut,
    category_monthly_stack,
    calendar_heatmap,
    monthly_trend_chart,
    waterfall_chart,
)

PERSON = "Yensa"

st.set_page_config(page_title=f"{PERSON} — Spending", page_icon="📊", layout="wide")
st.title(f"📊 {PERSON} — spending analysis")


@st.cache_data(show_spinner="Parsing files…")
def _load(data_dir_str: str):
    return load_person_data(PERSON, Path(data_dir_str))


bank_df, cc_df = _load(str(DATA_DIR))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload data (optional)")
    st.caption("Overrides files in data/ folder")
    up_bank = st.file_uploader("Bank statement CSVs", type="csv", accept_multiple_files=True, key="y_bank")
    up_cc = st.file_uploader("Credit card CSVs", type="csv", accept_multiple_files=True, key="y_cc")

    if up_bank or up_cc:
        bank_df, cc_df = load_from_uploads(up_bank or [], up_cc or [])
        st.success(f"Loaded {len(up_bank or [])} bank + {len(up_cc or [])} CC file(s)")

    st.divider()
    st.header("Filters")
    sources = st.multiselect("Data source", ["bank", "cc"], default=["bank", "cc"])
    _tmp = get_spending_df(bank_df, cc_df)
    all_cats = sorted(_tmp["Category"].unique().tolist()) if not _tmp.empty else []
    # Filter out any default categories that aren't in the available options for Yensa
    valid_defaults = [cat for cat in EXCLUDE_FROM_LIFESTYLE if cat in all_cats]
    
    # Pass the filtered list as the default
    exclude_cats = st.multiselect(
        "Exclude categories", 
        all_cats, 
        default=valid_defaults
    )

    date_col = st.container()

# ── Load & filter ─────────────────────────────────────────────────────────────
spending = get_spending_df(bank_df, cc_df)

if spending.empty:
    st.warning(
        f"No data found for {PERSON}.\n\n"
        f"Place CSV files in `data/{PERSON}/BankAccount/` and `data/{PERSON}/CreditCard/`  \n"
        f"or upload them in the sidebar."
    )
    st.stop()

with date_col:
    d_min = spending["Date"].min().date()
    d_max = spending["Date"].max().date()
    date_range = st.date_input("Date range", value=(d_min, d_max), min_value=d_min, max_value=d_max)

filtered = spending[spending["source"].isin(sources)]
if exclude_cats:
    filtered = filtered[~filtered["Category"].isin(exclude_cats)]
if len(date_range) == 2:
    filtered = filtered[
        (filtered["Date"] >= pd.Timestamp(date_range[0]))
        & (filtered["Date"] <= pd.Timestamp(date_range[1]))
    ]

if filtered.empty:
    st.info("No transactions match the current filters.")
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
total = filtered["Amount"].sum()
n_months = filtered["YearMonth"].nunique()
avg = total / n_months if n_months else 0
top_cat = filtered.groupby("Category")["Amount"].sum().idxmax()
n_tx = len(filtered)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total spend", f"฿{total:,.0f}")
c2.metric("Avg / month", f"฿{avg:,.0f}")
c3.metric("Transactions", f"{n_tx:,}")
c4.metric("Months covered", n_months)
c5.metric("Top category", top_cat)

st.info(
    "💡 **Note on Yensa's data:** Bank transactions are mostly ATM withdrawals "
    "(cash-economy). The credit card captures the main visible spending categories "
    "(Groceries, Fuel, Home). Total lifestyle spend is likely higher than recorded here.",
    icon="ℹ️",
)

st.markdown("---")

# ── Monthly trend ─────────────────────────────────────────────────────────────
st.subheader("Monthly spending trend")
t1, t2, t3 = st.tabs(["Total", "Bank vs CC", "By category"])
with t1:
    st.plotly_chart(monthly_trend_chart(filtered, PERSON, split_source=False), use_container_width=True)
with t2:
    st.plotly_chart(monthly_trend_chart(filtered, PERSON, split_source=True), use_container_width=True)
with t3:
    st.plotly_chart(category_monthly_stack(filtered, PERSON), use_container_width=True)

st.plotly_chart(
    waterfall_chart(
        filtered.groupby("YearMonth")["Amount"].sum().sort_index(),
        PERSON,
    ),
    use_container_width=True,
)

# ── Category breakdown ────────────────────────────────────────────────────────
st.subheader("Category breakdown")
col1, col2 = st.columns([1, 1.4])
with col1:
    st.plotly_chart(category_donut(filtered), use_container_width=True)
with col2:
    st.plotly_chart(category_bar(filtered), use_container_width=True)

# ── Heatmap ───────────────────────────────────────────────────────────────────
st.subheader("Spend heatmap (day of week × ISO week)")
st.plotly_chart(calendar_heatmap(filtered, PERSON), use_container_width=True)

# ── Transaction table ─────────────────────────────────────────────────────────
st.subheader("Transaction detail")

search = st.text_input("Search merchant / category", "")
display = filtered.copy()
if search:
    mask = (
        display["Merchant"].str.contains(search, case=False, na=False)
        | display["Category"].str.contains(search, case=False, na=False)
    )
    display = display[mask]

display_cols = ["Date", "Amount", "Merchant", "Category", "source"]
show = display[display_cols].sort_values("Date", ascending=False).copy()
show["Amount"] = show["Amount"].map("฿{:,.2f}".format)
st.dataframe(show, use_container_width=True, hide_index=True)

st.download_button(
    "⬇ Download filtered data (CSV)",
    display[display_cols].to_csv(index=False).encode("utf-8"),
    f"{PERSON}_spending_filtered.csv",
    "text/csv",
)
