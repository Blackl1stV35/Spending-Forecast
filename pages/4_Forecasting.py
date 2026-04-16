"""
Forecasting — multi-model spending forecast with cross-validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, PEOPLE, EXCLUDE_FROM_LIFESTYLE
from src.parsers import load_person_data
from src.categoriser import get_spending_df
from src.forecaster import (
    PROPHET_AVAILABLE,
    prepare_monthly_series,
    run_all_forecasts,
    leave_n_out_cv,
)
from src.charts import forecast_chart

st.set_page_config(page_title="Forecasting", page_icon="🔮", layout="wide")
st.title("🔮 Spending forecast")


@st.cache_data(show_spinner="Loading…")
def load_all(d: str) -> dict:
    result = {}
    for p in PEOPLE:
        bank, cc = load_person_data(p, Path(d))
        result[p] = get_spending_df(bank, cc)
    return result


all_data = load_all(str(DATA_DIR))

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    person = st.selectbox("Person", PEOPLE)
    n_months = st.slider("Forecast horizon (months)", min_value=1, max_value=6, value=3)

    df = all_data[person]
    default_exc = EXCLUDE_FROM_LIFESTYLE
    if not df.empty:
        avail_cats = sorted(df["Category"].unique().tolist())
        exclude_cats = st.multiselect("Exclude categories", avail_cats, default=[c for c in default_exc if c in avail_cats])
    else:
        exclude_cats = default_exc

    st.divider()
    st.markdown("**Available models**")
    st.markdown("- Rolling average ✓")
    st.markdown("- ETS / Holt ✓")
    st.markdown("- ARIMA(1,1,1) ✓")
    st.markdown("- Ridge regression ✓")
    if PROPHET_AVAILABLE:
        st.markdown("- Prophet ✓")
    else:
        st.markdown("- Prophet ✗ _(not installed)_")

# ── Main ───────────────────────────────────────────────────────────────────────
df = all_data[person]

if df.empty:
    st.warning(f"No data loaded for {person}. Add CSV files to the data/ folder.")
    st.stop()

series = prepare_monthly_series(df, exclude_categories=exclude_cats)

if len(series) < 24:
    st.info(
        f"Running **baseline models only** (Rolling avg + ETS) — "
        f"{len(series)} months available, need 24 for full suite.",
        icon="ℹ️",
    )

if len(series) < 3:
    st.warning(f"Need at least 3 months of data (found {len(series)}). Add more CSV files.")
    st.stop()

st.subheader(f"{person} — {n_months}-month spending forecast")
st.caption(f"Training on {len(series)} monthly data points · excludes: {', '.join(exclude_cats) or 'none'}")

# ── Run models ────────────────────────────────────────────────────────────────
with st.spinner("Running forecast models…"):
    forecasts = run_all_forecasts(series, n_months)

# ── Forecast chart ─────────────────────────────────────────────────────────────
st.plotly_chart(forecast_chart(series, forecasts, person), use_container_width=True)

st.markdown("---")

# ── CV evaluation ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1.6, 1])

with col1:
    n_test = min(3, max(1, len(series) // 4))
    st.subheader(f"Model evaluation — leave-{n_test}-out CV")
    with st.spinner("Running cross-validation…"):
        cv = leave_n_out_cv(series, n_test=n_test)

    if not cv.empty:
        best = cv.iloc[0]["Model"]
        st.success(f"Best model by MAE: **{best}**")
        st.dataframe(cv, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data for cross-validation (need ≥6 months).")

with col2:
    st.subheader("Next-month point estimates")
    for name, (fc, lo, hi) in forecasts.items():
        if fc is not None and len(fc) > 0:
            val = float(fc.iloc[0])
            lo_v = float(lo.iloc[0]) if lo is not None else val * 0.85
            hi_v = float(hi.iloc[0]) if hi is not None else val * 1.15
            st.metric(
                name,
                f"฿{val:,.0f}",
                delta=f"95% CI  ฿{lo_v:,.0f} – ฿{hi_v:,.0f}",
                delta_color="off",
            )

st.markdown("---")

# ── Historical series table ────────────────────────────────────────────────────
with st.expander("Historical monthly series (training data)"):
    hist_df = series.reset_index()
    hist_df.columns = ["Month", "Amount (฿)"]
    hist_df["Month"] = hist_df["Month"].dt.strftime("%b %Y")
    hist_df["Amount (฿)"] = hist_df["Amount (฿)"].map("฿{:,.0f}".format)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

# ── Forecast values table ──────────────────────────────────────────────────────
with st.expander("Forecast values by model"):
    rows = []
    for name, (fc, lo, hi) in forecasts.items():
        if fc is None:
            continue
        for i, (date, val) in enumerate(fc.items()):
            lo_v = float(lo.iloc[i]) if lo is not None else None
            hi_v = float(hi.iloc[i]) if hi is not None else None
            rows.append({
                "Model": name,
                "Month": date.strftime("%b %Y"),
                "Forecast (฿)": round(val),
                "Lower 95% CI": round(lo_v) if lo_v is not None else "—",
                "Upper 95% CI": round(hi_v) if hi_v is not None else "—",
            })
    if rows:
        fc_df = pd.DataFrame(rows)
        st.dataframe(fc_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download forecast table (CSV)",
            fc_df.to_csv(index=False).encode("utf-8"),
            f"{person}_forecast.csv",
            "text/csv",
        )
