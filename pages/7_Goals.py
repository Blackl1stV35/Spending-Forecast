"""
Saving Goals — per-person monthly target + per-category budget caps.
"""
from __future__ import annotations
import sys
from datetime import date
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, PEOPLE, EXCLUDE_FROM_LIFESTYLE, CATEGORY_COLORS
from src.parsers import load_person_data
from src.categoriser import get_spending_df
from src.forecaster import prepare_monthly_series, ets_forecast
from src.charts import goals_progress_chart
from src.supabase_store import fetch_goals, upsert_goals, is_available as sb_available
import plotly.graph_objects as go

st.set_page_config(page_title="Goals", page_icon="🎯", layout="wide")
st.title("🎯 Saving goals")
st.caption("Set monthly targets and per-category caps — tracked against actual spending")


@st.cache_data(show_spinner="Loading…")
def load_all(d: str) -> dict:
    result = {}
    for p in PEOPLE:
        bank, cc = load_person_data(p, Path(d))
        result[p] = get_spending_df(bank, cc)
    return result


all_data = load_all(str(DATA_DIR))

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    person = st.selectbox("Person", PEOPLE)
    st.divider()
    if sb_available():
        st.success("Goals saved to Supabase", icon="✅")
    else:
        st.info("Supabase not configured — goals saved in session only.", icon="ℹ️")

# ── Load current goals ─────────────────────────────────────────────────────────
saved_goals: dict = {}
if sb_available():
    saved_goals = fetch_goals(person) or {}

if "goals" not in st.session_state:
    st.session_state.goals = {}
if person not in st.session_state.goals:
    st.session_state.goals[person] = saved_goals

current_goals = st.session_state.goals[person]

# ── Spending data for context ──────────────────────────────────────────────────
df = all_data[person]
series = pd.Series(dtype=float)
cat_avgs: dict = {}
avg_monthly = 0.0

if not df.empty:
    lifestyle = df[~df["Category"].isin(EXCLUDE_FROM_LIFESTYLE)]
    series    = prepare_monthly_series(lifestyle, clip=True)
    n_mo      = len(series)
    avg_monthly = float(series.mean()) if not series.empty else 0.0
    cat_totals  = lifestyle.groupby("Category")["Amount"].sum()
    cat_avgs    = {
        cat: round(amt / n_mo) if n_mo else 0
        for cat, amt in cat_totals.items()
    }

# ── Set Goals ──────────────────────────────────────────────────────────────────
st.subheader(f"Set goals for {person}")

col_form, col_preview = st.columns([1.2, 1])

with col_form:
    st.markdown("**Monthly savings target**")
    current_target = float(current_goals.get("monthly_savings_target", 0))
    new_target = st.number_input(
        "Monthly savings target (฿)", min_value=0.0, step=500.0,
        value=current_target,
        help="How much you want to spend in total each month (lifestyle only).",
    )

    if avg_monthly > 0:
        gap = avg_monthly - new_target
        delta_str = f"฿{abs(gap):,.0f} {'over' if gap > 0 else 'under'} current avg"
        st.caption(f"Current monthly avg: ฿{avg_monthly:,.0f} — {delta_str}")

    st.markdown("**Per-category budget caps** _(optional)_")
    st.caption("Leave at 0 to set no cap for that category.")

    avail_cats = sorted(cat_avgs.keys()) if cat_avgs else []
    cap_values: dict[str, float] = {}
    saved_caps = current_goals.get("category_caps", {})

    if avail_cats:
        cap_cols = st.columns(2)
        for i, cat in enumerate(avail_cats):
            with cap_cols[i % 2]:
                default_cap = float(saved_caps.get(cat, 0))
                cap = st.number_input(
                    cat,
                    min_value=0.0,
                    step=100.0,
                    value=default_cap,
                    key=f"cap_{person}_{cat}",
                    help=f"Current avg: ฿{cat_avgs.get(cat, 0):,.0f}/mo",
                )
                if cap > 0:
                    cap_values[cat] = cap
    else:
        st.info("Load spending data to see per-category suggestions.", icon="ℹ️")

    if st.button("💾 Save goals", type="primary"):
        st.session_state.goals[person] = {
            "monthly_savings_target": new_target,
            "category_caps":          cap_values,
        }
        if sb_available():
            success = upsert_goals(
                person=person,
                monthly_target=new_target,
                category_caps=cap_values,
                effective_month=date.today().replace(day=1),
            )
            if success:
                st.success("Goals saved to Supabase.", icon="✅")
            else:
                st.warning("Saved locally (Supabase write failed).", icon="⚠️")
        else:
            st.success("Goals saved for this session.", icon="✅")
        st.rerun()

with col_preview:
    st.markdown("**Goal summary**")
    goal_target = float(st.session_state.goals[person].get("monthly_savings_target", 0))
    if goal_target > 0 and avg_monthly > 0:
        pct_used = min(avg_monthly / goal_target, 1.5)
        bar_color = "#1D9E75" if avg_monthly <= goal_target else "#E24B4A"
        st.markdown(
            f'<div style="background:var(--color-background-secondary);border-radius:8px;padding:14px 16px">'
            f'<div style="font-size:12px;color:var(--color-text-secondary);margin-bottom:8px">Monthly target</div>'
            f'<div style="font-size:22px;font-weight:500">฿{goal_target:,.0f}</div>'
            f'<div style="margin:10px 0 4px;height:8px;background:var(--color-border-tertiary);border-radius:4px">'
            f'<div style="height:8px;width:{min(pct_used*100,100):.0f}%;background:{bar_color};border-radius:4px;transition:width .3s"></div>'
            f'</div>'
            f'<div style="font-size:11px;color:var(--color-text-tertiary)">'
            f'Current avg: ฿{avg_monthly:,.0f} ({pct_used*100:.0f}% of target)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Set a monthly target above to see progress.")

    if cap_values:
        st.markdown("**Active category caps**")
        for cat, cap in cap_values.items():
            actual = cat_avgs.get(cat, 0)
            over   = actual > cap
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:12px;padding:4px 0;color:{"#E24B4A" if over else "var(--color-text-secondary)"}">'
                f'<span>{cat}</span>'
                f'<span>฿{actual:,.0f} / ฿{cap:,.0f}</span></div>',
                unsafe_allow_html=True,
            )

# ── Progress chart ─────────────────────────────────────────────────────────────
if not series.empty:
    st.markdown("---")
    st.subheader("Actual spend vs target")
    g = st.session_state.goals[person]
    target = float(g.get("monthly_savings_target", 0))
    st.plotly_chart(goals_progress_chart(series, target, person), use_container_width=True)

    # ── Projection ────────────────────────────────────────────────────────────
    if target > 0 and len(series) >= 2:
        st.markdown("---")
        st.subheader("Projection")
        fc, lo, hi = ets_forecast(series, n_months=6)
        if fc is not None:
            months_on_target = sum(1 for v in fc.values if v <= target)
            over_by = [v - target for v in fc.values if v > target]
            avg_over = sum(over_by) / len(over_by) if over_by else 0

            p1, p2, p3 = st.columns(3)
            p1.metric("Months on target (next 6)", f"{months_on_target} / 6")
            p2.metric("Avg over-target amount",
                      f"฿{avg_over:,.0f}" if avg_over > 0 else "On track ✓")
            next_fc = float(fc.iloc[0])
            p3.metric("Next month forecast", f"฿{next_fc:,.0f}",
                      delta=f"฿{next_fc - target:+,.0f} vs target",
                      delta_color="inverse" if next_fc > target else "normal")

# ── Category caps vs actual table ─────────────────────────────────────────────
caps = st.session_state.goals[person].get("category_caps", {})
if caps and cat_avgs:
    st.markdown("---")
    st.subheader("Category cap compliance")
    rows = []
    for cat, cap in caps.items():
        actual = cat_avgs.get(cat, 0)
        rows.append({
            "Category": cat,
            "Monthly cap": f"฿{cap:,.0f}",
            "Avg actual":  f"฿{actual:,.0f}",
            "Status":      "✅ Within cap" if actual <= cap else f"❌ Over by ฿{actual-cap:,.0f}",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
