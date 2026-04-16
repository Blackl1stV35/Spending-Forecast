"""
Categorisation panel — hybrid manual + Groq LLM review.

Workflow:
  1. App scans all transactions still in 'Other' across both people.
  2. Unique merchants are sent to Groq in batches → structured suggestions.
  3. Human reviews each row: accept LLM suggestion, pick override, or skip.
  4. Approved mappings are saved to data/manual_overrides.json.
  5. All analysis pages pick up the new mappings on next data load.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR, PEOPLE
from src.parsers import load_person_data
from src.categoriser import get_spending_df
from src.groq_classifier import (
    GROQ_AVAILABLE,
    CATEGORY_LIST,
    extract_other_merchants,
    suggest_categories,
)
from src.overrides_store import (
    bulk_upsert,
    delete,
    load_overrides,
    override_stats,
    upsert,
)

st.set_page_config(
    page_title="Categorise — Other",
    page_icon="🏷️",
    layout="wide",
)

st.title("🏷️ Merchant categorisation panel")
st.caption("Resolve 'Other' merchants using Groq LLM suggestions + manual review")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — API key + options
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Groq API")

    # Key can come from: sidebar input → st.secrets → env var
    secret_key = ""
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret_key = os.environ.get("GROQ_API_KEY", "")

    api_key_input = st.text_input(
        "API key",
        value="",
        type="password",
        placeholder="gsk_... (leave blank to use server key)",
        help="Get a free key at console.groq.com. Leave blank if a server key is configured.",
    )
    api_key = api_key_input.strip() or secret_key

    if not GROQ_AVAILABLE:
        st.error("groq package not installed.\n\nRun: `pip install groq`")
    elif api_key:
        st.success("API key set", icon="✓")
    else:
        st.warning("Enter a Groq API key to enable LLM suggestions.")

    st.divider()

    model_choice = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
        help="70b = best accuracy. 8b = fastest.",
    )
    min_confidence = st.slider(
        "Auto-accept confidence threshold",
        min_value=0.70,
        max_value=1.00,
        value=0.90,
        step=0.01,
        help="Suggestions above this threshold will be highlighted for bulk accept.",
    )
    person_filter = st.multiselect(
        "Show people",
        PEOPLE,
        default=PEOPLE,
    )

    st.divider()
    st.header("Override store")
    overrides_now = load_overrides()
    stats = override_stats(overrides_now)
    st.metric("Total overrides", stats["total"])
    st.caption(
        f"Manual: {stats['manual']}  "
        f"LLM accepted: {stats['llm_accepted']}  "
        f"LLM auto: {stats.get('llm_auto', 0)}"
    )
    if st.button("🗑 Clear ALL overrides", type="secondary"):
        from src.config import DATA_DIR
        p = DATA_DIR / "manual_overrides.json"
        if p.exists():
            p.unlink()
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading transactions…")
def _load(d: str) -> dict:
    result = {}
    for p in PEOPLE:
        bank, cc = load_person_data(p, Path(d))
        result[p] = get_spending_df(bank, cc)
    return result


all_data = _load(str(DATA_DIR))

# Combine selected people
frames = []
for p in person_filter:
    df = all_data.get(p, pd.DataFrame())
    if not df.empty:
        df = df.copy()
        df["_person"] = p
        frames.append(df)

if not frames:
    st.warning("No data loaded. Add CSV files to the data/ folder.")
    st.stop()

combined_df = pd.concat(frames, ignore_index=True)
other_summary = extract_other_merchants(combined_df)

# ─────────────────────────────────────────────────────────────────────────────
# Summary banner
# ─────────────────────────────────────────────────────────────────────────────
total_tx = len(combined_df)
other_tx = len(combined_df[combined_df["Category"] == "Other"])
other_pct = other_tx / total_tx * 100 if total_tx else 0
other_merchants_n = len(other_summary)
overrides_applied = stats["total"]
resolved_pct = (
    (1 - other_tx / (total_tx or 1)) * 100
)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Total transactions", f"{total_tx:,}")
col_b.metric(
    "Still in 'Other'",
    f"{other_tx:,}",
    delta=f"{other_pct:.1f}% of total",
    delta_color="inverse",
)
col_c.metric("Unique 'Other' merchants", other_merchants_n)
col_d.metric("Overrides saved", overrides_applied)

if other_tx == 0:
    st.success("All transactions are categorised! Nothing left in 'Other'.", icon="✅")
    st.stop()

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# LLM suggestion panel
# ─────────────────────────────────────────────────────────────────────────────

# Session state for suggestions
if "llm_suggestions" not in st.session_state:
    st.session_state.llm_suggestions = {}
if "pending_approvals" not in st.session_state:
    st.session_state.pending_approvals = {}

merchant_list = other_summary["Merchant"].tolist()

llm_col, action_col = st.columns([3, 1])
with llm_col:
    st.subheader(f"LLM suggestions — {other_merchants_n} merchants to classify")

with action_col:
    run_llm = st.button(
        "▶ Run Groq suggestions",
        disabled=(not GROQ_AVAILABLE or not api_key),
        type="primary",
        use_container_width=True,
    )

if run_llm:
    if not api_key:
        st.error("No API key set. Enter your Groq key in the sidebar.")
    else:
        progress = st.progress(0, text="Sending merchants to Groq…")

        def _cb(done: int, total: int) -> None:
            pct = done / total if total else 0
            progress.progress(pct, text=f"Classifying {done}/{total} merchants…")

        try:
            with st.spinner("Calling Groq API…"):
                suggestions = suggest_categories(
                    merchants=merchant_list,
                    api_key=api_key,
                    model=model_choice,
                    progress_callback=_cb,
                )
            st.session_state.llm_suggestions = suggestions
            progress.empty()
            st.success(
                f"Got suggestions for {len(suggestions)} merchants.", icon="✓"
            )
        except Exception as exc:
            progress.empty()
            st.error(f"Groq API error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Bulk-accept high-confidence suggestions
# ─────────────────────────────────────────────────────────────────────────────

suggestions = st.session_state.llm_suggestions
high_conf = {
    m: s
    for m, s in suggestions.items()
    if s["confidence"] >= min_confidence and s["category"] != "Other"
}

if high_conf:
    st.info(
        f"**{len(high_conf)}** suggestions meet the {min_confidence:.0%} confidence threshold.",
        icon="ℹ️",
    )
    bcol1, bcol2 = st.columns([1, 5])
    with bcol1:
        if st.button("✅ Accept all high-confidence", use_container_width=True):
            mappings = {m: s["category"] for m, s in high_conf.items()}
            bulk_upsert(mappings, source="llm_accepted")
            st.cache_data.clear()
            st.success(f"Saved {len(mappings)} mappings.", icon="✓")
            st.rerun()

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Per-merchant review table
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Per-merchant review")
st.caption(
    "Each row = one unique merchant still in 'Other'. "
    "Accept the LLM suggestion or pick your own category, then click Save."
)

# Tabs: Unresolved | Already resolved
tab_unresolved, tab_resolved = st.tabs(
    [f"Unresolved ({other_merchants_n})", f"Resolved ({overrides_applied})"]
)

with tab_unresolved:
    save_queue: dict[str, str] = {}  # merchant → chosen category

    for _, row in other_summary.iterrows():
        merchant = row["Merchant"]
        tx_count = int(row["tx_count"])
        total_amt = float(row["total_amount"])
        sample_date = pd.to_datetime(row["sample_date"]).strftime("%d %b %Y")

        # Suggestion data
        sug = suggestions.get(merchant, {})
        sug_cat = sug.get("category", "—")
        sug_conf = sug.get("confidence", 0.0)
        sug_reason = sug.get("reasoning", "")

        is_high = sug_conf >= min_confidence and sug_cat != "Other" and sug_cat != "—"
        conf_color = "🟢" if sug_conf >= 0.9 else "🟡" if sug_conf >= 0.7 else "🔴"

        with st.expander(
            f"{conf_color if sug else '⬜'} **{merchant[:70]}** — "
            f"{tx_count} tx · ฿{total_amt:,.0f} · last {sample_date}",
            expanded=is_high,
        ):
            r1, r2, r3 = st.columns([2, 2, 1])

            with r1:
                if sug_cat != "—":
                    conf_badge = (
                        f"{'✅' if sug_conf >= min_confidence else '⚠️'} "
                        f"**{sug_cat}** ({sug_conf:.0%})"
                    )
                    st.markdown(f"LLM: {conf_badge}")
                    if sug_reason:
                        st.caption(f"_{sug_reason}_")
                else:
                    st.caption("No LLM suggestion yet — run Groq above.")

            with r2:
                default_idx = (
                    CATEGORY_LIST.index(sug_cat)
                    if sug_cat in CATEGORY_LIST
                    else CATEGORY_LIST.index("Other")
                )
                chosen = st.selectbox(
                    "Category",
                    CATEGORY_LIST,
                    index=default_idx,
                    key=f"sel_{merchant}",
                    label_visibility="collapsed",
                )

            with r3:
                save_src = "llm_accepted" if chosen == sug_cat and sug_cat != "Other" else "manual"
                if st.button("💾 Save", key=f"save_{merchant}", use_container_width=True):
                    upsert(merchant, chosen, source=save_src)
                    st.cache_data.clear()
                    st.toast(f"Saved: {merchant[:40]} → {chosen}")
                    st.rerun()

    # Batch save button at the bottom
    st.markdown("---")
    bc1, bc2 = st.columns([1, 3])
    with bc1:
        if st.button(
            "💾 Save all selections",
            type="primary",
            use_container_width=True,
            help="Saves the current selectbox value for every merchant above",
        ):
            to_save: dict[str, str] = {}
            for merchant in merchant_list:
                chosen_val = st.session_state.get(f"sel_{merchant}")
                if chosen_val and chosen_val != "Other":
                    sug = suggestions.get(merchant, {})
                    src = (
                        "llm_accepted"
                        if chosen_val == sug.get("category")
                        else "manual"
                    )
                    to_save[merchant] = chosen_val

            if to_save:
                bulk_upsert(to_save, source="manual")
                st.cache_data.clear()
                st.success(f"Saved {len(to_save)} merchant mappings.", icon="✓")
                st.rerun()
            else:
                st.info("No non-'Other' categories selected yet.")

with tab_resolved:
    overrides_all = load_overrides()
    if not overrides_all:
        st.info("No overrides saved yet.")
    else:
        rows = []
        for key, entry in overrides_all.items():
            rows.append(
                {
                    "Merchant": entry.get("original", key),
                    "Category": entry["category"],
                    "Source": entry.get("source", "manual"),
                    "Approved at": entry.get("approved_at", "")[:10],
                }
            )
        ov_df = pd.DataFrame(rows).sort_values("Approved at", ascending=False)
        st.dataframe(ov_df, use_container_width=True, hide_index=True)

        # Per-row delete
        st.markdown("**Remove an override:**")
        del_merchant = st.selectbox(
            "Select merchant to remove",
            [r["Merchant"] for r in rows],
            label_visibility="collapsed",
        )
        if st.button("🗑 Remove selected override"):
            delete(del_merchant)
            st.cache_data.clear()
            st.rerun()

        st.download_button(
            "⬇ Export overrides (JSON)",
            data=Path(DATA_DIR / "manual_overrides.json").read_text(encoding="utf-8")
            if (DATA_DIR / "manual_overrides.json").exists()
            else "{}",
            file_name="manual_overrides.json",
            mime="application/json",
        )

# ─────────────────────────────────────────────────────────────────────────────
# Footer hint
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Saved overrides are stored in `data/manual_overrides.json` and applied "
    "automatically to all analysis pages on next data load."
)
