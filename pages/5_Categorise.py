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
from src.groq_classifier import (GROQ_AVAILABLE, CATEGORY_LIST,
                                   extract_other_merchants, suggest_categories)
from src.overrides_store import (bulk_upsert, delete, load_overrides,
                                  override_stats, upsert)

st.set_page_config(page_title="Categorise — Other", page_icon="🏷️", layout="wide")
st.title("🏷️ Merchant categorisation panel")
st.caption("Resolve 'Other' merchants using Groq LLM suggestions + manual review")

with st.sidebar:
    st.header("Groq API")
    secret_key = ""
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret_key = os.environ.get("GROQ_API_KEY", "")

    # BUG security: value="" — key never echoed to UI
    api_key_input = st.text_input(
        "API key", value="", type="password",
        placeholder="gsk_... (leave blank to use server key)",
        help="Get a free key at console.groq.com. Leave blank if server key is configured.",
    )
    api_key = api_key_input.strip() or secret_key

    if not GROQ_AVAILABLE:
        st.error("groq package not installed.\n\nRun: `pip install groq`", icon="❌")
    elif api_key:
        st.success("API key set", icon="✅")
    else:
        st.warning("Enter a Groq API key to enable LLM suggestions.", icon="⚠️")

    st.divider()
    model_choice    = st.selectbox("Model",
        ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"])
    min_confidence  = st.slider("Auto-accept threshold", 0.70, 1.00, 0.90, 0.01)
    person_filter   = st.multiselect("Show people", PEOPLE, default=PEOPLE)

    st.divider()
    st.header("Override store")
    overrides_now = load_overrides()
    stats         = override_stats(overrides_now)
    st.metric("Total overrides", stats["total"])
    st.caption(f"Manual: {stats['manual']}  LLM: {stats['llm_accepted']}")
    if st.button("🗑 Clear ALL overrides", type="secondary"):
        p = DATA_DIR / "manual_overrides.json"
        if p.exists():
            p.unlink()
        st.cache_data.clear()
        st.rerun()


@st.cache_data(show_spinner="Loading transactions…")
def _load(d: str) -> dict:
    result = {}
    for p in PEOPLE:
        bank, cc = load_person_data(p, Path(d))
        result[p] = get_spending_df(bank, cc)
    return result


all_data = load_all_data = _load(str(DATA_DIR))

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

combined_df  = pd.concat(frames, ignore_index=True)
other_summary = extract_other_merchants(combined_df)

total_tx = len(combined_df)
other_tx = len(combined_df[combined_df["Category"] == "Other"])
other_pct = other_tx / total_tx * 100 if total_tx else 0

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Total transactions",     f"{total_tx:,}")
col_b.metric("Still in 'Other'",       f"{other_tx:,}",
             delta=f"{other_pct:.1f}%", delta_color="inverse")
col_c.metric("Unique 'Other' merchants", len(other_summary))
col_d.metric("Overrides saved",          stats["total"])

if other_tx == 0:
    st.success("All transactions are categorised!", icon="✅")
    st.stop()

st.markdown("---")

if "llm_suggestions" not in st.session_state:
    st.session_state.llm_suggestions = {}

merchant_list = other_summary["Merchant"].tolist()

llm_col, action_col = st.columns([3, 1])
with llm_col:
    st.subheader(f"LLM suggestions — {len(merchant_list)} merchants")
with action_col:
    run_llm = st.button("▶ Run Groq suggestions",
                         disabled=(not GROQ_AVAILABLE or not api_key),
                         type="primary", use_container_width=True)

if run_llm:
    progress = st.progress(0, text="Sending merchants to Groq…")
    def _cb(done, total):
        progress.progress(done / total if total else 0,
                          text=f"Classifying {done}/{total}…")
    try:
        with st.spinner("Calling Groq API…"):
            suggestions = suggest_categories(
                merchants=merchant_list, api_key=api_key,
                model=model_choice, progress_callback=_cb,
            )
        st.session_state.llm_suggestions = suggestions
        progress.empty()
        st.success(f"Got suggestions for {len(suggestions)} merchants.", icon="✅")
    except Exception as exc:
        progress.empty()
        st.error(f"Groq API error: {exc}", icon="❌")

suggestions  = st.session_state.llm_suggestions
high_conf = {
    m: s for m, s in suggestions.items()
    if s["confidence"] >= min_confidence and s["category"] not in ("Other", "—")
}

if high_conf:
    st.info(f"**{len(high_conf)}** suggestions meet the {min_confidence:.0%} threshold.", icon="ℹ️")
    if st.button("✅ Accept all high-confidence"):
        bulk_upsert({m: s["category"] for m, s in high_conf.items()}, source="llm_accepted")
        st.cache_data.clear()
        st.success(f"Saved {len(high_conf)} mappings.", icon="✅")
        st.rerun()

st.markdown("---")
st.subheader("Per-merchant review")

tab_unresolved, tab_resolved = st.tabs(
    [f"Unresolved ({len(other_summary)})", f"Resolved ({stats['total']})"]
)

with tab_unresolved:
    for _, row in other_summary.iterrows():
        merchant   = row["Merchant"]
        tx_count   = int(row["tx_count"])
        total_amt  = float(row["total_amount"])
        sample_date = pd.to_datetime(row["sample_date"]).strftime("%d %b %Y")
        sug        = suggestions.get(merchant, {})
        sug_cat    = sug.get("category", "—")
        sug_conf   = sug.get("confidence", 0.0)
        sug_reason = sug.get("reasoning", "")
        is_high    = sug_conf >= min_confidence and sug_cat not in ("Other", "—")
        flag       = "🟢" if sug_conf >= 0.9 else ("🟡" if sug_conf >= 0.7 else "🔴") if sug else "⬜"

        with st.expander(
            f"{flag} **{merchant[:70]}** — {tx_count} tx · ฿{total_amt:,.0f} · last {sample_date}",
            expanded=is_high,
        ):
            r1, r2, r3 = st.columns([2, 2, 1])
            with r1:
                if sug_cat != "—":
                    badge = "✅" if sug_conf >= min_confidence else "⚠️"
                    st.markdown(f"LLM: {badge} **{sug_cat}** ({sug_conf:.0%})")
                    if sug_reason:
                        st.caption(f"_{sug_reason}_")
                else:
                    st.caption("No LLM suggestion yet.")
            with r2:
                default_idx = (
                    CATEGORY_LIST.index(sug_cat)
                    if sug_cat in CATEGORY_LIST
                    else CATEGORY_LIST.index("Other")
                )
                chosen = st.selectbox("Category", CATEGORY_LIST,
                                       index=default_idx, key=f"sel_{merchant}",
                                       label_visibility="collapsed")
            with r3:
                src = "llm_accepted" if chosen == sug_cat else "manual"
                if st.button("💾 Save", key=f"save_{merchant}", use_container_width=True):
                    upsert(merchant, chosen, source=src)
                    st.cache_data.clear()
                    st.toast(f"Saved: {merchant[:40]} → {chosen}")
                    st.rerun()

    st.markdown("---")
    if st.button("💾 Save all selections", type="primary"):
        to_save = {}
        for merchant in merchant_list:
            val = st.session_state.get(f"sel_{merchant}")
            if val and val != "Other":
                to_save[merchant] = val
        if to_save:
            bulk_upsert(to_save, source="manual")
            st.cache_data.clear()
            st.success(f"Saved {len(to_save)} mappings.", icon="✅")
            st.rerun()
        else:
            st.info("No non-'Other' categories selected yet.", icon="ℹ️")

with tab_resolved:
    overrides_all = load_overrides()
    if not overrides_all:
        st.info("No overrides saved yet.", icon="ℹ️")
    else:
        rows = [{"Merchant": e.get("original", k), "Category": e["category"],
                 "Source": e.get("source", "manual"), "Approved at": e.get("approved_at","")[:10]}
                for k, e in overrides_all.items()]
        ov_df = pd.DataFrame(rows).sort_values("Approved at", ascending=False)
        st.dataframe(ov_df, use_container_width=True, hide_index=True)

        del_merchant = st.selectbox("Remove override", [r["Merchant"] for r in rows],
                                     label_visibility="collapsed")
        if st.button("🗑 Remove selected"):
            delete(del_merchant)
            st.cache_data.clear()
            st.rerun()

        ovr_path = DATA_DIR / "manual_overrides.json"
        if ovr_path.exists():
            st.download_button("⬇ Export overrides (JSON)",
                               ovr_path.read_text(encoding="utf-8"),
                               "manual_overrides.json", "application/json")

st.markdown("---")
st.caption("Saved overrides are stored in `data/manual_overrides.json` "
           "and applied automatically to all analysis pages on next data load.")
