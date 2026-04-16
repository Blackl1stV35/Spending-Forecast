# Spending Pattern & Forecast Dashboard

Interactive Streamlit app for analysing **KBank** bank statement and credit card spending for **Kanokphan** and **Yensa**, with LLM behavioral insights, saving goals, multi-model forecasting, and a hybrid merchant categorisation panel.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## Pages

| Page | Description |
|---|---|
| **Home** | Summary KPIs + side-by-side monthly trend |
| **Kanokphan** | Bank + CC analysis, category breakdown, day/month heatmap, transaction table |
| **Yensa** | Same as above with cash-economy note |
| **Comparison** | Side-by-side metrics and full category table |
| **Forecasting** | ETS · ARIMA · Ridge · Prophet — model selector, CI bands, leave-N-out CV |
| **Categorise** | Groq LLM + manual panel to resolve "Other" merchants |
| **Insights** | LLM behavioral report with priority actions (RAG-cached) |
| **Goals** | Per-person monthly target + per-category budget caps + forecast projection |

---

## Infrastructure

```
Streamlit Community Cloud  ←  app deployment (free, no sleep)
         +
Supabase free tier  ←  PostgreSQL + pgvector + 1 GB file storage
```

**What Supabase stores:**

| Table | Contents |
|---|---|
| `merchant_overrides` | Approved LLM / manual category mappings |
| `saving_goals` | Monthly targets + per-category caps per person |
| `llm_cache` | Generated reports with optional vector embeddings |
| `csv_files` | Metadata for CSVs uploaded via browser |

All state survives Streamlit Cloud redeployment. The app degrades gracefully to local-file-only mode when Supabase is not configured.

---

## Data folder structure

```
data/
├── Kanokphan/
│   ├── BankAccount/   ← resultFile_YYYYMMDD_HHMMSS.csv
│   └── CreditCard/    ← credit_card_statement_YYYYMMDD_HHMMSS.csv
└── Yensa/
    ├── BankAccount/
    └── CreditCard/
```

Multiple CSV files per folder are automatically concatenated and deduplicated.
You can also upload files via the sidebar on each person's page.

> Never commit real financial data to a public repository. The `.gitignore` excludes all CSVs inside `data/`.

---

## Setup

### 1. Supabase (one-time)

1. Create a free project at [supabase.com](https://supabase.com)
2. Open the **SQL editor** and run `migrations/001_initial_schema.sql`
3. In **Storage**, create a bucket named `csv-uploads` (private, 10 MB limit)
4. Copy your **Project URL** and **anon public key** from Settings → API

### 2. Secrets

**Streamlit Cloud:** Settings → Secrets → paste:

```toml
GROQ_API_KEY   = "gsk_YOUR_KEY_HERE"
SUPABASE_URL   = "https://YOUR_PROJECT_ID.supabase.co"
SUPABASE_KEY   = "eyJ_YOUR_ANON_KEY_HERE"
```

**Local:** copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` and fill in keys.

Get a free Groq key at [console.groq.com](https://console.groq.com).

### 3. Install & run locally

```bash
git clone https://github.com/your-username/spending-forecast.git
cd spending-forecast
pip install -r requirements.txt

# Add CSV files
cp your_bank.csv   data/Kanokphan/BankAccount/
cp your_cc.csv     data/Kanokphan/CreditCard/

streamlit run app.py
```

### 4. Deploy to Streamlit Cloud

1. Push to a **private** GitHub fork
2. [share.streamlit.io](https://share.streamlit.io) → New app → select repo, branch `main`, file `app.py`
3. Add secrets (step 2 above)
4. Deploy

---

## Categorisation pipeline (4 layers)

| Layer | Source | Covers |
|---|---|---|
| 0 | `src/config.py` BANK/CC dicts | Broad keyword rules |
| 1 | `src/categoriser.py` BANK_KEYWORD_EXTRA | Thai merchant names, SCB QR wrappers, utilities |
| 2 | `src/categoriser.py` MERCHANT_OVERRIDES | International brands, CC merchants |
| 3 | `data/manual_overrides.json` + Supabase | Human/LLM-approved mappings (persistent) |

**Transfer exclusion is surgical** — only KBank card bill payments and investment transfers to KSecurities are excluded. All `Paid for Ref` QR merchant payments are kept as real spending.

---

## LLM features

### Insights page (`6_Insights.py`)
- Generates a behavioral spending report per person using `llama-3.3-70b-versatile`
- Report sections: behavioral summary, top patterns, priority actions, goal gap analysis
- **RAG cache:** SHA-256 fingerprint cache (+ optional pgvector semantic similarity) avoids redundant Groq calls
- Cache TTL: 7 days (configurable), manual invalidation button available

### Categorise page (`5_Categorise.py`)
- Sends unclassified "Other" merchants to Groq in batches of 20
- Returns `{category, confidence, reasoning}` per merchant
- One-click bulk-accept for suggestions ≥ confidence threshold
- Per-row review with selectbox override
- All approvals written to Supabase + local JSON (dual write)

### API key security
The sidebar `st.text_input` always renders **blank** (`value=""`). The backend key (from Streamlit secrets or env var) is used silently — it is never echoed into the DOM, so the "show" eye icon cannot reveal it.

---

## Forecasting models

| Model | Min months | Notes |
|---|---|---|
| Rolling average | 1 | Naive 3-month window baseline |
| ETS (Holt's) | 2 | Trend-aware exponential smoothing |
| ARIMA(1,1,1) | 24 | Auto-gated — skipped with info banner if < 24 months |
| Ridge regression | 24 | Time + lag features; same gate as ARIMA |
| Prophet | 24 | Optional install; same gate |

Outlier clipping (IQR cap, optional) is applied before fitting to prevent investment spikes from distorting model parameters.

---

## Known bug fixes applied (permanent)

| Bug | Fix |
|---|---|
| `st.multiselect` crash on missing defaults | Every `default=` filtered through `[x for x in defaults if x in options]` |
| Invalid `icon=` strings in `st.success` etc. | All icons use true Unicode emoji (`"✅"`, `"❌"`, `"⚠️"`, `"ℹ️"`) |
| Plotly rejects 8-digit hex (`#RRGGBBAA`) | All transparency via `_rgba(hex, alpha)` helper → `rgba(r,g,b,a)` |
| Plotly `add_vline(annotation_text=)` crash on Timestamp x-axis | Split into `add_vline()` + separate `add_annotation()` |

---

## Project structure

```
spending-forecast/
├── app.py
├── pages/
│   ├── 1_Kanokphan.py
│   ├── 2_Yensa.py
│   ├── 3_Comparison.py
│   ├── 4_Forecasting.py
│   ├── 5_Categorise.py          # Groq LLM + manual panel
│   ├── 6_Insights.py            # Behavioral report + priority actions
│   └── 7_Goals.py               # Saving goals + projection
├── src/
│   ├── config.py                 # Constants, keyword maps, colours
│   ├── parsers.py                # KBank CSV parsers
│   ├── categoriser.py            # 4-layer transaction categoriser
│   ├── overrides_store.py        # Dual-write: local JSON + Supabase
│   ├── supabase_store.py         # All Supabase I/O
│   ├── rag_cache.py              # Fingerprint + optional pgvector cache
│   ├── llm_insights.py           # Groq report generation pipeline
│   ├── groq_classifier.py        # Merchant classification via Groq
│   ├── forecaster.py             # Rolling, ETS, ARIMA, Ridge, Prophet
│   └── charts.py                 # Plotly chart builders (bug-fixed)
├── migrations/
│   └── 001_initial_schema.sql    # Run once in Supabase SQL editor
├── data/
│   ├── Kanokphan/BankAccount/
│   ├── Kanokphan/CreditCard/
│   ├── Yensa/BankAccount/
│   ├── Yensa/CreditCard/
│   └── manual_overrides.json     # auto-created; synced to Supabase
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
└── requirements.txt
```

---

## License

MIT
