# Spending Pattern & Forecast Dashboard

Interactive Streamlit app for analysing **KBank** bank statement and credit card spending data for **Kanokphan** and **Yensa**, with multi-model time-series forecasting and a hybrid LLM + manual merchant categorisation panel.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## Pages

| Page | Description |
|---|---|
| **Home** | Summary KPIs + side-by-side monthly trend for both people |
| **Kanokphan** | Bank + CC analysis, category breakdown, heatmap, transaction table |
| **Yensa** | Bank + CC analysis, category breakdown, heatmap, transaction table |
| **Comparison** | Side-by-side metrics and full category comparison table |
| **Forecasting** | ETS · ARIMA · Ridge · Prophet — model selector, CI bands, CV metrics |
| **Categorise** | Hybrid Groq LLM + manual panel to resolve "Other" merchants |

---

## Data folder structure

Place your exported KBank CSV files using **exactly** this path convention:

```
data/
├── Kanokphan/
│   ├── BankAccount/
│   │   ├── resultFile_20260324_175715.csv
│   │   └── resultFile_20260924_120000.csv   ← multiple files supported
│   └── CreditCard/
│       ├── credit_card_statement_20260324_153128.csv
│       └── credit_card_statement_20260924_153128.csv
└── Yensa/
    ├── BankAccount/
    │   └── resultFile_20260324_180748.csv
    └── CreditCard/
        └── credit_card_statement_20260324_180930.csv
```

Multiple CSV files per folder are **automatically concatenated and deduplicated**.

> ⚠️ Never commit real financial data to a public repository. The `.gitignore` excludes all CSVs inside `data/`.

You can also **upload files directly in the sidebar** of each person's page — no folder required.

---

## Supported CSV formats

### Bank statement (KBank savings account)
- Exported as `resultFile_YYYYMMDD_HHMMSS.csv`
- Thai encoding (UTF-8 with BOM)
- 9-row metadata header, transaction data from row 10
- Columns: Date (DD-MM-YY), Time, Description, Withdrawal, Deposit, Balance, Channel, Details

### Credit card statement (KBank)
- Exported as `credit_card_statement_YYYYMMDD_HHMMSS.csv`
- 5-row metadata header, transaction data from row 6
- Columns: Effective Date (DD/MM/YYYY), Posting Date, Transfer Name (Merchant), Transfer Amount

---

## Categorisation pipeline

Transactions are classified in four sequential layers — first match wins:

| Layer | Source | Covers |
|---|---|---|
| 0 | `src/config.py` — `BANK_CATEGORIES` / `CC_CATEGORIES` | Broad keyword rules (Thai + English) |
| 1 | `src/categoriser.py` — `BANK_KEYWORD_EXTRA` | Thai merchant names, SCB QR wrappers, utilities, EV charging |
| 2 | `src/categoriser.py` — `MERCHANT_OVERRIDES` | CC merchants and international brands |
| 3 | `src/overrides_store.py` — `manual_overrides.json` | Human/LLM-approved mappings (persistent) |

Transfer exclusion is **surgical** — only genuine double-counts are dropped:
- KBank credit card bill payments (`kbank card`)
- Investment transfers to KSecurities / Kasikorn Securities

All `Paid for Ref` QR merchant payments are **kept** as real spending.

---

## Groq LLM categorisation panel

The **Categorise** page (`pages/5_Categorise.py`) resolves merchants that remain in "Other" using a hybrid workflow:

```
Run Groq suggestions
  → batches up to 20 merchants per API call
  → llama-3.3-70b-versatile (falls back to llama3-8b on rate limits)
  → returns {category, confidence 0–1, reasoning} per merchant

Bulk-accept: one click to approve all suggestions ≥ configured threshold
Per-row review: accept LLM suggestion or pick your own from the full list
Save: written to data/manual_overrides.json (exact + substring match)
All pages: pick up corrections on next cache clear / reload
```

### Security — API key handling

The sidebar `st.text_input` for the Groq key always renders **blank** (`value=""`).  
The backend key is loaded from Streamlit secrets or environment variables and used silently — it is never echoed back into the UI, so the Streamlit "eye" icon cannot reveal it.

Resolution priority:

```
1. User types a key into the sidebar field  →  used for this session only
2. Field left blank  →  falls back to GROQ_API_KEY from secrets / env
```

No key is ever stored in the widget value, a cookie, or session state.

### Setting up the Groq API key

**Streamlit Cloud:**

Go to your app → *Settings* → *Secrets* → paste:

```toml
GROQ_API_KEY = "gsk_YOUR_KEY_HERE"
```

**Local development:**

```bash
# Option A — .streamlit/secrets.toml (already in .gitignore)
echo 'GROQ_API_KEY = "gsk_YOUR_KEY_HERE"' >> .streamlit/secrets.toml

# Option B — environment variable
export GROQ_API_KEY="gsk_YOUR_KEY_HERE"
streamlit run app.py
```

Get a free key at [console.groq.com](https://console.groq.com).

---

## Forecasting models

| Model | Min data points | Notes |
|---|---|---|
| Rolling average | 1 | Naive baseline — 3-month window |
| ETS (Holt's) | 2 | Trend-aware exponential smoothing |
| ARIMA(1,1,1) | 24 | Requires ≥ 2 years; skipped automatically on shorter series |
| Ridge regression | 24 | Time + lag features; skipped on shorter series |
| Prophet | 24 | Facebook Prophet — optional install |

Model quality is evaluated with **leave-N-out cross-validation** (hold out last 3 months), reporting MAE, RMSE, and MAPE.

When fewer than 24 months of data are available the app runs baseline models only and shows an info banner explaining why.

---

## Local setup

```bash
# 1. Clone
git clone https://github.com/your-username/spending-forecast.git
cd spending-forecast

# 2. Install
pip install -r requirements.txt

# 3. Add your CSV files
#    data/Kanokphan/BankAccount/*.csv
#    data/Kanokphan/CreditCard/*.csv
#    data/Yensa/BankAccount/*.csv
#    data/Yensa/CreditCard/*.csv

# 4. (Optional) set Groq key
echo 'GROQ_API_KEY = "gsk_..."' >> .streamlit/secrets.toml

# 5. Run
streamlit run app.py
```

### Optional: install Prophet

```bash
pip install prophet
```

Prophet requires a C++ compiler. If it fails, the app still works — it is silently skipped and the remaining four models run normally.

---

## Deploy to Streamlit Cloud

1. Fork this repository (use a **private** fork if your data is included)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your fork · branch `main` · main file `app.py`
4. Add secrets: Settings → Secrets → `GROQ_API_KEY = "gsk_..."`
5. Click **Deploy**

To add data files: push CSVs to `data/` in your private fork, or use the sidebar file uploader on each person's page after deployment.

---

## Project structure

```
spending-forecast/
├── app.py                          # Home page entry point
├── pages/
│   ├── 1_Kanokphan.py              # Kanokphan analysis
│   ├── 2_Yensa.py                  # Yensa analysis
│   ├── 3_Comparison.py             # Side-by-side comparison
│   ├── 4_Forecasting.py            # Multi-model forecast
│   └── 5_Categorise.py             # Groq LLM + manual categorisation panel
├── src/
│   ├── config.py                   # Constants, keyword maps, colours
│   ├── parsers.py                  # KBank CSV parsers + upload handler
│   ├── categoriser.py              # 4-layer transaction categoriser
│   ├── overrides_store.py          # Persistent JSON override store
│   ├── groq_classifier.py          # Groq API batch classifier
│   ├── forecaster.py               # Rolling, ETS, ARIMA, Ridge, Prophet
│   └── charts.py                   # Plotly chart builders
├── data/
│   ├── Kanokphan/
│   │   ├── BankAccount/            # ← drop bank CSVs here
│   │   └── CreditCard/             # ← drop CC CSVs here
│   ├── Yensa/
│   │   ├── BankAccount/
│   │   └── CreditCard/
│   └── manual_overrides.json       # auto-created by Categorise page
├── .streamlit/
│   ├── config.toml                 # Theme + server config
│   └── secrets.toml.example        # Template — copy to secrets.toml
├── requirements.txt
└── .gitignore
```

---

## Category keyword mapping

Categories are assigned by matching transaction text against keyword lists in `src/config.py` and `src/categoriser.py`. To add or adjust a rule, edit `BANK_CATEGORIES`, `CC_CATEGORIES`, `BANK_KEYWORD_EXTRA`, or `MERCHANT_OVERRIDES` — no code changes required elsewhere.

Human-approved and LLM-suggested corrections are saved to `data/manual_overrides.json` and applied automatically as the final classification layer.

---

## License

MIT
