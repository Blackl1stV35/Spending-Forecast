"""
Groq-backed LLM merchant classifier.

Sends batches of unknown merchant names to the Groq API and returns
structured category suggestions with confidence scores and brief reasoning.

Usage
-----
    from src.groq_classifier import suggest_categories, GROQ_AVAILABLE

    if GROQ_AVAILABLE:
        results = suggest_categories(
            merchants=["SCB มณี SHOP (โกเฮง ข้าวมันไก่)", "NORTH PARK PROPERT++"],
            api_key="gsk_...",
        )
        # results = {
        #   "SCB มณี SHOP (โกเฮง ข้าวมันไก่)": {
        #       "category": "Food & Dining",
        #       "confidence": 0.95,
        #       "reasoning": "SCB QR wrapper for a chicken-rice stall (ข้าวมันไก่)"
        #   },
        #   ...
        # }

Model selection
---------------
Uses llama-3.3-70b-versatile by default — best accuracy on Thai merchant names.
Falls back to llama3-8b-8192 on rate-limit errors.

Error handling
--------------
Any merchant that fails to parse returns category="Other" with confidence=0.0.
The caller (Streamlit page) always gets a complete dict keyed by every input
merchant so the UI never has missing rows.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

GROQ_AVAILABLE = False
try:
    from groq import Groq  # type: ignore
    GROQ_AVAILABLE = True
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Category list (must stay in sync with src/config.py)
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_LIST = [
    "Beverages & Cafe",
    "Cash Withdrawal",
    "E-commerce",
    "Education",
    "Entertainment",
    "Family / Personal",
    "Food & Dining",
    "Fuel",
    "Government & Fees",
    "Groceries",
    "Health & Fitness",
    "Healthcare",
    "Home & Hardware",
    "Home & Services",
    "Incoming Transfer",
    "Interest",
    "Investment",
    "Regular Fixed Transfer",
    "Shopping & Fashion",
    "Tech & Digital",
    "Transport",
    "Travel",
    "Utilities",
    "Other",
]

# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a financial transaction categoriser for a Thai personal banking app.
Your job is to classify merchant names (which may be in Thai, English, or both)
into spending categories.

Context:
- Transactions come from KBank (Kasikorn Bank) in Thailand.
- "SCB มณี SHOP (X)" = X is the actual merchant inside an SCB QR wrapper.
- "Paid for Ref XXXX <name>" = QR payment to <name>.
- "To PromptPay XXXX <name>" = PromptPay transfer to <name>.
- Thai text clues: ข้าวมันไก่=chicken rice, ร้าน=shop, คลินิก=clinic, เภสัช=pharmacy,
  สยาม ซีนีเพล็กซ์=cinema, ชา=tea, กาแฟ=coffee, ทำความสะอาด=cleaning.

Respond with ONLY a valid JSON object.  No markdown, no explanation outside JSON.
Format:
{
  "merchant name exactly as given": {
    "category": "<one category from the list>",
    "confidence": <float 0.0–1.0>,
    "reasoning": "<one short sentence>"
  },
  ...
}
"""

_USER_TEMPLATE = """\
Classify each merchant into one category from this list:
{categories}

Merchants to classify:
{merchants_json}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Core classification function
# ─────────────────────────────────────────────────────────────────────────────

_BATCH_SIZE = 20          # merchants per API call
_RETRY_LIMIT = 2
_FALLBACK_MODEL = "llama3-8b-8192"
_PRIMARY_MODEL = "llama-3.3-70b-versatile"


def _build_prompt(merchants: list[str]) -> str:
    return _USER_TEMPLATE.format(
        categories="\n".join(f"  - {c}" for c in CATEGORY_LIST),
        merchants_json=json.dumps(merchants, ensure_ascii=False, indent=2),
    )


def _parse_response(raw: str, merchants: list[str]) -> dict[str, dict]:
    """
    Parse the raw LLM text into a structured dict.
    Handles JSON embedded in markdown fences and partial responses.
    """
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting the first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}

    result: dict[str, dict] = {}
    for merchant in merchants:
        entry = parsed.get(merchant, {})
        if not isinstance(entry, dict):
            entry = {}
        cat = entry.get("category", "Other")
        if cat not in CATEGORY_LIST:
            cat = "Other"
        result[merchant] = {
            "category": cat,
            "confidence": float(entry.get("confidence", 0.0)),
            "reasoning": str(entry.get("reasoning", "")).strip(),
        }
    return result


def _call_groq(
    client: "Groq",  # type: ignore[name-defined]
    merchants: list[str],
    model: str,
) -> dict[str, dict]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(merchants)},
        ],
        temperature=0.1,        # low temperature → consistent classifications
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    return _parse_response(raw, merchants)


def suggest_categories(
    merchants: list[str],
    api_key: str | None = None,
    model: str = _PRIMARY_MODEL,
    progress_callback: Any = None,
) -> dict[str, dict]:
    """
    Classify a list of merchant strings using Groq LLM.

    Parameters
    ----------
    merchants : list[str]
        Unique merchant name strings to classify.
    api_key : str | None
        Groq API key.  Falls back to GROQ_API_KEY env variable.
    model : str
        Groq model name.  Defaults to llama-3.3-70b-versatile.
    progress_callback : callable(completed: int, total: int) | None
        Optional callback for Streamlit progress bar updates.

    Returns
    -------
    dict[merchant → {"category", "confidence", "reasoning"}]
    All input merchants are guaranteed to appear in the output.
    """
    if not GROQ_AVAILABLE:
        raise ImportError("groq package not installed. Run: pip install groq")

    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError(
            "No Groq API key found.  Set GROQ_API_KEY in Streamlit secrets "
            "or as an environment variable."
        )

    client = Groq(api_key=key)
    results: dict[str, dict] = {}
    total = len(merchants)

    # Process in batches
    for i in range(0, total, _BATCH_SIZE):
        batch = merchants[i : i + _BATCH_SIZE]
        attempt = 0
        current_model = model

        while attempt < _RETRY_LIMIT:
            try:
                batch_results = _call_groq(client, batch, current_model)
                results.update(batch_results)
                break
            except Exception as exc:
                attempt += 1
                err = str(exc).lower()
                if "rate" in err and attempt < _RETRY_LIMIT:
                    time.sleep(2 ** attempt)        # exponential back-off
                    current_model = _FALLBACK_MODEL  # drop to smaller model
                elif attempt >= _RETRY_LIMIT:
                    # Fill with safe defaults so UI never breaks
                    for m in batch:
                        results[m] = {
                            "category": "Other",
                            "confidence": 0.0,
                            "reasoning": f"Classification failed: {exc}",
                        }

        if progress_callback:
            progress_callback(min(i + _BATCH_SIZE, total), total)

    # Guarantee every input merchant has an entry
    for m in merchants:
        if m not in results:
            results[m] = {
                "category": "Other",
                "confidence": 0.0,
                "reasoning": "Not returned by LLM.",
            }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Utility: group "Other" transactions from a spending DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def extract_other_merchants(df: "pd.DataFrame") -> "pd.DataFrame":  # noqa: F821
    """
    From a spending DataFrame, return a summary of merchants still in 'Other'.

    Output columns:
        Merchant, tx_count, total_amount, sample_date, sample_detail
    """
    import pandas as pd  # local import

    if df.empty:
        return pd.DataFrame(
            columns=["Merchant", "tx_count", "total_amount", "sample_date", "sample_detail"]
        )
    others = df[df["Category"] == "Other"].copy()
    if others.empty:
        return pd.DataFrame(
            columns=["Merchant", "tx_count", "total_amount", "sample_date", "sample_detail"]
        )

    agg = (
        others.groupby("Merchant")
        .agg(
            tx_count=("Amount", "count"),
            total_amount=("Amount", "sum"),
            sample_date=("Date", "max"),
        )
        .reset_index()
        .sort_values("total_amount", ascending=False)
    )
    return agg
