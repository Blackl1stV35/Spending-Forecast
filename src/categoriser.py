"""
Keyword-based transaction categoriser.

Each transaction's text (merchant name or description + details) is matched
against the keyword dictionaries in config.py.  First match wins.
"""

from __future__ import annotations

import pandas as pd

from src.config import BANK_CATEGORIES, CC_CATEGORIES

TRANSFER_PATTERNS = [
    "paid for ref",
    "credit card payment",
    "transfer",
    "promptpay",
    "bill payment",
    "direct debit",
]

MERCHANT_OVERRIDES: dict[str, str] = {
    # Fuel
    "shell": "Fuel",
    "caltex": "Fuel",
    "esso": "Fuel",
    "pt gas": "Fuel",
    # Coffee / cafe
    "starbucks": "Beverages & Cafe",
    "cafe amazon": "Beverages & Cafe",
    "black canyon": "Beverages & Cafe",
    "inthanin": "Beverages & Cafe",
    "doi chaang": "Beverages & Cafe",
    # Fast food / dining
    "mcdonald": "Food & Dining",
    "burger king": "Food & Dining",
    "kfc": "Food & Dining",
    "pizza hut": "Food & Dining",
    "pizza company": "Food & Dining",
    "subway": "Food & Dining",
    "sizzler": "Food & Dining",
    "sukishi": "Food & Dining",
    # Grocery / supermarket
    "villa market": "Groceries",
    "gourmet market": "Groceries",
    "rimping": "Groceries",
    "foodland": "Groceries",
    "makro": "Groceries",
    "big c": "Groceries",
    "bigc": "Groceries",
    # Pharmacy / health
    "watsons": "Healthcare",
    "boots": "Healthcare",
    "fascino": "Healthcare",
    # Transport
    "grab": "Transport",
    "bolt": "Transport",
    "taxi": "Transport",
    # Streaming / digital
    "netflix": "Tech & Digital",
    "spotify": "Tech & Digital",
    "youtube": "Tech & Digital",
    "apple.com": "Tech & Digital",
    "google play": "Tech & Digital",
    "line pay": "Tech & Digital",
}

def _is_internal_transfer(merchant: str) -> bool:
    """Return True if the row is an internal transfer that would double-count."""
    m = str(merchant).lower()
    return any(p in m for p in TRANSFER_PATTERNS)


def _apply_merchant_overrides(category: str, merchant: str) -> str:
    """
    If the existing category is 'Other', try the override dict before giving up.
    This resolves common merchants that the keyword rules miss.
    """
    if category != "Other":
        return category
    m = str(merchant).lower()
    for keyword, override_cat in MERCHANT_OVERRIDES.items():
        if keyword in m:
            return override_cat
    return "Other"

def _match(text: str, rules: dict) -> str:
    """Return the first matching category key, or 'Other'."""
    t = str(text).lower()
    for category, keywords in rules.items():
        if any(kw.lower() in t for kw in keywords):
            return category
    return "Other"


def categorise_bank(df: pd.DataFrame) -> pd.DataFrame:
    """Add Category column to a bank DataFrame."""
    if df.empty:
        return df
    df = df.copy()
    text = (
        df.get("Details", pd.Series([""] * len(df), dtype=str)).fillna("").astype(str)
        + " "
        + df.get("Description", pd.Series([""] * len(df), dtype=str)).fillna("").astype(str)
    )
    df["Category"] = text.apply(lambda x: _match(x, BANK_CATEGORIES))
    return df


def categorise_cc(df: pd.DataFrame) -> pd.DataFrame:
    """Add Category column to a CC DataFrame."""
    if df.empty:
        return df
    df = df.copy()
    df["Category"] = df["Merchant"].apply(lambda x: _match(x, CC_CATEGORIES))
    return df


def get_spending_df(bank_df: pd.DataFrame, cc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a unified spending DataFrame from bank withdrawals + CC purchases.

    Applies two cleaning steps:
      1. Internal transfer filter  — drops rows that would double-count
      2. Merchant override dict    — shrinks the 'Other' bucket

    Output columns:
        Date, Amount, Merchant, Category, source, YearMonth, DayOfWeek
    """
    frames = []

    if not bank_df.empty:
        b = categorise_bank(bank_df)
        mask = b["Withdrawal"].notna() & (b["Withdrawal"] > 0)
        out = b[mask].copy()
        out["Amount"] = out["Withdrawal"]
        out["Merchant"] = out.apply(
            lambda r: (str(r.get("Details", "")) or str(r.get("Description", ""))).strip(),
            axis=1,
        )
        frames.append(out[["Date", "Amount", "Merchant", "Category", "source"]].copy())

    if not cc_df.empty:
        c = categorise_cc(cc_df)
        spend = c[c["Amount"] > 0].copy()
        frames.append(spend[["Date", "Amount", "Merchant", "Category", "source"]].copy())

    if not frames:
        return pd.DataFrame(
            columns=["Date", "Amount", "Merchant", "Category", "source", "YearMonth", "DayOfWeek"]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])

    # ── Patch 1a: drop internal transfers ────────────────────────────────────
    transfer_mask = combined["Merchant"].apply(_is_internal_transfer)
    combined = combined[~transfer_mask].copy()

    # ── Patch 1b: apply merchant override dict to shrink 'Other' bucket ──────
    combined["Category"] = combined.apply(
        lambda r: _apply_merchant_overrides(r["Category"], r["Merchant"]),
        axis=1,
    )

    combined["YearMonth"] = combined["Date"].dt.to_period("M").dt.to_timestamp()
    combined["DayOfWeek"] = combined["Date"].dt.day_name()
    return combined.sort_values("Date").reset_index(drop=True)
