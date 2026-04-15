"""
Keyword-based transaction categoriser.

Each transaction's text (merchant name or description + details) is matched
against the keyword dictionaries in config.py.  First match wins.
"""

from __future__ import annotations

import pandas as pd

from src.config import BANK_CATEGORIES, CC_CATEGORIES


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

    Output columns:
        Date, Amount, Merchant, Category, source, YearMonth, DayOfWeek
    """
    frames = []

    if not bank_df.empty:
        b = categorise_bank(bank_df)
        # Only outflows
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
        return pd.DataFrame(columns=["Date", "Amount", "Merchant", "Category", "source", "YearMonth", "DayOfWeek"])

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined["YearMonth"] = combined["Date"].dt.to_period("M").dt.to_timestamp()
    combined["DayOfWeek"] = combined["Date"].dt.day_name()
    return combined.sort_values("Date").reset_index(drop=True)
