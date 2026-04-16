"""
Persistent override store for manual and LLM-assisted merchant categorisation.

Storage format: data/manual_overrides.json
{
  "merchant_lower_key": {
    "category":    "Food & Dining",
    "source":      "manual" | "llm_accepted" | "llm_auto",
    "original":    "SCB มณี SHOP (โกเฮง ข้าวมันไก่)",
    "approved_at": "2026-04-15T12:34:56"
  },
  ...
}

The store key is the lowercase-stripped merchant string so lookups are
case-insensitive.  Partial / substring matching is done at apply-time so
one override entry can resolve many variant spellings of the same merchant.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import DATA_DIR

OVERRIDES_PATH = DATA_DIR / "manual_overrides.json"


# ─────────────────────────────────────────────────────────────────────────────
# Load / save
# ─────────────────────────────────────────────────────────────────────────────

def load_overrides(path: Path | None = None) -> dict[str, dict]:
    """Return the full overrides dict.  Returns {} if file absent."""
    p = Path(path or OVERRIDES_PATH)
    if not p.exists():
        return {}
    try:
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_overrides(overrides: dict[str, dict], path: Path | None = None) -> None:
    """Atomically write the overrides dict to disk."""
    p = Path(path or OVERRIDES_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(overrides, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def upsert(
    merchant_original: str,
    category: str,
    source: str = "manual",
    path: Path | None = None,
) -> dict[str, dict]:
    """
    Add or update a single merchant→category mapping and save immediately.

    Parameters
    ----------
    merchant_original : str
        The raw merchant string from the transaction (any case).
    category : str
        Target category label.
    source : "manual" | "llm_accepted" | "llm_auto"
    path : optional override file path

    Returns
    -------
    The updated overrides dict.
    """
    overrides = load_overrides(path)
    key = merchant_original.strip().lower()
    overrides[key] = {
        "category": category,
        "source": source,
        "original": merchant_original.strip(),
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }
    save_overrides(overrides, path)
    return overrides


def bulk_upsert(
    mappings: dict[str, str],
    source: str = "llm_accepted",
    path: Path | None = None,
) -> dict[str, dict]:
    """
    Save multiple merchant→category mappings at once.

    Parameters
    ----------
    mappings : {merchant_original: category}
    source   : source label for all entries
    """
    overrides = load_overrides(path)
    ts = datetime.now(timezone.utc).isoformat()
    for merchant, category in mappings.items():
        key = merchant.strip().lower()
        overrides[key] = {
            "category": category,
            "source": source,
            "original": merchant.strip(),
            "approved_at": ts,
        }
    save_overrides(overrides, path)
    return overrides


def delete(merchant_original: str, path: Path | None = None) -> dict[str, dict]:
    """Remove a single merchant override entry."""
    overrides = load_overrides(path)
    key = merchant_original.strip().lower()
    overrides.pop(key, None)
    save_overrides(overrides, path)
    return overrides


# ─────────────────────────────────────────────────────────────────────────────
# Apply overrides to a DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def apply_overrides(
    df,          # pd.DataFrame with 'Merchant' and 'Category' columns
    overrides: dict[str, dict] | None = None,
) -> "pd.DataFrame":  # noqa: F821 — avoid circular import
    """
    Apply the stored overrides to the spending DataFrame.

    Matching uses two passes:
      1. Exact key match (merchant.lower() == key)
      2. Substring match (key in merchant.lower()) for partial names

    Only rows currently in 'Other' are updated, preserving all keyword-rule
    assignments made earlier in the pipeline.
    """
    import pandas as pd  # local import avoids circular dependency

    if overrides is None:
        overrides = load_overrides()
    if not overrides or df.empty:
        return df

    df = df.copy()
    merchant_lower = df["Merchant"].fillna("").str.lower()

    for key, entry in overrides.items():
        cat = entry.get("category", "Other")
        if cat == "Other":
            continue

        # Pass 1: exact match
        exact = merchant_lower == key
        # Pass 2: substring match (only for rows still in 'Other')
        substr = (~exact) & merchant_lower.str.contains(
            key, regex=False, na=False
        )

        mask = (exact | substr) & (df["Category"] == "Other")
        if mask.any():
            df.loc[mask, "Category"] = cat

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def override_stats(overrides: dict[str, dict]) -> dict[str, Any]:
    """Return summary counts broken down by source."""
    counts: dict[str, int] = {"manual": 0, "llm_accepted": 0, "llm_auto": 0}
    for entry in overrides.values():
        src = entry.get("source", "manual")
        counts[src] = counts.get(src, 0) + 1
    counts["total"] = len(overrides)
    return counts
