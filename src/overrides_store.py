"""
Persistent override store — dual write: local JSON + Supabase.

Local JSON (data/manual_overrides.json) is the primary source of truth
so the app works fully offline.  Supabase is synced on every write so
state survives Streamlit Cloud redeployments and is shared across sessions.

Read priority: Supabase (if available) → local JSON fallback.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import src.supabase_store as _sb

try:
    from src.config import DATA_DIR
    OVERRIDES_PATH = DATA_DIR / "manual_overrides.json"
except ImportError:
    OVERRIDES_PATH = Path("data/manual_overrides.json")


# ── Local JSON helpers ────────────────────────────────────────────────────────

def _load_local(path: Path | None = None) -> dict[str, dict]:
    p = Path(path or OVERRIDES_PATH)
    if not p.exists():
        return {}
    try:
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_local(overrides: dict[str, dict], path: Path | None = None) -> None:
    p = Path(path or OVERRIDES_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(overrides, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


# ── Public API ────────────────────────────────────────────────────────────────

def load_overrides(path: Path | None = None) -> dict[str, dict]:
    """
    Load overrides.  Reads from Supabase if available (most up-to-date),
    falls back to local JSON.
    """
    if _sb.is_available():
        remote = _sb.fetch_overrides()
        if remote:
            # Keep local JSON in sync for offline fallback
            _save_local(remote, path)
            return remote

    return _load_local(path)


def save_overrides(overrides: dict[str, dict], path: Path | None = None) -> None:
    """Write overrides to local JSON only (bulk replace — used by legacy callers)."""
    _save_local(overrides, path)


def upsert(
    merchant_original: str,
    category: str,
    source: str = "manual",
    path: Path | None = None,
) -> dict[str, dict]:
    """Add/update one merchant mapping.  Writes to both local JSON and Supabase."""
    # Local
    overrides = _load_local(path)
    key = merchant_original.strip().lower()
    overrides[key] = {
        "category":    category,
        "source":      source,
        "original":    merchant_original.strip(),
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_local(overrides, path)

    # Supabase (best-effort)
    _sb.upsert_override(merchant_original, category, source)

    return overrides


def bulk_upsert(
    mappings: dict[str, str],
    source: str = "llm_accepted",
    path: Path | None = None,
) -> dict[str, dict]:
    """Bulk add/update merchant mappings."""
    overrides = _load_local(path)
    ts = datetime.now(timezone.utc).isoformat()
    for merchant, category in mappings.items():
        key = merchant.strip().lower()
        overrides[key] = {
            "category":    category,
            "source":      source,
            "original":    merchant.strip(),
            "approved_at": ts,
        }
    _save_local(overrides, path)
    _sb.bulk_upsert_overrides(mappings, source)
    return overrides


def delete(merchant_original: str, path: Path | None = None) -> dict[str, dict]:
    """Remove one override entry."""
    overrides = _load_local(path)
    key = merchant_original.strip().lower()
    overrides.pop(key, None)
    _save_local(overrides, path)
    _sb.delete_override(merchant_original)
    return overrides


def apply_overrides(df, overrides: dict[str, dict] | None = None):
    """Apply overrides to a spending DataFrame.  Only updates 'Other' rows."""
    import pandas as pd
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
        exact  = merchant_lower == key
        substr = (~exact) & merchant_lower.str.contains(key, regex=False, na=False)
        mask   = (exact | substr) & (df["Category"] == "Other")
        if mask.any():
            df.loc[mask, "Category"] = cat

    return df


def override_stats(overrides: dict[str, dict]) -> dict[str, Any]:
    counts: dict[str, int] = {"manual": 0, "llm_accepted": 0, "llm_auto": 0}
    for entry in overrides.values():
        src = entry.get("source", "manual")
        counts[src] = counts.get(src, 0) + 1
    counts["total"] = len(overrides)
    return counts
