"""
Supabase persistence layer.

All database operations are centralised here.  Every public function
degrades gracefully when Supabase is not configured — it returns
sensible defaults and logs a warning rather than crashing.

Configuration (add to .streamlit/secrets.toml or environment):
    SUPABASE_URL = "https://xxxx.supabase.co"
    SUPABASE_KEY = "eyJ..."   # anon public key
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Any

SUPABASE_AVAILABLE = False
_client = None

try:
    from supabase import create_client, Client  # type: ignore
    SUPABASE_AVAILABLE = True
except ImportError:
    pass


def _get_secrets() -> tuple[str, str]:
    """Return (url, key) from Streamlit secrets or env vars."""
    url, key = "", ""
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
    except Exception:
        pass
    if not url:
        url = os.environ.get("SUPABASE_URL", "")
    if not key:
        key = os.environ.get("SUPABASE_KEY", "")
    return url, key


def get_client():
    """Return a cached Supabase client, or None if not configured."""
    global _client
    if _client is not None:
        return _client
    if not SUPABASE_AVAILABLE:
        return None
    url, key = _get_secrets()
    if not url or not key:
        return None
    try:
        _client = create_client(url, key)
        return _client
    except Exception:
        return None


def is_available() -> bool:
    return get_client() is not None


# ── merchant_overrides ────────────────────────────────────────────────────────

def fetch_overrides() -> dict[str, dict]:
    """
    Return all merchant overrides as {merchant_key: {category, source, original, approved_at}}.
    Returns {} if Supabase not available.
    """
    client = get_client()
    if client is None:
        return {}
    try:
        resp = client.table("merchant_overrides").select("*").execute()
        result = {}
        for row in (resp.data or []):
            result[row["merchant_key"]] = {
                "category":    row["category"],
                "source":      row.get("source", "manual"),
                "original":    row.get("merchant_original", row["merchant_key"]),
                "approved_at": row.get("approved_at", ""),
            }
        return result
    except Exception:
        return {}


def upsert_override(merchant_original: str, category: str, source: str = "manual") -> bool:
    """Upsert a single override. Returns True on success."""
    client = get_client()
    if client is None:
        return False
    try:
        key = merchant_original.strip().lower()
        client.table("merchant_overrides").upsert({
            "merchant_key":      key,
            "merchant_original": merchant_original.strip(),
            "category":          category,
            "source":            source,
            "approved_at":       datetime.now(timezone.utc).isoformat(),
        }, on_conflict="merchant_key").execute()
        return True
    except Exception:
        return False


def bulk_upsert_overrides(mappings: dict[str, str], source: str = "llm_accepted") -> bool:
    """Bulk upsert {merchant_original: category}. Returns True on success."""
    client = get_client()
    if client is None:
        return False
    try:
        ts = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "merchant_key":      m.strip().lower(),
                "merchant_original": m.strip(),
                "category":          cat,
                "source":            source,
                "approved_at":       ts,
            }
            for m, cat in mappings.items()
        ]
        if rows:
            client.table("merchant_overrides").upsert(
                rows, on_conflict="merchant_key"
            ).execute()
        return True
    except Exception:
        return False


def delete_override(merchant_original: str) -> bool:
    client = get_client()
    if client is None:
        return False
    try:
        key = merchant_original.strip().lower()
        client.table("merchant_overrides").delete().eq("merchant_key", key).execute()
        return True
    except Exception:
        return False


# ── saving_goals ──────────────────────────────────────────────────────────────

def fetch_goals(person: str) -> dict[str, Any]:
    """
    Return the most recent saving goals for a person.
    Shape: {monthly_savings_target, category_caps, effective_month}
    """
    client = get_client()
    if client is None:
        return {}
    try:
        resp = (
            client.table("saving_goals")
            .select("*")
            .eq("person", person)
            .order("effective_month", desc=True)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return {}
        row = rows[0]
        return {
            "monthly_savings_target": float(row.get("monthly_savings_target", 0)),
            "category_caps":          row.get("category_caps", {}),
            "effective_month":        row.get("effective_month", ""),
        }
    except Exception:
        return {}


def upsert_goals(
    person: str,
    monthly_target: float,
    category_caps: dict[str, float],
    effective_month: date | None = None,
) -> bool:
    client = get_client()
    if client is None:
        return False
    try:
        if effective_month is None:
            effective_month = date.today().replace(day=1)
        client.table("saving_goals").upsert(
            {
                "person":                  person,
                "effective_month":         effective_month.isoformat(),
                "monthly_savings_target":  monthly_target,
                "category_caps":           category_caps,
                "updated_at":              datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="person,effective_month",
        ).execute()
        return True
    except Exception:
        return False


# ── llm_cache ─────────────────────────────────────────────────────────────────

def fetch_cached_report_by_hash(cache_key_hash: str, max_age_days: int = 7) -> str | None:
    """Exact-match cache lookup. Returns report markdown or None."""
    client = get_client()
    if client is None:
        return None
    try:
        cutoff = datetime.now(timezone.utc)
        from datetime import timedelta
        cutoff = cutoff - timedelta(days=max_age_days)
        resp = (
            client.table("llm_cache")
            .select("report_markdown")
            .eq("cache_key_hash", cache_key_hash)
            .gte("created_at", cutoff.isoformat())
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        return rows[0]["report_markdown"] if rows else None
    except Exception:
        return None


def fetch_cached_report_semantic(
    embedding: list[float],
    person: str,
    threshold: float = 0.92,
    max_age_days: int = 7,
) -> str | None:
    """Semantic similarity cache lookup via pgvector RPC. Returns report or None."""
    client = get_client()
    if client is None:
        return None
    try:
        resp = client.rpc(
            "search_llm_cache",
            {
                "query_embedding":      embedding,
                "query_person":         person,
                "similarity_threshold": threshold,
                "max_age_days":         max_age_days,
            },
        ).execute()
        rows = resp.data or []
        return rows[0]["report_markdown"] if rows else None
    except Exception:
        return None


def store_cached_report(
    cache_key_hash: str,
    report_markdown: str,
    person: str,
    model_used: str,
    date_range_start: date | None = None,
    date_range_end: date | None = None,
    embedding: list[float] | None = None,
) -> bool:
    client = get_client()
    if client is None:
        return False
    try:
        payload: dict[str, Any] = {
            "cache_key_hash":   cache_key_hash,
            "report_markdown":  report_markdown,
            "person":           person,
            "model_used":       model_used,
            "created_at":       datetime.now(timezone.utc).isoformat(),
        }
        if date_range_start:
            payload["date_range_start"] = date_range_start.isoformat()
        if date_range_end:
            payload["date_range_end"] = date_range_end.isoformat()
        if embedding:
            payload["embedding"] = embedding
        client.table("llm_cache").upsert(
            payload, on_conflict="cache_key_hash"
        ).execute()
        return True
    except Exception:
        return False


def invalidate_cache(person: str) -> bool:
    """Delete all cached reports for a person."""
    client = get_client()
    if client is None:
        return False
    try:
        client.table("llm_cache").delete().eq("person", person).execute()
        return True
    except Exception:
        return False


# ── csv_files (Supabase Storage) ──────────────────────────────────────────────

def upload_csv(person: str, source_type: str, filename: str, content: bytes) -> str | None:
    """Upload a CSV to Supabase Storage. Returns storage path or None."""
    client = get_client()
    if client is None:
        return None
    try:
        path = f"{person}/{source_type}/{filename}"
        client.storage.from_("csv-uploads").upload(
            path, content, {"content-type": "text/csv", "upsert": "true"}
        )
        client.table("csv_files").upsert(
            {
                "person":          person,
                "source_type":     source_type,
                "filename":        filename,
                "storage_path":    path,
                "file_size_bytes": len(content),
                "uploaded_at":     datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="person,source_type,filename",
        ).execute()
        return path
    except Exception:
        return None


def list_csv_files(person: str) -> list[dict]:
    """Return list of uploaded CSVs for a person."""
    client = get_client()
    if client is None:
        return []
    try:
        resp = (
            client.table("csv_files")
            .select("*")
            .eq("person", person)
            .order("uploaded_at", desc=True)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


def download_csv(storage_path: str) -> bytes | None:
    """Download a CSV from Supabase Storage."""
    client = get_client()
    if client is None:
        return None
    try:
        return client.storage.from_("csv-uploads").download(storage_path)
    except Exception:
        return None
