"""
RAG cache for LLM-generated behavioral reports.

Two-tier lookup:
  Tier 1 (always available): exact SHA-256 hash match — zero-latency,
          works without any ML dependencies or Supabase.
  Tier 2 (optional):         semantic similarity via pgvector + sentence-transformers
          or fastembed.  Enabled automatically when the embedding library
          is available and Supabase is configured.

Cache key fingerprint:
    SHA-256 of (person, date_start, date_end, top_categories, goals_hash)

TTL: 7 days.  Invalidated explicitly via the Insights page refresh button.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import src.supabase_store as db

EMBEDDINGS_AVAILABLE = False
_embed_fn = None


def _try_load_embedder():
    """Lazily load an embedding model. Returns callable or None."""
    global EMBEDDINGS_AVAILABLE, _embed_fn
    if _embed_fn is not None:
        return _embed_fn

    # Try sentence-transformers (best quality)
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _model = SentenceTransformer("all-MiniLM-L6-v2")

        def _st_embed(text: str) -> list[float]:
            return _model.encode(text, normalize_embeddings=True).tolist()

        _embed_fn = _st_embed
        EMBEDDINGS_AVAILABLE = True
        return _embed_fn
    except Exception:
        pass

    return None


def make_fingerprint(
    person: str,
    date_start: Any,
    date_end: Any,
    top_categories: list[str],
    goals: dict | None,
) -> str:
    """
    Return a 16-char hex fingerprint for exact cache lookup.
    Stable across Python restarts — uses SHA-256 of sorted JSON.
    """
    payload = {
        "p":    person,
        "ds":   str(date_start)[:10],
        "de":   str(date_end)[:10],
        "cats": sorted(str(c) for c in top_categories),
        "g":    json.dumps(goals or {}, sort_keys=True),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()[:32]


def get_cached_report(
    person: str,
    date_start: Any,
    date_end: Any,
    top_categories: list[str],
    goals: dict | None,
    max_age_days: int = 7,
) -> str | None:
    """
    Look up a previously generated report.

    1. Try exact hash match first (fast, no ML).
    2. If embeddings + Supabase available, try semantic similarity match.

    Returns report markdown string or None.
    """
    key = make_fingerprint(person, date_start, date_end, top_categories, goals)

    # Tier 1: exact match
    cached = db.fetch_cached_report_by_hash(key, max_age_days)
    if cached:
        return cached

    # Tier 2: semantic similarity (optional)
    embedder = _try_load_embedder()
    if embedder and db.is_available():
        query_text = (
            f"{person} spending {date_start} to {date_end} "
            f"categories: {', '.join(top_categories[:5])}"
        )
        try:
            emb = embedder(query_text)
            cached = db.fetch_cached_report_semantic(emb, person, threshold=0.92, max_age_days=max_age_days)
            if cached:
                return cached
        except Exception:
            pass

    return None


def store_report(
    report_markdown: str,
    person: str,
    model_used: str,
    date_start: Any,
    date_end: Any,
    top_categories: list[str],
    goals: dict | None,
) -> None:
    """Store a generated report in the cache (both hash key + optional embedding)."""
    key = make_fingerprint(person, date_start, date_end, top_categories, goals)

    embedding = None
    embedder = _try_load_embedder()
    if embedder:
        query_text = (
            f"{person} spending {date_start} to {date_end} "
            f"categories: {', '.join(top_categories[:5])}"
        )
        try:
            embedding = embedder(query_text)
        except Exception:
            pass

    from datetime import date as _date
    ds = _date.fromisoformat(str(date_start)[:10]) if date_start else None
    de = _date.fromisoformat(str(date_end)[:10]) if date_end else None

    db.store_cached_report(
        cache_key_hash=key,
        report_markdown=report_markdown,
        person=person,
        model_used=model_used,
        date_range_start=ds,
        date_range_end=de,
        embedding=embedding,
    )
