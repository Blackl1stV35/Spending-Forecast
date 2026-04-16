"""
LLM behavioral insights generator.

Builds a structured prompt from spending data, forecasts, and saving goals,
then calls Groq to produce a markdown report with:
  1. Behavioral summary
  2. Top spending patterns (with evidence)
  3. Prioritised action items per category
  4. Goal gap analysis

Results are cached via src/rag_cache.py to avoid redundant API calls.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

import src.rag_cache as cache
from src.config import EXCLUDE_FROM_LIFESTYLE

GROQ_AVAILABLE = False
try:
    from groq import Groq  # type: ignore
    GROQ_AVAILABLE = True
except ImportError:
    pass

_MODEL = "llama-3.3-70b-versatile"

_SYSTEM_PROMPT = """\
You are a personal finance advisor specializing in Thai household spending.
Analyze the provided data and write a practical, specific report in English.
Use exact ฿ amounts from the data. Be direct and actionable — not generic.
Format your response as clean markdown with these exact section headers:

## Behavioral summary
## Top spending patterns
## Priority actions
## Goal gap analysis

Keep the total report under 600 words. Prioritize accuracy over completeness.
"""


def _build_prompt(
    person: str,
    spending_df: pd.DataFrame,
    series: pd.Series,
    forecasts: dict,
    goals: dict | None,
    n_months: int,
) -> str:
    """Construct the user prompt from data context."""

    # ── Summary stats ────────────────────────────────────────
    lifestyle = spending_df[~spending_df["Category"].isin(EXCLUDE_FROM_LIFESTYLE)]
    total = lifestyle["Amount"].sum()
    avg = total / n_months if n_months else 0

    # Trend
    if len(series) >= 3:
        trend = series.iloc[-1] - series.iloc[0]
        trend_str = f"{'↑ increasing' if trend > 0 else '↓ decreasing'} ({abs(trend/series.iloc[0]*100):.0f}% over period)"
    else:
        trend_str = "insufficient data for trend"

    # Category breakdown
    cat_totals = (
        lifestyle.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
        .head(8)
    )
    cat_lines = "\n".join(
        f"  - {cat}: ฿{amt:,.0f} ({amt/total*100:.1f}%)"
        for cat, amt in cat_totals.items()
    )

    # Monthly series
    series_lines = "\n".join(
        f"  - {idx.strftime('%b %Y')}: ฿{val:,.0f}"
        for idx, val in series.items()
    )

    # Forecast
    fc_lines = ""
    for name, (fc, lo, hi) in (forecasts or {}).items():
        if fc is not None and len(fc) > 0:
            vals = ", ".join(f"฿{v:,.0f}" for v in fc.values[:3])
            fc_lines += f"  - {name}: {vals}\n"
    if not fc_lines:
        fc_lines = "  - No forecast available\n"

    # Goals
    if goals and goals.get("monthly_savings_target"):
        target = float(goals["monthly_savings_target"])
        gap = avg - target
        caps = goals.get("category_caps", {})
        cap_lines = "\n".join(
            f"  - {cat}: ฿{cap:,.0f}/mo cap"
            for cat, cap in (caps or {}).items()
            if cap > 0
        ) or "  - No per-category caps set"
        goals_section = f"""
## Saving goals
- Monthly savings target: ฿{target:,.0f}
- Current monthly average: ฿{avg:,.0f}
- Gap: ฿{gap:,.0f} per month ({'need to reduce' if gap > 0 else 'already on track ✓'})
- Per-category caps:
{cap_lines}
"""
    else:
        goals_section = "\n## Saving goals\n- No saving goals set.\n"

    prompt = f"""Analyze the following spending data for {person} and write the report.

## Period covered
- {n_months} months of data
- Total lifestyle spend: ฿{total:,.0f}
- Monthly average: ฿{avg:,.0f}
- Trend: {trend_str}

## Top categories
{cat_lines}

## Monthly breakdown
{series_lines}

## 3-month forecast
{fc_lines}
{goals_section}
---
Now write the full report following the section structure in your system instructions.
Be specific: reference actual categories and ฿ amounts from the data above.
"""
    return prompt


def generate_insights(
    person: str,
    spending_df: pd.DataFrame,
    series: pd.Series,
    forecasts: dict,
    goals: dict | None,
    api_key: str,
    model: str = _MODEL,
    force_refresh: bool = False,
    max_age_days: int = 7,
) -> tuple[str, bool]:
    """
    Generate a behavioral insights report for a person.

    Parameters
    ----------
    force_refresh : bool
        If True, skip cache and always call Groq.

    Returns
    -------
    (report_markdown, from_cache)
    """
    if spending_df.empty:
        return "_No spending data available to analyse._", False

    # Derive cache inputs
    date_start = series.index.min() if not series.empty else None
    date_end = series.index.max() if not series.empty else None
    lifestyle = spending_df[~spending_df["Category"].isin(EXCLUDE_FROM_LIFESTYLE)]
    top_cats = (
        lifestyle.groupby("Category")["Amount"]
        .sum()
        .nlargest(6)
        .index.tolist()
    )
    n_months = series.nunique() if not series.empty else 1

    # Check cache (unless force_refresh)
    if not force_refresh:
        cached = cache.get_cached_report(
            person, date_start, date_end, top_cats, goals, max_age_days
        )
        if cached:
            return cached, True

    # Build prompt and call Groq
    if not GROQ_AVAILABLE:
        return "_Groq package not installed. Run: `pip install groq`_", False

    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return "_No Groq API key configured._", False

    prompt = _build_prompt(person, spending_df, series, forecasts, goals, n_months)

    try:
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        report = response.choices[0].message.content or ""

        # Store in cache
        cache.store_report(
            report_markdown=report,
            person=person,
            model_used=model,
            date_start=date_start,
            date_end=date_end,
            top_categories=top_cats,
            goals=goals,
        )
        return report, False

    except Exception as exc:
        return f"_Report generation failed: {exc}_", False
