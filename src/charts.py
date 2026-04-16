"""
Plotly chart builders for the Streamlit dashboard.

Bug fixes applied (all permanent):
  BUG-3: Never use 8-digit hex (#RRGGBBAA) for Plotly fillcolor.
          All transparency uses rgba(r,g,b,alpha) via _rgba() helper.
  BUG-4: Never pass annotation_text to add_vline() when x-axis is
          Timestamp.  Always split into add_vline() + add_annotation().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import CATEGORY_COLORS, PERSON_COLORS

_TPL  = "plotly_white"
_MARG = dict(l=4, r=4, t=10, b=4)


def _color(person: str) -> str:
    return PERSON_COLORS.get(person, "#888780")


def _rgba(hex6: str, alpha: float) -> str:
    """Convert a 6-digit hex color to rgba() string.  BUG-3 fix."""
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def monthly_trend_chart(
    df: pd.DataFrame,
    person: str,
    split_source: bool = False,
) -> go.Figure:
    if df.empty:
        return go.Figure()

    fig    = go.Figure()
    monthly = df.groupby(["YearMonth", "source"])["Amount"].sum().reset_index()
    monthly["YearMonth"] = pd.to_datetime(monthly["YearMonth"])

    if split_source:
        src_colors  = {"bank": _color(person), "cc": "#EF9F27"}
        src_labels  = {"bank": "Bank",          "cc": "Credit card"}
        for src in ["bank", "cc"]:
            subset = monthly[monthly["source"] == src]
            if not subset.empty:
                fig.add_trace(go.Bar(
                    x=subset["YearMonth"], y=subset["Amount"],
                    name=src_labels[src], marker_color=src_colors[src],
                ))
        fig.update_layout(barmode="stack")
    else:
        total = monthly.groupby("YearMonth")["Amount"].sum().reset_index()
        fig.add_trace(go.Bar(
            x=total["YearMonth"], y=total["Amount"],
            name="Total spend", marker_color=_color(person),
        ))
        if len(total) >= 3:
            xi      = np.arange(len(total))
            z       = np.polyfit(xi, total["Amount"].values, 1)
            trend_y = np.poly1d(z)(xi)
            fig.add_trace(go.Scatter(
                x=total["YearMonth"], y=trend_y,
                name="Trend",
                line=dict(color="#888780", dash="dash", width=1.5),
                mode="lines",
            ))

    fig.update_layout(
        template=_TPL,
        xaxis_title="",
        yaxis_title="Amount (฿)",
        yaxis_tickformat=",.0f",
        legend=dict(orientation="h", y=-0.22, x=0),
        height=320,
        margin=_MARG,
    )
    return fig


def category_donut(df: pd.DataFrame, min_pct: float = 2.0) -> go.Figure:
    if df.empty:
        return go.Figure()

    totals = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
    total  = totals.sum()
    main   = totals[totals / total * 100 >= min_pct]
    other  = totals[totals / total * 100 < min_pct].sum()
    if other > 0:
        main = pd.concat([main, pd.Series({"Other": other})])

    colors = [CATEGORY_COLORS.get(c, "#D3D1C7") for c in main.index]

    fig = go.Figure(go.Pie(
        labels=main.index, values=main.values,
        hole=0.55, marker_colors=colors,
        textinfo="percent",
        hovertemplate="%{label}<br>฿%{value:,.0f} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        template=_TPL, showlegend=True,
        legend=dict(orientation="v", x=1.01, y=0.5, font_size=11),
        height=300, margin=dict(l=4, r=140, t=10, b=4),
    )
    return fig


def category_bar(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    if df.empty:
        return go.Figure()

    totals = df.groupby("Category")["Amount"].sum().nlargest(top_n).sort_values()
    colors = [CATEGORY_COLORS.get(c, "#D3D1C7") for c in totals.index]

    fig = go.Figure(go.Bar(
        x=totals.values, y=totals.index, orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>฿%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        template=_TPL,
        xaxis_title="Amount (฿)", xaxis_tickformat=",.0f",
        height=max(260, top_n * 32),
        margin=_MARG,
    )
    return fig


def forecast_chart(series: pd.Series, forecasts: dict, person: str) -> go.Figure:
    """
    Historical series + multi-model forecast lines with CI bands.

    BUG-3 fix: fillcolor uses _rgba() — never 8-digit hex.
    BUG-4 fix: vline annotation split into add_vline() + add_annotation().
    """
    if series.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        name="Actual",
        line=dict(color=_color(person), width=2.5),
        mode="lines+markers", marker=dict(size=7),
    ))

    palette = ["#378ADD", "#1D9E75", "#EF9F27", "#D85A30", "#7F77DD", "#534AB7"]

    for i, (name, (fc, lo, hi)) in enumerate(forecasts.items()):
        if fc is None:
            continue
        c = palette[i % len(palette)]

        # CI band — BUG-3: use rgba, never 8-digit hex
        if lo is not None and hi is not None:
            band_x = list(hi.index) + list(lo.index[::-1])
            band_y = list(hi.values) + list(lo.values[::-1])
            fig.add_trace(go.Scatter(
                x=band_x, y=band_y,
                fill="toself",
                fillcolor=_rgba(c, 0.10),   # ← BUG-3 fix
                line_color="rgba(0,0,0,0)",
                showlegend=False, hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter(
            x=fc.index, y=fc.values,
            name=name,
            line=dict(color=c, width=1.8, dash="dot"),
            mode="lines+markers", marker=dict(size=5),
            hovertemplate=f"{name}<br>%{{x|%b %Y}}<br>฿%{{y:,.0f}}<extra></extra>",
        ))

    # BUG-4 fix: split add_vline + add_annotation (no annotation_text on Timestamp x-axis)
    if not series.empty:
        vline_x = series.index[-1].isoformat()
        fig.add_vline(
            x=vline_x,
            line_dash="dash",
            line_color="#B4B2A9",
            line_width=1,
        )
        fig.add_annotation(
            x=vline_x,
            y=1, yref="paper",
            text=" forecast →",
            showarrow=False,
            xanchor="left",
            font=dict(size=11, color="#B4B2A9"),
        )

    fig.update_layout(
        template=_TPL,
        xaxis_title="",
        yaxis_title="Amount (฿)",
        yaxis_tickformat=",.0f",
        height=400,
        legend=dict(orientation="h", y=-0.2, x=0),
        margin=_MARG,
    )
    return fig


def comparison_chart(series_dict: dict) -> go.Figure:
    fig = go.Figure()
    for person, series in series_dict.items():
        if series is None or series.empty:
            continue
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            name=person,
            line=dict(color=PERSON_COLORS.get(person, "#888780"), width=2.2),
            mode="lines+markers", marker=dict(size=6),
            hovertemplate=f"{person}<br>%{{x|%b %Y}}<br>฿%{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        template=_TPL,
        xaxis_title="",
        yaxis_title="Monthly spend (฿)",
        yaxis_tickformat=",.0f",
        height=320,
        legend=dict(orientation="h", y=-0.2),
        margin=_MARG,
    )
    return fig


def calendar_heatmap(df: pd.DataFrame, person: str) -> go.Figure:
    """Day-of-month × calendar-month heatmap."""
    if df.empty:
        return go.Figure()

    d = df.copy()
    d["Day"]       = d["Date"].dt.day
    d["MonthNum"]  = d["Date"].dt.month

    agg = (
        d.groupby(["Day", "MonthNum"])["Amount"]
        .sum()
        .reset_index()
    )

    all_days   = list(range(1, 32))
    all_months = list(range(1, 13))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    pivot = (
        agg.pivot_table(values="Amount", index="Day", columns="MonthNum",
                        aggfunc="sum", fill_value=0)
        .reindex(index=all_days, columns=all_months, fill_value=0)
    )

    z = pivot.values.astype(float)
    z[z == 0] = float("nan")

    fig = go.Figure(go.Heatmap(
        z=z, x=month_labels, y=all_days,
        colorscale=[[0, "#F1EFE8"], [0.5, _rgba(_color(person), 0.6)], [1, _color(person)]],
        showscale=True,
        hoverongaps=False,
        hovertemplate="<b>%{x} %{y}</b><br>฿%{z:,.0f}<extra></extra>",
        colorbar=dict(tickformat=",.0f", title=dict(text="฿", side="right"), thickness=12),
    ))
    fig.update_layout(
        template=_TPL,
        xaxis=dict(title="", side="top", tickmode="array", tickvals=month_labels),
        yaxis=dict(title="Day of month", autorange="reversed", dtick=5, tickmode="linear"),
        height=520,
        margin=dict(l=40, r=40, t=40, b=10),
    )
    return fig


def category_monthly_stack(df: pd.DataFrame, person: str, top_n: int = 8) -> go.Figure:
    if df.empty:
        return go.Figure()

    top_cats = (
        df.groupby("Category")["Amount"].sum().nlargest(top_n).index.tolist()
    )
    sub   = df[df["Category"].isin(top_cats)].copy()
    pivot = (
        sub.groupby(["YearMonth", "Category"])["Amount"]
        .sum().unstack(fill_value=0).sort_index()
    )

    fig = go.Figure()
    for cat in top_cats:
        if cat not in pivot.columns:
            continue
        fig.add_trace(go.Bar(
            x=pivot.index, y=pivot[cat], name=cat,
            marker_color=CATEGORY_COLORS.get(cat, "#D3D1C7"),
            hovertemplate=f"{cat}<br>%{{x|%b %Y}}<br>฿%{{y:,.0f}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack", template=_TPL,
        xaxis_title="",
        yaxis_title="Amount (฿)", yaxis_tickformat=",.0f",
        height=340,
        legend=dict(orientation="h", y=-0.25, x=0),
        margin=_MARG,
    )
    return fig


def waterfall_chart(series: pd.Series, person: str) -> go.Figure:
    if len(series) < 2:
        return go.Figure()

    diff   = series.diff().dropna()
    labels = [d.strftime("%b %Y") for d in diff.index]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(diff),
        x=labels, y=diff.values,
        connector=dict(line=dict(color="#D3D1C7", width=0.5)),
        increasing_marker_color=_color(person),
        decreasing_marker_color="#D85A30",
        texttemplate="%{y:+,.0f}",
        textposition="outside",
    ))
    fig.update_layout(
        template=_TPL,
        xaxis_title="",
        yaxis_title="Month-over-month change (฿)", yaxis_tickformat=",.0f",
        height=300, margin=_MARG, showlegend=False,
    )
    return fig


def goals_progress_chart(
    series: pd.Series,
    monthly_target: float,
    person: str,
) -> go.Figure:
    """Actual monthly spend vs saving goal target."""
    if series.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=series.index, y=series.values,
        name="Actual spend",
        marker_color=_color(person),
        hovertemplate="%{x|%b %Y}<br>Actual: ฿%{y:,.0f}<extra></extra>",
    ))
    if monthly_target > 0:
        fig.add_hline(
            y=monthly_target,
            line_dash="dash",
            line_color="#1D9E75",
            line_width=1.5,
        )
        fig.add_annotation(
            x=1, xref="paper",
            y=monthly_target,
            text=f" target ฿{monthly_target:,.0f}",
            showarrow=False,
            xanchor="left",
            font=dict(size=11, color="#1D9E75"),
        )

    fig.update_layout(
        template=_TPL,
        xaxis_title="",
        yaxis_title="Monthly spend (฿)", yaxis_tickformat=",.0f",
        height=300, margin=_MARG,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig
