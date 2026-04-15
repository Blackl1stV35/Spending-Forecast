"""
Plotly chart builders for the Streamlit dashboard.
All functions return a go.Figure ready for st.plotly_chart().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import CATEGORY_COLORS, PERSON_COLORS

_TPL = "plotly_white"
_MARG = dict(l=4, r=4, t=10, b=4)


def _color(person: str) -> str:
    return PERSON_COLORS.get(person, "#888780")


def monthly_trend_chart(
    df: pd.DataFrame,
    person: str,
    split_source: bool = False,
) -> go.Figure:
    """Monthly spending bar chart with optional bank/CC split and trend line."""
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    monthly = df.groupby(["YearMonth", "source"])["Amount"].sum().reset_index()
    monthly["YearMonth"] = pd.to_datetime(monthly["YearMonth"])

    if split_source:
        src_colors = {"bank": _color(person), "cc": "#EF9F27"}
        src_labels = {"bank": "Bank", "cc": "Credit card"}
        for src in ["bank", "cc"]:
            subset = monthly[monthly["source"] == src]
            if not subset.empty:
                fig.add_trace(
                    go.Bar(
                        x=subset["YearMonth"],
                        y=subset["Amount"],
                        name=src_labels[src],
                        marker_color=src_colors[src],
                    )
                )
        fig.update_layout(barmode="stack")
    else:
        total = monthly.groupby("YearMonth")["Amount"].sum().reset_index()
        fig.add_trace(
            go.Bar(
                x=total["YearMonth"],
                y=total["Amount"],
                name="Total spend",
                marker_color=_color(person),
            )
        )
        if len(total) >= 3:
            xi = np.arange(len(total))
            z = np.polyfit(xi, total["Amount"].values, 1)
            trend_y = np.poly1d(z)(xi)
            fig.add_trace(
                go.Scatter(
                    x=total["YearMonth"],
                    y=trend_y,
                    name="Trend",
                    line=dict(color="#888780", dash="dash", width=1.5),
                    mode="lines",
                )
            )

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
    """Donut chart of spending by category."""
    if df.empty:
        return go.Figure()

    totals = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
    total = totals.sum()
    main = totals[totals / total * 100 >= min_pct]
    other = totals[totals / total * 100 < min_pct].sum()
    if other > 0:
        main = pd.concat([main, pd.Series({"Other": other})])

    colors = [CATEGORY_COLORS.get(c, "#D3D1C7") for c in main.index]

    fig = go.Figure(
        go.Pie(
            labels=main.index,
            values=main.values,
            hole=0.55,
            marker_colors=colors,
            textinfo="percent",
            hovertemplate="%{label}<br>฿%{value:,.0f} (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        template=_TPL,
        showlegend=True,
        legend=dict(orientation="v", x=1.01, y=0.5, font_size=11),
        height=300,
        margin=dict(l=4, r=140, t=10, b=4),
    )
    return fig


def category_bar(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Horizontal bar chart of top-N categories."""
    if df.empty:
        return go.Figure()

    totals = df.groupby("Category")["Amount"].sum().nlargest(top_n).sort_values()
    colors = [CATEGORY_COLORS.get(c, "#D3D1C7") for c in totals.index]

    fig = go.Figure(
        go.Bar(
            x=totals.values,
            y=totals.index,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>฿%{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        template=_TPL,
        xaxis_title="Amount (฿)",
        xaxis_tickformat=",.0f",
        height=max(260, top_n * 32),
        margin=_MARG,
    )
    return fig


def forecast_chart(series: pd.Series, forecasts: dict, person: str) -> go.Figure:
    """Historical series + multi-model forecast lines with CI bands."""
    if series.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            name="Actual",
            line=dict(color=_color(person), width=2.5),
            mode="lines+markers",
            marker=dict(size=7),
        )
    )

    palette = ["#378ADD", "#1D9E75", "#EF9F27", "#D85A30", "#7F77DD", "#534AB7"]

    for i, (name, (fc, lo, hi)) in enumerate(forecasts.items()):
        if fc is None:
            continue
        c = palette[i % len(palette)]

        if lo is not None and hi is not None:
            band_x = list(hi.index) + list(lo.index[::-1])
            band_y = list(hi.values) + list(lo.values[::-1])
        
            h = c.lstrip('#')
            r, g, b = tuple(int(h[j:j+2], 16) for j in (0, 2, 4))
            rgba_color = f"rgba({r}, {g}, {b}, 0.12)" 

            fig.add_trace(
                go.Scatter(
                    x=band_x,
                    y=band_y,
                    fill="toself",
                    fillcolor=rgba_color,  
                    line_color="rgba(0,0,0,0)",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=fc.index,
                y=fc.values,
                name=name,
                line=dict(color=c, width=1.8, dash="dot"),
                mode="lines+markers",
                marker=dict(size=5),
                hovertemplate=f"{name}<br>%{{x|%b %Y}}<br>฿%{{y:,.0f}}<extra></extra>",
            )
        )

    fig.add_vline(
        x=series.index[-1],
        line_dash="dash",
        line_color="#B4B2A9",
        line_width=1,
    )

    fig.add_annotation(
        x=series.index[-1],
        y=1,
        yref="paper",
        text=" forecast →",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=11),
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
    """Side-by-side monthly spending lines for multiple people."""
    fig = go.Figure()
    for person, series in series_dict.items():
        if series is None or series.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                name=person,
                line=dict(color=PERSON_COLORS.get(person, "#888780"), width=2.2),
                mode="lines+markers",
                marker=dict(size=6),
                hovertemplate=f"{person}<br>%{{x|%b %Y}}<br>฿%{{y:,.0f}}<extra></extra>",
            )
        )
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
    """Weekly heatmap: rows = day of week, columns = ISO week number."""
    if df.empty:
        return go.Figure()

    d = df.copy()
    d["DOW"] = d["Date"].dt.dayofweek
    d["Week"] = d["Date"].dt.isocalendar().week.astype(int)

    pivot = d.pivot_table(values="Amount", index="DOW", columns="Week", aggfunc="sum", fill_value=0)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=[days[i] for i in pivot.index],
            colorscale=[[0, "#F1EFE8"], [1, _color(person)]],
            hovertemplate="Wk %{x}, %{y}<br>฿%{z:,.0f}<extra></extra>",
            showscale=True,
        )
    )
    fig.update_layout(
        template=_TPL,
        xaxis_title="ISO week",
        yaxis_title="",
        height=230,
        margin=_MARG,
    )
    return fig


def category_monthly_stack(df: pd.DataFrame, person: str, top_n: int = 8) -> go.Figure:
    """Stacked bar chart of top-N categories over months."""
    if df.empty:
        return go.Figure()

    top_cats = (
        df.groupby("Category")["Amount"].sum().nlargest(top_n).index.tolist()
    )
    sub = df[df["Category"].isin(top_cats)].copy()
    pivot = (
        sub.groupby(["YearMonth", "Category"])["Amount"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )

    fig = go.Figure()
    for cat in top_cats:
        if cat not in pivot.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=pivot.index,
                y=pivot[cat],
                name=cat,
                marker_color=CATEGORY_COLORS.get(cat, "#D3D1C7"),
                hovertemplate=f"{cat}<br>%{{x|%b %Y}}<br>฿%{{y:,.0f}}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        template=_TPL,
        xaxis_title="",
        yaxis_title="Amount (฿)",
        yaxis_tickformat=",.0f",
        height=340,
        legend=dict(orientation="h", y=-0.25, x=0),
        margin=_MARG,
    )
    return fig


def waterfall_chart(series: pd.Series, person: str) -> go.Figure:
    """Waterfall chart showing month-over-month change."""
    if len(series) < 2:
        return go.Figure()

    diff = series.diff().dropna()
    labels = [d.strftime("%b %Y") for d in diff.index]
    colors = [_color(person) if v >= 0 else "#D85A30" for v in diff.values]

    fig = go.Figure(
        go.Waterfall(
            name="MoM change",
            orientation="v",
            measure=["relative"] * len(diff),
            x=labels,
            y=diff.values,
            connector=dict(line=dict(color="#D3D1C7", width=0.5)),
            increasing_marker_color=_color(person),
            decreasing_marker_color="#D85A30",
            texttemplate="%{y:+,.0f}",
            textposition="outside",
        )
    )
    fig.update_layout(
        template=_TPL,
        xaxis_title="",
        yaxis_title="Month-over-month change (฿)",
        yaxis_tickformat=",.0f",
        height=300,
        margin=_MARG,
        showlegend=False,
    )
    return fig
