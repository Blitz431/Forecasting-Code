from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

"""
Purpose: Shared Plotly figure builders — candlestick, forecast overlay, signal heatmap, ranking bars, sentiment timeline, vol chart, regime gauge, and more.

Connections:
  - dashboard/pages/*: imported by every dashboard page that shows a chart
  - dashboard/components/tables.py: shares colour palette (_GREEN, _RED, _BLUE)

In:  pandas DataFrame (price, signal, sentiment, ML results, etc.)
Out: plotly.graph_objects.Figure ready for st.plotly_chart()
"""

# ---------------------------------------------------------------------------#
# Theme defaults
# ---------------------------------------------------------------------------#

_TEMPLATE = "plotly_dark"
_BG       = "#0e1117"
_GRID     = "#2a2a3a"
_UP       = "#26a69a"   # green
_DOWN     = "#ef5350"   # red
_BLUE     = "#5c7cfa"
_ORANGE   = "#ffa726"
_PURPLE   = "#ab47bc"

_LAYOUT_BASE = dict(
    template=_TEMPLATE,
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(family="Inter, Arial, sans-serif", size=12, color="#e0e0e0"),
    margin=dict(l=50, r=20, t=50, b=40),
    xaxis=dict(gridcolor=_GRID, showgrid=True),
    yaxis=dict(gridcolor=_GRID, showgrid=True),
)


def _apply_base(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(**_LAYOUT_BASE, title=title)
    return fig


def _merged_layout(**overrides) -> dict:
    """Merge caller overrides into _LAYOUT_BASE without duplicating kwargs.

    xaxis/yaxis are deep-merged so callers can add fields (e.g. ``range``)
    without losing the base grid styling.
    """
    layout = {k: v for k, v in _LAYOUT_BASE.items()}
    for key, val in overrides.items():
        if key in ("xaxis", "yaxis") and isinstance(val, dict):
            layout[key] = {**_LAYOUT_BASE.get(key, {}), **val}
        else:
            layout[key] = val
    return layout


# ---------------------------------------------------------------------------#
# Price charts
# ---------------------------------------------------------------------------#

def candlestick(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """OHLCV candlestick chart with volume bar subplot."""
    if df.empty:
        return _empty_fig(f"{ticker} — No price data")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df.get("Open"),
            high=df.get("High"),
            low=df.get("Low"),
            close=df.get("Close"),
            name=ticker,
            increasing_line_color=_UP,
            decreasing_line_color=_DOWN,
        ),
        row=1, col=1,
    )

    if "Volume" in df.columns:
        colors = [_UP if c >= o else _DOWN
                  for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume",
                   marker_color=colors, opacity=0.7),
            row=2, col=1,
        )

    fig.update_layout(
        **_LAYOUT_BASE,
        title=f"{ticker} — OHLCV",
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(gridcolor=_GRID)
    return fig


def price_line(df: pd.DataFrame, ticker: str = "", col: str = "Close") -> go.Figure:
    """Simple close-price line chart."""
    if df.empty or col not in df.columns:
        return _empty_fig(f"{ticker} — No data")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df[col],
        mode="lines",
        name=ticker,
        line=dict(color=_BLUE, width=2),
    ))
    return _apply_base(fig, f"{ticker} — {col}")


def forecast_overlay(
    history: pd.DataFrame,
    results: list,          # list[ForecastResult]
    ticker: str = "",
) -> go.Figure:
    """Historical close + all 12 method forecasts overlaid."""
    fig = go.Figure()

    # History
    if not history.empty and "Close" in history.columns:
        fig.add_trace(go.Scatter(
            x=history.index, y=history["Close"],
            mode="lines",
            name="History",
            line=dict(color=_BLUE, width=2),
        ))

    # Forecast methods
    colors = px.colors.qualitative.Plotly
    for i, r in enumerate(results):
        if r.error is not None or not hasattr(r, "forecasts") or r.forecasts.empty:
            continue
        fig.add_trace(go.Scatter(
            x=r.forecasts.index,
            y=r.forecasts.values,
            mode="lines+markers",
            name=f"#{r.method_number} {r.method_name}",
            line=dict(color=colors[i % len(colors)], width=1, dash="dot"),
            marker=dict(size=5),
        ))

    return _apply_base(fig, f"{ticker} — Long-Term Forecasts")


# ---------------------------------------------------------------------------#
# Signal / ranking charts
# ---------------------------------------------------------------------------#

def signal_heatmap(indicator_results: list, ticker: str = "") -> go.Figure:
    """Coloured heatmap: one row of indicator signals for a single ticker."""
    if not indicator_results:
        return _empty_fig("No indicator data")

    names  = [r.indicator_name for r in indicator_results if not r.error]
    values = [int(r.signal) for r in indicator_results if not r.error]

    if not names:
        return _empty_fig("All indicators failed")

    fig = go.Figure(go.Heatmap(
        z=[values],
        x=names,
        y=[ticker],
        colorscale=[
            [0.0,  _DOWN],
            [0.25, "#ef9a9a"],
            [0.5,  "#424242"],
            [0.75, "#a5d6a7"],
            [1.0,  _UP],
        ],
        zmin=-2, zmax=2,
        showscale=True,
        colorbar=dict(
            title="Signal",
            tickvals=[-2, -1, 0, 1, 2],
            ticktext=["Strong Sell", "Sell", "Neutral", "Buy", "Strong Buy"],
        ),
        text=[[f"{v:+d}" for v in values]],
        texttemplate="%{text}",
    ))
    return _apply_base(fig, f"{ticker} — Indicator Signals")


def ranking_bar(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of composite scores for top-N picks."""
    if df.empty:
        return _empty_fig("No ranking data")

    df = df.head(top_n).copy()
    colors = [_UP if s >= 0 else _DOWN for s in df["Score"]]

    fig = go.Figure(go.Bar(
        x=df["Score"],
        y=df["Ticker"],
        orientation="h",
        marker_color=colors,
        text=[f"{s:.3f}" for s in df["Score"]],
        textposition="outside",
    ))
    fig.update_layout(**_merged_layout(
        title=f"Top {top_n} Composite Scores",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[-1.1, 1.1]),
        height=max(400, top_n * 28),
    ))
    return fig


def score_breakdown(signals: dict[str, float], ticker: str = "") -> go.Figure:
    """Stacked bar decomposing one ticker's individual signal contributions."""
    if not signals:
        return _empty_fig(f"{ticker} — No signal data")

    labels = list(signals.keys())
    values = list(signals.values())
    colors = [_UP if v >= 0 else _DOWN for v in values]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(**_merged_layout(
        title=f"{ticker} — Signal Breakdown",
        yaxis=dict(range=[-1.2, 1.2]),
    ))
    return fig


# ---------------------------------------------------------------------------#
# News sentiment
# ---------------------------------------------------------------------------#

def sentiment_timeline(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """Daily average sentiment score over time."""
    if df.empty:
        return _empty_fig(f"{ticker} — No sentiment data")

    score_col = "avg_score" if "avg_score" in df.columns else df.columns[0]
    colors = [_UP if s >= 0 else _DOWN for s in df[score_col]]

    fig = go.Figure(go.Bar(
        x=df.index,
        y=df[score_col],
        marker_color=colors,
        name="Sentiment",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#888")
    fig.add_hline(y=0.1, line_dash="dot", line_color=_UP, opacity=0.5)
    fig.add_hline(y=-0.1, line_dash="dot", line_color=_DOWN, opacity=0.5)
    return _apply_base(fig, f"{ticker} — Daily Sentiment Score")


# ---------------------------------------------------------------------------#
# Options
# ---------------------------------------------------------------------------#

def options_chart(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """Put/call ratio timeline."""
    if df.empty or "put_call_ratio" not in df.columns:
        return _empty_fig(f"{ticker} — No options data")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.05)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["put_call_ratio"],
        mode="lines+markers", name="P/C Ratio",
        line=dict(color=_ORANGE, width=2),
    ), row=1, col=1)
    fig.add_hline(y=0.7, line_dash="dot", line_color=_UP,   annotation_text="Bullish 0.7", row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dot", line_color=_DOWN, annotation_text="Bearish 1.0",  row=1, col=1)

    if "flow_signal" in df.columns:
        bar_colors = [_UP if v >= 0 else _DOWN for v in df["flow_signal"]]
        fig.add_trace(go.Bar(
            x=df.index, y=df["flow_signal"],
            name="Flow Signal", marker_color=bar_colors,
        ), row=2, col=1)

    fig.update_layout(**_LAYOUT_BASE, title=f"{ticker} — Options Flow")
    fig.update_yaxes(gridcolor=_GRID)
    return fig


def vol_chart(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """Implied vol vs historical vol overlay."""
    if df.empty:
        return _empty_fig(f"{ticker} — No vol data")

    fig = go.Figure()
    if "implied_vol" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["implied_vol"] * 100,
            mode="lines", name="IV (%)", line=dict(color=_ORANGE, width=2),
        ))
    if "historical_vol" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["historical_vol"] * 100,
            mode="lines", name="HV (%)", line=dict(color=_BLUE, width=2, dash="dash"),
        ))
    return _apply_base(fig, f"{ticker} — Implied vs Historical Volatility")


# ---------------------------------------------------------------------------#
# Portfolio / analytics
# ---------------------------------------------------------------------------#

def correlation_matrix(corr_df: pd.DataFrame) -> go.Figure:
    """Heatmap of a correlation matrix."""
    if corr_df.empty:
        return _empty_fig("No correlation data")

    fig = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns.tolist(),
        y=corr_df.index.tolist(),
        colorscale="RdBu",
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlation"),
        text=corr_df.round(2).values,
        texttemplate="%{text}",
    ))
    return _apply_base(fig, "Correlation Matrix")


def sector_donut(sector_series: pd.Series) -> go.Figure:
    """Donut chart of sector weights."""
    if sector_series.empty:
        return _empty_fig("No sector data")

    fig = go.Figure(go.Pie(
        labels=sector_series.index,
        values=sector_series.values,
        hole=0.4,
        textinfo="label+percent",
    ))
    return _apply_base(fig, "Sector Breakdown")


# ---------------------------------------------------------------------------#
# ML / backtesting
# ---------------------------------------------------------------------------#

def ml_comparison_bar(df: pd.DataFrame) -> go.Figure:
    """Model RMSE comparison bar chart."""
    if df.empty or "RMSE" not in df.columns:
        return _empty_fig("No ML results")

    numeric = pd.to_numeric(df["RMSE"], errors="coerce").dropna()
    plot_df = df.loc[numeric.index].copy()
    plot_df["_rmse"] = numeric

    fig = go.Figure(go.Bar(
        x=plot_df["Model"],
        y=plot_df["_rmse"],
        marker_color=_BLUE,
        text=[f"{v:.5f}" for v in plot_df["_rmse"]],
        textposition="outside",
    ))
    return _apply_base(fig, "ML Model RMSE Comparison")


def feature_importance(feat_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of top feature importances."""
    if feat_df.empty:
        return _empty_fig("No feature importance data")

    df = feat_df.nlargest(top_n, feat_df.columns[1]) if len(feat_df.columns) > 1 else feat_df.head(top_n)
    feature_col  = df.columns[0]
    importance_col = df.columns[1]

    fig = go.Figure(go.Bar(
        x=df[importance_col],
        y=df[feature_col],
        orientation="h",
        marker_color=_PURPLE,
    ))
    fig.update_layout(**_merged_layout(
        title=f"Top {top_n} Feature Importances",
        yaxis=dict(autorange="reversed"),
        height=max(400, top_n * 25),
    ))
    return fig


# ---------------------------------------------------------------------------#
# Market regime
# ---------------------------------------------------------------------------#

def regime_gauge(vix: float | None = None, regime: str = "Unknown") -> go.Figure:
    """Gauge-style indicator showing VIX level and regime label."""
    vix_val = vix if vix is not None else 20.0

    color = _UP if regime in ("Bull", "Recovery") else \
            _DOWN if regime in ("Bear", "Crisis") else _ORANGE

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=vix_val,
        title=dict(text=f"VIX — Market Regime: {regime}", font=dict(size=14)),
        gauge=dict(
            axis=dict(range=[0, 80], tickwidth=1),
            bar=dict(color=color),
            bgcolor=_BG,
            borderwidth=1,
            steps=[
                dict(range=[0, 15],  color="#1b5e20"),
                dict(range=[15, 25], color="#f57f17"),
                dict(range=[25, 40], color="#b71c1c"),
                dict(range=[40, 80], color="#880e4f"),
            ],
            threshold=dict(line=dict(color="white", width=3), thickness=0.75, value=vix_val),
        ),
        number=dict(suffix=" VIX"),
    ))
    fig.update_layout(paper_bgcolor=_BG, font_color="#e0e0e0", height=280)
    return fig


# ---------------------------------------------------------------------------#
# Calendar
# ---------------------------------------------------------------------------#

def earnings_timeline(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of upcoming earnings dates."""
    if df.empty:
        return _empty_fig("No earnings calendar data")

    fig = go.Figure(go.Scatter(
        x=df.get("next_earnings_date", df.index),
        y=df.get("Ticker", df.index),
        mode="markers+text",
        text=df.get("Ticker"),
        textposition="middle right",
        marker=dict(color=_ORANGE, size=10, symbol="diamond"),
    ))
    fig.update_layout(**_merged_layout(
        title="Upcoming Earnings",
        xaxis_title="Date",
        yaxis_title="Ticker",
        height=max(300, len(df) * 25),
    ))
    return fig


# ---------------------------------------------------------------------------#
# Helper
# ---------------------------------------------------------------------------#

def _empty_fig(message: str = "No data available") -> go.Figure:
    """Return a blank figure with a centred annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="#888"),
    )
    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=300,
    )
    return fig
