from __future__ import annotations

import pandas as pd

"""
Purpose: Shared Streamlit DataFrame renderers — returns colour-coded Styler objects for ranking, signals, forecasts, ML, sentiment, congress, and generic tables.

Connections:
  - dashboard/pages/*: imported by every dashboard page that shows a table
  - dashboard/components/charts.py: shares colour constants (_GREEN, _RED)

In:  pandas DataFrame from any signal/data source
Out: pandas Styler ready for st.dataframe()
"""

# ---------------------------------------------------------------------------#
# Colour constants (matches charts.py)
# ---------------------------------------------------------------------------#

_GREEN   = "#26a69a"
_RED     = "#ef5350"
_NEUTRAL = "#2a2a3a"
_YELLOW  = "#ffa726"
_BLUE    = "#1e3a5f"

_ROW_ODD  = "#0e1117"
_ROW_EVEN = "#161b22"


# ---------------------------------------------------------------------------#
# Helpers
# ---------------------------------------------------------------------------#

def _score_color(val: float) -> str:
    """Map a [-1, +1] score to a background colour."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v >= 0.5:
        return f"background-color: {_GREEN}; color: white"
    if v >= 0.1:
        return f"background-color: #1b5e20; color: #a5d6a7"
    if v <= -0.5:
        return f"background-color: {_RED}; color: white"
    if v <= -0.1:
        return f"background-color: #7f0000; color: #ef9a9a"
    return f"background-color: {_NEUTRAL}; color: #888"


def _signal_color(val) -> str:
    """Colour for textual signal labels (Strong Buy … Strong Sell)."""
    s = str(val).lower()
    if "strong buy" in s:
        return f"background-color: {_GREEN}; color: white; font-weight: bold"
    if "buy" in s:
        return f"background-color: #1b5e20; color: #a5d6a7"
    if "strong sell" in s:
        return f"background-color: {_RED}; color: white; font-weight: bold"
    if "sell" in s:
        return f"background-color: #7f0000; color: #ef9a9a"
    return f"background-color: {_NEUTRAL}; color: #888"


def _positive_red_color(val) -> str:
    """Sells highlighted red, buys green."""
    s = str(val).lower()
    if s in ("purchase", "buy", "p", "b"):
        return f"color: {_GREEN}; font-weight: bold"
    if s in ("sale", "sell", "s"):
        return f"color: {_RED}; font-weight: bold"
    return ""


def _beat_color(val) -> str:
    """Beat rate > 60 % → green, < 40 % → red."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v >= 0.6:
        return f"color: {_GREEN}"
    if v <= 0.4:
        return f"color: {_RED}"
    return f"color: {_YELLOW}"


# ---------------------------------------------------------------------------#
# Public renderers
# ---------------------------------------------------------------------------#

def style_ranking_table(df: pd.DataFrame):
    """Style the top-N composite ranking table."""
    if df.empty:
        return df.style

    score_cols = [c for c in df.columns if c in (
        "Score", "Forecast", "Indicators", "ML", "News",
        "Short Int.", "Congress", "Insider", "Options", "IV", "Earnings"
    )]

    styler = df.style.format(
        {c: "{:.3f}" for c in score_cols if c in df.columns},
        na_rep="—"
    )

    for col in score_cols:
        if col in df.columns:
            styler = styler.map(_score_color, subset=[col])

    # Rank column bold
    if "Rank" in df.columns:
        styler = styler.map(lambda _: "font-weight: bold", subset=["Rank"])

    return styler.set_properties(**{"font-size": "13px"})


def style_signal_table(df: pd.DataFrame):
    """Style indicator signal results table."""
    if df.empty:
        return df.style

    signal_cols = [c for c in df.columns if "signal" in c.lower()]

    styler = df.style.format(na_rep="—")
    for col in signal_cols:
        styler = styler.map(_signal_color, subset=[col])

    return styler.set_properties(**{"font-size": "13px"})


def style_forecast_table(df: pd.DataFrame):
    """Style forecast method comparison table — highlight best RMSE row."""
    if df.empty:
        return df.style

    styler = df.style.format(na_rep="—")

    # Highlight the row with the lowest numeric RMSE
    if "RMSE" in df.columns:
        numeric_rmse = pd.to_numeric(df["RMSE"], errors="coerce")
        if not numeric_rmse.isna().all():
            best_idx = numeric_rmse.idxmin()

            def highlight_best(row):
                if row.name == best_idx:
                    return [f"background-color: {_GREEN}22; border-left: 3px solid {_GREEN}"] * len(row)
                return [""] * len(row)

            styler = styler.apply(highlight_best, axis=1)

    return styler.set_properties(**{"font-size": "13px"})


def style_ml_table(df: pd.DataFrame):
    """Style ML model comparison table — highlight best RMSE row."""
    if df.empty:
        return df.style

    styler = df.style.format(na_rep="—")

    if "RMSE" in df.columns:
        numeric_rmse = pd.to_numeric(df["RMSE"], errors="coerce")
        if not numeric_rmse.isna().all():
            best_idx = numeric_rmse.idxmin()

            def highlight_best(row):
                if row.name == best_idx:
                    return [f"background-color: {_BLUE}; border-left: 3px solid #5c7cfa"] * len(row)
                return [""] * len(row)

            styler = styler.apply(highlight_best, axis=1)

    return styler.set_properties(**{"font-size": "13px"})


def style_sentiment_table(df: pd.DataFrame):
    """Style news articles with sentiment score colour coding."""
    if df.empty:
        return df.style

    styler = df.style.format(na_rep="—")
    score_col = next((c for c in df.columns if "score" in c.lower()), None)
    if score_col:
        styler = styler.map(_score_color, subset=[score_col])

    label_col = next((c for c in df.columns if "label" in c.lower() or "sentiment" in c.lower()), None)
    if label_col:
        styler = styler.map(lambda v: _signal_color(v.replace("positive", "buy").replace("negative", "sell")),
                            subset=[label_col])

    return styler.set_properties(**{"font-size": "13px"})


def style_congress_table(df: pd.DataFrame):
    """Style congressional trades — purchases green, sales red."""
    if df.empty:
        return df.style

    styler = df.style.format(na_rep="—")
    tx_col = next((c for c in df.columns if "transaction" in c.lower()), None)
    if tx_col:
        styler = styler.map(_positive_red_color, subset=[tx_col])

    return styler.set_properties(**{"font-size": "13px"})


def style_insider_table(df: pd.DataFrame):
    """Style insider filings — buys green, sells red."""
    if df.empty:
        return df.style

    styler = df.style.format(na_rep="—")
    tx_col = next((c for c in df.columns if "transaction" in c.lower() or "type" in c.lower()), None)
    if tx_col:
        styler = styler.map(_positive_red_color, subset=[tx_col])

    return styler.set_properties(**{"font-size": "13px"})


def style_options_table(df: pd.DataFrame):
    """Style options metrics snapshot."""
    if df.empty:
        return df.style

    styler = df.style.format(na_rep="—")
    signal_col = next((c for c in df.columns if "signal" in c.lower()), None)
    if signal_col:
        styler = styler.map(_score_color, subset=[signal_col])

    return styler.set_properties(**{"font-size": "13px"})


def style_earnings_table(df: pd.DataFrame):
    """Style earnings history with beat/miss colour coding."""
    if df.empty:
        return df.style

    styler = df.style.format(na_rep="—")
    beat_col = next((c for c in df.columns if "beat" in c.lower()), None)
    if beat_col:
        styler = styler.map(_beat_color, subset=[beat_col])

    return styler.set_properties(**{"font-size": "13px"})


def style_generic(df: pd.DataFrame):
    """Minimal alternating-row shading for any DataFrame."""
    if df.empty:
        return df.style

    return df.style.set_properties(**{"font-size": "13px"}).format(na_rep="—")
