"""Page 3 — Short-Term Signals.

9 technical indicator signals with composite aggregate, heatmap, and detail view.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import signal_heatmap, candlestick
from dashboard.components.tables import style_signal_table, style_generic

st.set_page_config(page_title="Short-Term Signals", page_icon="📡", layout="wide")
st.title("📡 Short-Term Technical Signals")
st.caption("9 indicator signals with weighted composite score — RSI, MACD, Bollinger, Stochastic, EMA/SMA, Volume, Momentum, Fundamentals, Correlation.")
st.divider()

settings = get_settings()


# ---------------------------------------------------------------------------#
# Ticker selector
# ---------------------------------------------------------------------------#

daily_tickers = sorted([fp.stem for fp in settings.raw_daily_dir.glob("*.parquet")]) \
    if settings.raw_daily_dir.exists() else []

if not daily_tickers:
    st.warning("No price data found. Run `python cli/scrape.py --backfill 2015` first.")
    st.stop()

col1, col2 = st.columns([2, 1])
with col1:
    selected = st.selectbox("Ticker", daily_tickers,
                             index=daily_tickers.index("AAPL") if "AAPL" in daily_tickers else 0)
with col2:
    lookback = st.number_input("Lookback days (chart)", min_value=30, max_value=756, value=252)

# ---------------------------------------------------------------------------#
# Load price data
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300)
def _load_daily(ticker: str) -> pd.DataFrame:
    fp = settings.raw_daily_dir / f"{ticker}.parquet"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_parquet(fp)

df_daily = _load_daily(selected)

if df_daily.empty:
    st.error(f"No data for {selected}")
    st.stop()

df_chart = df_daily.tail(lookback)

# ---------------------------------------------------------------------------#
# Run indicators
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300, show_spinner="Computing indicators …")
def _run_indicators(ticker: str):
    from src.indicators.signal_aggregator import run_and_aggregate
    df = _load_daily(ticker)
    if df.empty:
        return None
    return run_and_aggregate(df, ticker)

agg = _run_indicators(selected)

if agg is None:
    st.error("Failed to compute indicators — check price data.")
    st.stop()

# ---------------------------------------------------------------------------#
# Summary banner
# ---------------------------------------------------------------------------#

signal_colors = {
    "Strong Buy":  "🟢",
    "Buy":         "🔵",
    "Neutral":     "⚪",
    "Sell":        "🔴",
    "Strong Sell": "🔴",
}
label = agg.signal.label()
icon  = signal_colors.get(label, "⚪")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Composite Signal", f"{icon} {label}")
col2.metric("Score [-2, +2]",   f"{agg.score:+.3f}")
col3.metric("Score [-1, +1]",   f"{agg.score_normalized:+.3f}")
col4.metric("Indicators OK",    f"{agg.succeeded} / {agg.succeeded + agg.failed}")

st.divider()

# ---------------------------------------------------------------------------#
# Heatmap
# ---------------------------------------------------------------------------#

st.subheader("Signal Heatmap")
st.plotly_chart(signal_heatmap(agg.results, selected), use_container_width=True)

# ---------------------------------------------------------------------------#
# Indicator detail table
# ---------------------------------------------------------------------------#

st.subheader("Indicator Details")

rows = []
for r in agg.results:
    rows.append({
        "Indicator":       r.indicator_name,
        "Signal":          r.signal.label() if r.error is None else "ERROR",
        "Value":           int(r.signal)     if r.error is None else "—",
        "Interpretation":  r.interpretation  if r.error is None else r.error[:80],
    })

detail_df = pd.DataFrame(rows)
st.dataframe(style_signal_table(detail_df), use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------#
# Price chart
# ---------------------------------------------------------------------------#

st.subheader(f"{selected} — Recent Price Action")
st.plotly_chart(candlestick(df_chart, selected), use_container_width=True)

# SMA / EMA overlay note
st.caption(
    "Tip: The Moving Averages indicator computes 50-day and 200-day SMA/EMA crossovers. "
    "A golden cross (50 > 200) generates a BUY signal; a death cross generates a SELL signal."
)
