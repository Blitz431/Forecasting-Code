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
from dashboard.components.ticker_selector import render_ticker_sidebar

st.set_page_config(page_title="Short-Term Signals", page_icon="📡", layout="wide")
st.title("📡 Short-Term Technical Signals")
st.caption("9 indicator signals with weighted composite score — RSI, MACD, Bollinger, Stochastic, EMA/SMA, Volume, Momentum, Fundamentals, Correlation.")
st.divider()

settings = get_settings()


# ---------------------------------------------------------------------------#
# Ticker selector (global — persists across pages)
# ---------------------------------------------------------------------------#

selected = render_ticker_sidebar()
if not selected:
    st.warning("No price data found. Run `python cli/scrape.py --backfill 2015` first.")
    st.stop()

from dashboard.components.session_cache import format_freshness
st.caption(f"📅 Daily data through: {format_freshness(settings.raw_daily_dir / f'{selected}.parquet')}")

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

st.divider()
with st.expander("📖 How the short-term signals work", expanded=False):
    st.markdown("""
### Short-Term Signal Engine

This page aggregates **technical indicators** computed from daily price and volume data into
a single composite score ranging from **-1 (strong sell) to +1 (strong buy)**.
Each indicator independently generates a signal, and the results are averaged into the score
shown at the top.

| Indicator | What it measures | Signal logic |
|---|---|---|
| **RSI (Relative Strength Index)** | Momentum — how fast price has moved recently | RSI < 30 = oversold → BUY; RSI > 70 = overbought → SELL |
| **MACD** | Trend momentum — difference between fast and slow exponential moving averages | MACD line crossing above signal line → BUY; below → SELL |
| **Moving Averages (SMA/EMA)** | Price trend direction | 50-day above 200-day (golden cross) → BUY; below (death cross) → SELL |
| **Bollinger Bands** | Volatility + price extremes | Price at lower band → BUY; upper band → SELL |
| **Stochastic Oscillator** | Momentum relative to recent high/low range | Below 20 → BUY; above 80 → SELL |
| **ATR (Average True Range)** | Volatility level | High ATR = high risk; used to scale position size |
| **OBV (On-Balance Volume)** | Volume-confirms-price trend | Rising OBV with rising price → BUY confirmation |
| **ADX** | Trend strength (not direction) | ADX > 25 = strong trend; below = choppy/ranging market |
| **Williams %R** | Similar to Stochastic — momentum oscillator | Below -80 → BUY; above -20 → SELL |

### How the composite score is calculated
Each indicator produces a signal value of **-1, 0, or +1**. These are averaged, weighted by
indicator reliability, and normalised to the [-1, +1] range. A score above **+0.3** is
considered a weak buy, above **+0.6** a strong buy. Mirror thresholds apply for sells.

### Long-term vs short-term — what's the difference?

| | **Short-Term (this page)** | **Long-Term (Forecast page)** |
|---|---|---|
| **Input data** | Daily OHLCV price + volume | Quarterly earnings, revenue, fundamentals |
| **Horizon** | Days to weeks ahead | Quarters to years ahead |
| **Methods** | Technical indicators + momentum | Statistical time-series + ML on fundamentals |
| **Best for** | Timing trades, entry/exit precision | Valuation, long-term target prices |

Use short-term signals to decide **when** to enter or exit; use long-term forecasts to decide **whether** a stock is worth holding.
""")

