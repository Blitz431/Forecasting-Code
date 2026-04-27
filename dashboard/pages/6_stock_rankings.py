"""Page 5 — Stock Rankings.

Top-20 composite picks with per-signal score breakdown and drill-down charts.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import ranking_bar, score_breakdown
from dashboard.components.tables import style_ranking_table
from src.utils.input_sanitize import clean_ticker_list

st.set_page_config(page_title="Stock Rankings", page_icon="🏆", layout="wide")
st.title("🏆 Stock Rankings")
st.caption("Composite top-20 picks aggregating forecasts, indicators, ML, news, political, options, and earnings signals.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Sidebar controls
# ---------------------------------------------------------------------------#

with st.sidebar:
    st.header("Ranking Options")
    top_n      = st.slider("Top N picks", 5, 50, value=20)
    include_ml = st.checkbox("Include ML signal (slow)", value=False,
                              help="Calls predict_latest() per ticker — much slower.")
    no_indicators = st.checkbox("Skip indicators (fastest)", value=False)
    custom_tickers = st.text_input("Custom ticker list (comma-sep)",
                                    placeholder="AAPL,MSFT,NVDA … (blank = S&P 500)")

# ---------------------------------------------------------------------------#
# Check data availability
# ---------------------------------------------------------------------------#

daily_dir = settings.raw_daily_dir
if not daily_dir.exists() or not any(daily_dir.glob("*.parquet")):
    st.warning(
        "No scraped data found.\n\n"
        "Run `python cli/scrape.py --backfill 2015` then re-open this page."
    )
    st.stop()

# ---------------------------------------------------------------------------#
# Run ranking
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=900, show_spinner="Computing composite rankings …")
def _compute_rankings(
    top_n: int,
    include_ml: bool,
    include_indicators: bool,
    custom: str,
):
    from src.ranking.ranker import top_picks, to_dataframe

    tickers = None
    if custom.strip():
        tickers, rejected = clean_ticker_list(custom)
        if rejected:
            st.warning(f"Ignored {len(rejected)} invalid ticker(s): {', '.join(rejected[:10])}")
        if not tickers:
            tickers = None

    picks = top_picks(
        n=top_n,
        settings=None,
        include_ml=include_ml,
        include_indicators=include_indicators,
        tickers=tickers,
    )
    return to_dataframe(picks), picks


if st.button("▶  Run Ranking", type="primary"):
    st.cache_data.clear()   # force fresh run

with st.spinner("Loading rankings …"):
    try:
        df, picks = _compute_rankings(
            top_n,
            include_ml,
            not no_indicators,
            custom_tickers or "",
        )
    except Exception as e:
        st.error(f"Ranking failed: {e}")
        st.stop()

if df.empty:
    st.warning(
        "Ranking returned no results. "
        "Ensure daily price data exists (`cli/scrape.py`) and try again."
    )
    st.stop()

# ---------------------------------------------------------------------------#
# Summary metrics
# ---------------------------------------------------------------------------#

top_ticker = df.iloc[0]["Ticker"] if not df.empty else "—"
top_score  = df.iloc[0]["Score"]  if not df.empty else 0.0
avg_score  = df["Score"].mean()
n_positive = (df["Score"] > 0).sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("#1 Pick",          top_ticker)
col2.metric("Top Score",        f"{top_score:.3f}")
col3.metric("Avg Score",        f"{avg_score:.3f}")
col4.metric("Bullish (>0)",     f"{n_positive} / {len(df)}")

st.divider()

# ---------------------------------------------------------------------------#
# Bar chart + table
# ---------------------------------------------------------------------------#

tab_chart, tab_table = st.tabs(["📊 Chart", "📋 Table"])

with tab_chart:
    st.plotly_chart(ranking_bar(df, top_n=top_n), use_container_width=True)

with tab_table:
    # Identify signal columns (Score already in the base list — exclude from signal_cols to avoid duplicates)
    signal_cols = ["Forecast", "Indicators", "ML", "News",
                   "Short Int.", "Congress", "Insider", "Options", "IV", "Earnings"]
    avail_cols = [c for c in ["Rank", "Ticker", "Score", "Signals"] + signal_cols if c in df.columns]
    st.dataframe(style_ranking_table(df[avail_cols]), use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------#
# Drill-down: select one ticker and see breakdown
# ---------------------------------------------------------------------------#

st.subheader("Signal Breakdown — Drill-Down")

ticker_options = df["Ticker"].tolist()
drill = st.selectbox("Select ticker to inspect", ticker_options, index=0)

# Find the corresponding RankEntry
drill_entry = next((p for p in picks if p.ticker == drill), None)

if drill_entry:
    # Score breakdown bar
    st.plotly_chart(
        score_breakdown(drill_entry.signals, ticker=drill),
        use_container_width=True,
    )

    col_l, col_r = st.columns(2)
    with col_l:
        st.metric("Composite Score",     f"{drill_entry.composite_score:+.4f}")
        st.metric("Signals Contributing", drill_entry.signals_available)

    with col_r:
        # Signal details table
        sig_rows = [{"Signal": k, "Value": f"{v:+.3f}"} for k, v in drill_entry.signals.items()]
        if sig_rows:
            st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)
else:
    st.info("Select a ticker above to see the signal breakdown.")
