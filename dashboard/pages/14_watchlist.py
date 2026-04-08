"""Page 14 — Watchlist.

Manual watchlist management — add/remove tickers, multi-timeframe signal view.
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import price_line, signal_heatmap
from dashboard.components.tables import style_ranking_table, style_signal_table

st.set_page_config(page_title="Watchlist", page_icon="👁️", layout="wide")
st.title("👁️ Watchlist")
st.caption("Custom watchlist with composite signals, price charts, and multi-timeframe analysis.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Persistent watchlist (stored as JSON in data/watchlist.json)
# ---------------------------------------------------------------------------#

_WATCHLIST_FILE = settings.data_dir / "watchlist.json"


def _load_watchlist() -> list[str]:
    if not _WATCHLIST_FILE.exists():
        return []
    try:
        return json.loads(_WATCHLIST_FILE.read_text())
    except Exception:
        return []


def _save_watchlist(tickers: list[str]) -> None:
    _WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    _WATCHLIST_FILE.write_text(json.dumps(sorted(set(tickers))))


# Load current watchlist
if "watchlist" not in st.session_state:
    st.session_state.watchlist = _load_watchlist()

# ---------------------------------------------------------------------------#
# Add / remove controls
# ---------------------------------------------------------------------------#

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    add_ticker = st.text_input("Add ticker to watchlist", placeholder="AAPL").strip().upper()

with col2:
    if st.button("➕ Add", type="primary") and add_ticker:
        if add_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(add_ticker)
            _save_watchlist(st.session_state.watchlist)
            st.success(f"Added {add_ticker}")
        else:
            st.info(f"{add_ticker} already in watchlist.")

with col3:
    remove_ticker = st.selectbox("Remove", ["—"] + st.session_state.watchlist)
    if st.button("➖ Remove") and remove_ticker != "—":
        st.session_state.watchlist = [t for t in st.session_state.watchlist if t != remove_ticker]
        _save_watchlist(st.session_state.watchlist)
        st.success(f"Removed {remove_ticker}")
        st.rerun()

st.divider()

watchlist = st.session_state.watchlist

if not watchlist:
    st.info("Your watchlist is empty. Add tickers using the field above.")
    st.stop()

st.markdown(f"**{len(watchlist)} tickers** in watchlist: {', '.join(watchlist)}")

# ---------------------------------------------------------------------------#
# Compute signals for all watchlist tickers
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=600, show_spinner="Computing signals for watchlist …")
def _watchlist_signals(tickers: tuple) -> pd.DataFrame:
    from src.ranking.ranker import rank_tickers, to_dataframe
    entries = rank_tickers(list(tickers), include_ml=False, include_indicators=True)
    return to_dataframe(entries)

sig_df = _watchlist_signals(tuple(watchlist))

if not sig_df.empty:
    score_cols = ["Rank", "Ticker", "Score", "Signals", "Forecast", "Indicators",
                  "ML", "News", "Short Int.", "Congress", "Insider", "Options", "IV", "Earnings"]
    avail = [c for c in score_cols if c in sig_df.columns]
    st.subheader("Watchlist Signals")
    st.dataframe(style_ranking_table(sig_df[avail]), use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------#
# Per-ticker drill-down
# ---------------------------------------------------------------------------#

st.subheader("Ticker Detail")

selected = st.selectbox("Select ticker", watchlist)

@st.cache_data(ttl=300)
def _load_daily(ticker: str) -> pd.DataFrame:
    fp = settings.raw_daily_dir / f"{ticker}.parquet"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_parquet(fp)

@st.cache_data(ttl=300)
def _run_indicators(ticker: str):
    from src.indicators.signal_aggregator import run_and_aggregate
    df = _load_daily(ticker)
    if df.empty:
        return None
    return run_and_aggregate(df, ticker)

df_daily = _load_daily(selected)
agg      = _run_indicators(selected)

tab_price, tab_signals = st.tabs(["📈 Price", "📡 Indicators"])

with tab_price:
    lookback = st.slider("Lookback days", 30, 756, 252, key="wl_lookback")
    if not df_daily.empty:
        st.plotly_chart(price_line(df_daily.tail(lookback), selected),
                        use_container_width=True)
    else:
        st.info(f"No price data for {selected}.")

with tab_signals:
    if agg is None:
        st.info(f"No indicator data for {selected}.")
    else:
        # Signal summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Composite Signal", agg.signal.label())
        col2.metric("Score", f"{agg.score:+.3f}")
        col3.metric("Indicators OK", f"{agg.succeeded}/{agg.succeeded+agg.failed}")

        st.plotly_chart(signal_heatmap(agg.results, selected), use_container_width=True)

        rows = [{"Indicator": r.indicator_name,
                 "Signal": r.signal.label() if not r.error else "ERROR",
                 "Interpretation": r.interpretation if not r.error else r.error[:80]}
                for r in agg.results]
        st.dataframe(style_signal_table(pd.DataFrame(rows)), use_container_width=True)
