"""Page 8 — Options Flow.

Put/Call ratio, unusual volume detection, implied vs historical volatility.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import options_chart, vol_chart
from dashboard.components.tables import style_options_table, style_generic
from dashboard.components.ticker_selector import render_ticker_sidebar

st.set_page_config(page_title="Options Flow", page_icon="🎯", layout="wide")
st.title("🎯 Options Flow")
st.caption("Put/Call ratio, unusual activity flags, and implied vs historical volatility.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Ticker selector (global — persists across pages)
# ---------------------------------------------------------------------------#

selected = render_ticker_sidebar()
if not selected:
    st.warning("No data found. Run `python cli/scrape.py` first.")
    st.stop()

from dashboard.components.session_cache import format_freshness
st.caption(f"📅 Options snapshot: {format_freshness(settings.options_dir / f'{selected}.parquet')}")

# ---------------------------------------------------------------------------#
# Load data
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300)
def _load_options(ticker: str) -> pd.DataFrame:
    from src.options.options_data import load_options_metrics
    return load_options_metrics(ticker, settings.options_dir)

@st.cache_data(ttl=300)
def _load_iv(ticker: str) -> pd.DataFrame:
    from src.options.implied_vol import load_iv_metrics
    return load_iv_metrics(ticker, settings.options_dir)

@st.cache_data(ttl=300)
def _get_signal(ticker: str) -> dict:
    from src.options.options_data import get_options_signal
    return get_options_signal(ticker, settings.options_dir)

@st.cache_data(ttl=300)
def _get_iv_signal(ticker: str) -> dict:
    from src.options.implied_vol import get_iv_signal
    return get_iv_signal(ticker, settings.options_dir)

options_df = _load_options(selected)
iv_df      = _load_iv(selected)
opt_sig    = _get_signal(selected)
iv_sig     = _get_iv_signal(selected)

# ---------------------------------------------------------------------------#
# Fetch fresh options data
# ---------------------------------------------------------------------------#

if st.button("🔄 Fetch Fresh Options Data"):
    with st.spinner(f"Fetching options chains for {selected} …"):
        try:
            from src.options.options_data import compute_options_metrics, save_options_metrics
            from src.options.implied_vol import compute_iv_metrics, save_iv_metrics
            opt_data = compute_options_metrics(selected)
            save_options_metrics(opt_data, settings.options_dir)
            iv_data = compute_iv_metrics(selected, settings.raw_daily_dir)
            save_iv_metrics(iv_data, settings.options_dir)
            st.success("Options data refreshed!")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Fetch failed: {e}")

st.divider()

# ---------------------------------------------------------------------------#
# Current snapshot metrics
# ---------------------------------------------------------------------------#

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("P/C Ratio",      f"{opt_sig.get('put_call_ratio'):.3f}" if opt_sig.get('put_call_ratio') else "—",
            help="< 0.7 = bullish, > 1.0 = bearish")
col2.metric("Call Volume",    f"{opt_sig.get('total_call_volume', 0):,}")
col3.metric("Put Volume",     f"{opt_sig.get('total_put_volume', 0):,}")
col4.metric("Unusual Activity", "⚠️ YES" if opt_sig.get("unusual_activity") else "No")
col5.metric("Flow Signal",    f"{opt_sig.get('flow_signal', 0.0):+.3f}")

st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Implied Vol (ann.)",   f"{iv_sig.get('implied_vol', 0)*100:.1f}%" if iv_sig.get('implied_vol') else "—")
col2.metric("Historical Vol (ann.)",f"{iv_sig.get('historical_vol', 0)*100:.1f}%" if iv_sig.get('historical_vol') else "—")
iv_spread = iv_sig.get('iv_hv_spread')
col3.metric("IV-HV Spread",         f"{iv_spread*100:.1f}pp" if iv_spread is not None else "—",
            delta="⚠️ IV Spike" if iv_sig.get('iv_spike') else None)

if opt_sig.get("unusual_activity"):
    st.warning(f"⚠️ Unusual options activity detected for {selected} — large directional bet may be in play.")
if iv_sig.get("iv_spike"):
    st.warning(f"⚠️ IV spike detected for {selected} — options market pricing elevated uncertainty.")

st.divider()

# ---------------------------------------------------------------------------#
# Charts
# ---------------------------------------------------------------------------#

tab_options, tab_vol = st.tabs(["📊 P/C Ratio & Flow", "📈 Volatility Comparison"])

with tab_options:
    if options_df.empty:
        st.info(f"No stored options data for {selected}. Click **Fetch Fresh Options Data**.")
    else:
        st.plotly_chart(options_chart(options_df, selected), use_container_width=True)
        with st.expander("Raw options data"):
            st.dataframe(style_options_table(options_df.reset_index(names=["Date"])),
                         use_container_width=True)

with tab_vol:
    if iv_df.empty:
        st.info(f"No stored IV data for {selected}.")
    else:
        st.plotly_chart(vol_chart(iv_df, selected), use_container_width=True)
        with st.expander("Raw IV data"):
            st.dataframe(style_generic(iv_df.reset_index(names=["Date"])),
                         use_container_width=True)

st.divider()
with st.expander("📖 How options flow and volatility analysis works", expanded=False):
    st.markdown("""
### Options Flow Analysis

Options flow tracks how traders are **positioning in the options market** — which can reveal
institutional sentiment before it shows up in the stock price.

| Metric | What it means |
|---|---|
| **Put/Call Ratio (P/C)** | Ratio of put volume to call volume. Above 1.0 = more puts being bought (bearish); below 1.0 = more calls (bullish). A ratio below 0.7 is considered strongly bullish. |
| **Flow Signal** | Normalised directional score derived from the P/C ratio. Positive = call-heavy (bullish); negative = put-heavy (bearish). |
| **Unusual Activity** | Options volume significantly above the average for that strike/expiry — often indicates institutional positioning ahead of a move. |

### Implied Volatility (IV) Analysis

Implied volatility is the market's **forecast of how much a stock will move** over a given period,
derived from options prices. It is forward-looking — unlike historical volatility which measures
what already happened.

| Metric | What it means |
|---|---|
| **IV (Implied Volatility)** | Annualised expected move extracted from option prices. High IV = market expects a big move (could be up or down). |
| **HV (Historical Volatility)** | How much the stock actually moved over the past 30 days. Used as a baseline. |
| **IV vs HV** | When IV > HV significantly, options are expensive (IV crush risk after earnings). When IV < HV, options are cheap. |
| **IV Spike** | A sudden jump in IV often precedes a major event (earnings, FDA decision, news). The auto options strategy uses this as an entry signal. |
| **Vol Signal** | Normalised score: positive when IV is elevated relative to history (expected move up); negative when suppressed. |

### How the auto options strategy uses these signals
The automated options trader requires **all three** of the following to align before entering:
1. Flow signal above threshold (bullish call flow)
2. Vol signal above threshold (IV elevated)
3. Composite stock ranking score above 0.5 (fundamental + technical confirmation)

This triple-confirmation reduces false signals significantly.
""")

