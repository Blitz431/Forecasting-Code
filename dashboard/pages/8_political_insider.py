"""Page 7 — Political & Insider Trading.

Congressional STOCK Act trades (Quiver Quant) and SEC Form 4 insider filings.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import _empty_fig
from dashboard.components.tables import style_congress_table, style_insider_table, style_generic

st.set_page_config(page_title="Political & Insider", page_icon="🏛️", layout="wide")
st.title("🏛️ Political & Insider Trading")
st.caption("Congressional STOCK Act filings and SEC Form 4 insider transactions.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Ticker selector + fetch controls
# ---------------------------------------------------------------------------#

daily_tickers = sorted([fp.stem for fp in settings.raw_daily_dir.glob("*.parquet")]) \
    if settings.raw_daily_dir.exists() else []
congress_tickers = sorted([fp.stem for fp in settings.political_congress_dir.glob("*.parquet")]) \
    if settings.political_congress_dir.exists() else []
insider_tickers  = sorted([fp.stem for fp in settings.political_insider_dir.glob("*.parquet")]) \
    if settings.political_insider_dir.exists() else []
all_tickers = list(dict.fromkeys(congress_tickers + insider_tickers + daily_tickers))

if not all_tickers:
    st.warning("No data found. Run `python cli/scrape.py` first.")
    st.stop()

col1, col2 = st.columns([2, 1])
with col1:
    selected = st.selectbox("Ticker", all_tickers,
                             index=all_tickers.index("AAPL") if "AAPL" in all_tickers else 0)
with col2:
    lookback = st.number_input("Lookback days", min_value=7, max_value=365, value=90)

# ---------------------------------------------------------------------------#
# Load data helpers
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300)
def _load_congress(ticker: str) -> pd.DataFrame:
    from src.political.congress_tracker import load_congress_trades
    return load_congress_trades(ticker, settings.political_congress_dir)

@st.cache_data(ttl=300)
def _load_insider(ticker: str) -> pd.DataFrame:
    from src.political.insider_tracker import load_insider_trades
    return load_insider_trades(ticker, settings.political_insider_dir)

@st.cache_data(ttl=300)
def _congress_signal(ticker: str) -> dict:
    from src.political.congress_tracker import get_congress_signal
    return get_congress_signal(ticker, settings.political_congress_dir, window_days=lookback)

@st.cache_data(ttl=300)
def _insider_signal(ticker: str) -> dict:
    from src.political.insider_tracker import get_insider_signal
    return get_insider_signal(ticker, settings.political_insider_dir, window_days=lookback)

congress_df = _load_congress(selected)
insider_df  = _load_insider(selected)
c_sig       = _congress_signal(selected)
i_sig       = _insider_signal(selected)

# ---------------------------------------------------------------------------#
# Signal summary metrics
# ---------------------------------------------------------------------------#

col1, col2, col3, col4 = st.columns(4)
col1.metric("Congress Signal",  f"{c_sig.get('congress_signal', 0.0):+.3f}",
            help="Positive = net buys, Negative = net sells")
col2.metric("Congress Buys",    f"{c_sig.get('congress_net_buys', 0)}",
            delta=f"${c_sig.get('congress_buy_value', 0):,.0f}")
col3.metric("Insider Signal",   f"{i_sig.get('insider_signal', 0.0):+.3f}")
col4.metric("Insider Buys",     f"{i_sig.get('insider_net_buys', 0)}",
            delta=f"${i_sig.get('insider_buy_value', 0):,.0f}")

st.divider()

# ---------------------------------------------------------------------------#
# Congressional trades
# ---------------------------------------------------------------------------#

tab1, tab2 = st.tabs(["🏛️ Congressional Trades", "💼 Insider Filings"])

with tab1:
    fetch_col, _ = st.columns([1, 3])
    with fetch_col:
        if st.button("🔄 Fetch Congress Trades"):
            with st.spinner(f"Fetching congressional trades for {selected} …"):
                try:
                    from src.political.congress_tracker import fetch_congress_trades, save_congress_trades
                    trades = fetch_congress_trades(selected, lookback_days=lookback)
                    if trades:
                        save_congress_trades(trades, settings.political_congress_dir)
                        st.success(f"Fetched {len(trades)} trades")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.info("No trades found for this ticker.")
                except Exception as e:
                    st.error(f"Fetch failed: {e}")

    if congress_df.empty:
        st.info(
            f"No congressional trades for **{selected}** in the database.\n\n"
            "Click **Fetch Congress Trades** above, or run `python cli/scrape.py`."
        )
    else:
        # Filter to lookback window
        if congress_df.index.tz is not None:
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback)
        else:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback)
        recent = congress_df[congress_df.index >= cutoff] if lookback < 10000 else congress_df

        st.markdown(f"**{len(recent)} trades in last {lookback} days** (total: {len(congress_df)})")
        disp = recent.reset_index(names=["Trade Date"])
        disp_cols = [c for c in ["Trade Date", "representative", "transaction",
                                   "amount_mid", "party", "chamber", "report_date"]
                     if c in disp.columns]
        st.dataframe(style_congress_table(disp[disp_cols] if disp_cols else disp),
                     use_container_width=True, height=350)

        # Simple buy/sell bar
        import plotly.graph_objects as go
        if "transaction" in recent.columns and "amount_mid" in recent.columns:
            recent_copy = recent.copy()
            recent_copy["is_buy"] = recent_copy["transaction"].str.lower().str.startswith("p") | \
                                     recent_copy["transaction"].str.lower().str.startswith("b")
            buy_val  = recent_copy[recent_copy["is_buy"]]["amount_mid"].fillna(0).sum()
            sell_val = recent_copy[~recent_copy["is_buy"]]["amount_mid"].fillna(0).sum()

            fig = go.Figure(go.Bar(
                x=["Buys", "Sells"],
                y=[buy_val, sell_val],
                marker_color=["#26a69a", "#ef5350"],
                text=[f"${buy_val:,.0f}", f"${sell_val:,.0f}"],
                textposition="outside",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                title=f"{selected} — Congressional Trade Value ({lookback}d)",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    fetch_col2, _ = st.columns([1, 3])
    with fetch_col2:
        if st.button("🔄 Fetch Insider Filings"):
            with st.spinner(f"Fetching SEC Form 4 filings for {selected} …"):
                try:
                    from src.political.insider_tracker import fetch_insider_trades, save_insider_trades
                    trades = fetch_insider_trades(selected, lookback_days=lookback)
                    if trades:
                        save_insider_trades(trades, selected, settings.political_insider_dir)
                        st.success(f"Fetched {len(trades)} filings")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.info("No insider filings found.")
                except Exception as e:
                    st.error(f"Fetch failed: {e}")

    if insider_df.empty:
        st.info(
            f"No insider filings for **{selected}**.\n\n"
            "Click **Fetch Insider Filings** above."
        )
    else:
        if insider_df.index.tz is not None:
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback)
        else:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback)
        recent_i = insider_df[insider_df.index >= cutoff] if lookback < 10000 else insider_df

        st.markdown(f"**{len(recent_i)} filings in last {lookback} days** (total: {len(insider_df)})")
        disp_i = recent_i.reset_index(names=["Filing Date"])
        disp_cols_i = [c for c in ["Filing Date", "name", "title", "transaction_type",
                                    "shares", "price_per_share", "total_value",
                                    "shares_owned_after"]
                       if c in disp_i.columns]
        st.dataframe(style_insider_table(disp_i[disp_cols_i] if disp_cols_i else disp_i),
                     use_container_width=True, height=350)
