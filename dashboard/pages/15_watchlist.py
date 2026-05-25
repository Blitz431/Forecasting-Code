"""Page 14 — Watchlist.

Manual watchlist with multi-timeframe signal analysis (daily + weekly),
price charts, indicator scores, peer comparison, and quick add/remove UI.
Uses src/watchlist/watchlist.py from Phase 9.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import price_line, signal_heatmap, _empty_fig
from dashboard.components.tables import style_ranking_table, style_signal_table, style_generic
from src.utils.input_sanitize import clean_ticker

st.set_page_config(page_title="Watchlist", page_icon="👁️", layout="wide")
st.title("👁️ Watchlist")
st.caption("Custom watchlist with daily + weekly signals, peer comparison, and price charts.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Watchlist manager
# ---------------------------------------------------------------------------#

from src.watchlist.watchlist import WatchlistManager
wm = WatchlistManager(settings)

# Sync session state with persisted file
if "watchlist" not in st.session_state:
    st.session_state.watchlist = wm.load()

# ---------------------------------------------------------------------------#
# Add / Remove controls
# ---------------------------------------------------------------------------#

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    raw_ticker = st.text_input("Add ticker", placeholder="e.g. NVDA")
    add_ticker = clean_ticker(raw_ticker) or ""

with col2:
    if st.button("➕ Add", type="primary"):
        if not add_ticker:
            if raw_ticker.strip():
                st.warning(f"Invalid ticker: {raw_ticker!r}")
        elif wm.add(add_ticker):
            st.session_state.watchlist = wm.load()
            st.success(f"Added {add_ticker}")
        else:
            st.info(f"{add_ticker} is already in your watchlist.")

with col3:
    remove_ticker = st.selectbox("Remove", ["—"] + st.session_state.watchlist,
                                 label_visibility="collapsed")
    if st.button("➖ Remove") and remove_ticker != "—":
        wm.remove(remove_ticker)
        st.session_state.watchlist = wm.load()
        st.success(f"Removed {remove_ticker}")
        st.rerun()

# Reload from file (in case of external modification)
watchlist = wm.load()
st.session_state.watchlist = watchlist

st.divider()

if not watchlist:
    st.info("Your watchlist is empty. Add tickers using the field above.")
    st.stop()

st.markdown(f"**{len(watchlist)} tickers:** {', '.join(watchlist)}")

# ---------------------------------------------------------------------------#
# Multi-timeframe signal table
# ---------------------------------------------------------------------------#

timeframe = st.radio("Signal timeframe", ["Daily", "Weekly"],
                     horizontal=True, label_visibility="collapsed")

@st.cache_data(ttl=600, show_spinner="Computing watchlist signals …")
def _get_signals(tickers: tuple, tf: str) -> pd.DataFrame:
    from src.watchlist.watchlist import WatchlistManager
    _wm = WatchlistManager(settings)
    return _wm.get_signals(list(tickers), timeframe=tf.lower())

sig_df = _get_signals(tuple(watchlist), timeframe)

if not sig_df.empty:
    # Colour code the Signal column
    signal_display_cols = [
        c for c in ["Ticker", "Current Price", "Signal", "Score", "RSI",
                    "SMA Cross", "20d Momentum %", "Above 200d MA", "52w High %"]
        if c in sig_df.columns
    ]
    st.subheader(f"{timeframe} Signals")
    st.dataframe(style_generic(sig_df[signal_display_cols]), use_container_width=True)

    # Quick signal bar chart
    if "Score" in sig_df.columns:
        scores = sig_df.dropna(subset=["Score"]).set_index("Ticker")["Score"]
        if not scores.empty:
            import plotly.graph_objects as go
            colors = ["#26a69a" if s >= 0 else "#ef5350" for s in scores]
            fig_bar = go.Figure(go.Bar(
                x=scores.values,
                y=scores.index,
                orientation="h",
                marker_color=colors,
                text=[f"{s:+.2f}" for s in scores],
                textposition="outside",
            ))
            fig_bar.update_layout(
                template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                title=f"{timeframe} Composite Signal Score",
                yaxis=dict(autorange="reversed", gridcolor="#2a2a3a"),
                xaxis=dict(gridcolor="#2a2a3a"),
                height=max(300, len(scores) * 30),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------#
# Per-ticker drill-down
# ---------------------------------------------------------------------------#

st.subheader("Ticker Detail")

selected = st.selectbox("Select ticker", watchlist)

@st.cache_data(ttl=300)
def _load_daily(ticker: str) -> pd.DataFrame:
    fp = settings.raw_daily_dir / f"{ticker}.parquet"
    return pd.read_parquet(fp) if fp.exists() else pd.DataFrame()


@st.cache_data(ttl=600)
def _multi_tf(ticker: str) -> dict:
    from src.watchlist.watchlist import WatchlistManager
    _wm = WatchlistManager(settings)
    return _wm.get_multi_timeframe(ticker)


@st.cache_data(ttl=600)
def _peer_compare(ticker: str) -> pd.DataFrame:
    from src.analytics.peer_comparison import PeerComparison
    pc = PeerComparison(settings)
    return pc.compare(ticker, lookback_days=63)


@st.cache_data(ttl=600)
def _run_indicators(ticker: str):
    try:
        from src.indicators.signal_aggregator import run_and_aggregate
        df = _load_daily(ticker)
        return run_and_aggregate(df, ticker) if not df.empty else None
    except Exception:
        return None


df_daily = _load_daily(selected)
mtf      = _multi_tf(selected)
peer_df  = _peer_compare(selected)
agg      = _run_indicators(selected)

tab_price, tab_signals, tab_mtf, tab_peers = st.tabs([
    "📈 Price", "📡 Indicators", "⏱️ Multi-Timeframe", "⚖️ Peers",
])

# -- Price tab --
with tab_price:
    lookback = st.slider("Lookback days", 30, 756, 252, key="wl_lookback")
    if not df_daily.empty:
        st.plotly_chart(price_line(df_daily.tail(lookback), selected),
                        use_container_width=True)

        # Show 52-week stats
        if len(df_daily) >= 252:
            closes = df_daily["Close"].dropna()
            hi52   = float(closes.tail(252).max())
            lo52   = float(closes.tail(252).min())
            cur    = float(closes.iloc[-1])
            range_pct = (cur - lo52) / max(hi52 - lo52, 0.01) * 100
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Price", f"${cur:.2f}")
            c2.metric("52w High",      f"${hi52:.2f}", delta=f"{(cur/hi52-1)*100:.1f}%")
            c3.metric("52w Low",       f"${lo52:.2f}", delta=f"{(cur/lo52-1)*100:.1f}%")
            c4.metric("52w Position",  f"{range_pct:.0f}%",
                      help="0% = at 52w low, 100% = at 52w high")
    else:
        st.info(f"No price data for {selected}. Run `python cli/scrape.py`.")

# -- Indicators tab --
with tab_signals:
    if agg is not None:
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
    else:
        st.info(f"No indicator data for {selected}.")

# -- Multi-timeframe tab --
with tab_mtf:
    daily_sig  = mtf.get("daily", {})
    weekly_sig = mtf.get("weekly", {})

    col_d, col_w = st.columns(2)

    with col_d:
        st.markdown("**Daily Signals**")
        if daily_sig:
            signal_label = daily_sig.get("Signal", "—")
            score        = daily_sig.get("Score", 0) or 0
            color = "#26a69a" if score >= 1.5 else "#ef5350" if score <= -0.5 else "#ffa726"
            st.markdown(
                f'<div style="background:{color}22; border-left:3px solid {color}; '
                f'padding:8px 12px; border-radius:4px; margin-bottom:12px;">'
                f'<b style="color:{color}">{signal_label}</b> '
                f'(score {score:+.2f})</div>',
                unsafe_allow_html=True,
            )
            for k, v in daily_sig.items():
                if k not in ("Signal", "Score"):
                    st.markdown(f"- **{k}**: {v}")
        else:
            st.info("No daily data.")

    with col_w:
        st.markdown("**Weekly Signals**")
        if weekly_sig:
            signal_label = weekly_sig.get("Signal", "—")
            score        = weekly_sig.get("Score", 0) or 0
            color = "#26a69a" if score >= 1.5 else "#ef5350" if score <= -0.5 else "#ffa726"
            st.markdown(
                f'<div style="background:{color}22; border-left:3px solid {color}; '
                f'padding:8px 12px; border-radius:4px; margin-bottom:12px;">'
                f'<b style="color:{color}">{signal_label}</b> '
                f'(score {score:+.2f})</div>',
                unsafe_allow_html=True,
            )
            for k, v in weekly_sig.items():
                if k not in ("Signal", "Score"):
                    st.markdown(f"- **{k}**: {v}")
        else:
            st.info("No weekly data.")

# -- Peers tab --
with tab_peers:
    if not peer_df.empty:
        st.markdown(f"**{selected}** vs {len(peer_df)-1} sector peers — 63-day return")
        import plotly.graph_objects as go

        colors = [
            "#26a69a" if t == selected else
            "#5c7cfa" if r >= 0 else "#ef5350"
            for t, r in zip(peer_df["Ticker"], peer_df["Return%"])
        ]
        fig_peers = go.Figure(go.Bar(
            x=peer_df["Return%"],
            y=peer_df["Ticker"],
            orientation="h",
            marker_color=colors,
            text=[f"{r:.1f}%" for r in peer_df["Return%"]],
            textposition="outside",
        ))
        fig_peers.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            title=f"{selected} Relative Strength vs Sector Peers (63-day)",
            yaxis=dict(autorange="reversed", gridcolor="#2a2a3a"),
            xaxis=dict(gridcolor="#2a2a3a", ticksuffix="%"),
            height=max(300, len(peer_df) * 28),
        )
        st.plotly_chart(fig_peers, use_container_width=True)

        st.dataframe(style_generic(peer_df.drop(columns=["Subject"], errors="ignore")),
                     use_container_width=True)
    else:
        st.info(f"No peer comparison data for {selected}.")
