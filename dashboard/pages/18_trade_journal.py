"""Page 15 — Trade Journal.

Audit log of every trade: entry/exit reasons, signal triggers, P&L, holding period.
Tax lot tracking with FIFO/LIFO cost basis and tax-loss harvesting suggestions.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import get_settings
from dashboard.components.tables import style_generic

st.set_page_config(page_title="Trade Journal", page_icon="📓", layout="wide")

from dashboard.components.market_clock import render_market_clock
render_market_clock()

st.title("📓 Trade Journal")
st.caption("Append-only audit log of every trade with entry/exit reasons, signal triggers, P&L, and tax lots.")
st.divider()

settings = get_settings()


# ---------------------------------------------------------------------------#
# Load journal data
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=60, show_spinner="Loading trade journal …")
def _load_journal() -> pd.DataFrame:
    try:
        from src.trading.trade_journal import TradeJournal
        journal = TradeJournal(settings)
        return journal.load()
    except Exception as e:
        return pd.DataFrame(), str(e)


@st.cache_data(ttl=60, show_spinner="Loading tax lots …")
def _load_tax_lots():
    try:
        from src.trading.tax_lots import TaxLotTracker
        tracker = TaxLotTracker(settings)
        open_lots = tracker.open_lots_summary()
        closed_lots = tracker.realized_pnl_summary()
        harvest = tracker.harvest_candidates(threshold_pct=5.0)
        return open_lots, closed_lots, harvest, None
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), str(e)


# ---------------------------------------------------------------------------#
# Top-level summary metrics
# ---------------------------------------------------------------------------#

journal_data = _load_journal()
if isinstance(journal_data, tuple):
    df_journal, journal_err = journal_data
else:
    df_journal, journal_err = journal_data, None

has_journal = isinstance(df_journal, pd.DataFrame) and not df_journal.empty

if has_journal:
    exits = df_journal[df_journal["side"] == "SELL"]
    entries = df_journal[df_journal["side"] == "BUY"]
    total_pnl = float(exits["realized_pnl"].dropna().sum()) if not exits.empty else 0.0
    win_rate  = (
        (exits["realized_pnl"] > 0).sum() / len(exits) * 100
        if not exits.empty else 0.0
    )
    avg_hold  = float(exits["holding_days"].dropna().mean()) if not exits.empty and "holding_days" in exits.columns else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades (closed)", len(exits))
    col2.metric("Win Rate", f"{win_rate:.1f}%")
    col3.metric("Total Realized P&L", f"${total_pnl:+,.2f}",
                delta_color="normal" if total_pnl >= 0 else "inverse")
    col4.metric("Avg Holding Period", f"{avg_hold:.1f} days")
    st.divider()

# ---------------------------------------------------------------------------#
# Section 1: Trade log with filters
# ---------------------------------------------------------------------------#

st.subheader("Trade Log")

if not has_journal:
    if journal_err:
        st.error(f"Error loading journal: {journal_err}")
    else:
        st.info(
            "No trades logged yet.  Run `python cli/trade.py --mode paper` to start trading.\n\n"
            f"Journal will be written to: `{settings.trade_journal_dir / 'journal.parquet'}`"
        )
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        side_filter = st.multiselect("Side", ["BUY", "SELL"], default=["BUY", "SELL"])
    with col2:
        strategies = df_journal["strategy"].dropna().unique().tolist() if "strategy" in df_journal.columns else []
        strat_filter = st.multiselect("Strategy", strategies, default=strategies)
    with col3:
        tickers_in_journal = sorted(df_journal["ticker"].unique().tolist())
        ticker_filter = st.multiselect("Ticker", tickers_in_journal, default=[])

    filtered = df_journal.copy()
    if side_filter:
        filtered = filtered[filtered["side"].isin(side_filter)]
    if strat_filter and "strategy" in filtered.columns:
        filtered = filtered[filtered["strategy"].isin(strat_filter)]
    if ticker_filter:
        filtered = filtered[filtered["ticker"].isin(ticker_filter)]

    # Format for display
    display_cols = ["timestamp", "ticker", "side", "strategy", "price", "shares",
                    "dollar_value", "realized_pnl", "realized_pnl_pct", "exit_reason",
                    "holding_days"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    disp = filtered[display_cols].copy()
    if "realized_pnl" in disp.columns:
        disp["realized_pnl"] = disp["realized_pnl"].map(
            lambda x: f"${x:+,.2f}" if pd.notna(x) else "—"
        )
    if "realized_pnl_pct" in disp.columns:
        disp["realized_pnl_pct"] = disp["realized_pnl_pct"].map(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"
        )

    st.metric("Trades shown", len(disp))
    st.dataframe(style_generic(disp), use_container_width=True, height=400)

st.divider()

# ---------------------------------------------------------------------------#
# Section 2: P&L charts (only if exits exist)
# ---------------------------------------------------------------------------#

if has_journal and not exits.empty:
    st.subheader("Realized P&L")

    tab1, tab2, tab3 = st.tabs(["Cumulative P&L", "P&L by Ticker", "P&L by Strategy"])

    with tab1:
        exits_sorted = exits.sort_values("timestamp")
        exits_sorted["cumulative_pnl"] = exits_sorted["realized_pnl"].cumsum()
        fig = px.line(exits_sorted, x="timestamp", y="cumulative_pnl",
                      title="Cumulative Realized P&L",
                      labels={"cumulative_pnl": "P&L ($)", "timestamp": "Date"})
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        pnl_by_ticker = (
            exits.groupby("ticker")["realized_pnl"]
            .sum()
            .reset_index()
            .sort_values("realized_pnl", ascending=True)
        )
        fig2 = px.bar(pnl_by_ticker, x="realized_pnl", y="ticker",
                      orientation="h",
                      color="realized_pnl",
                      color_continuous_scale=["#ef5350", "#26a69a"],
                      title="Total P&L by Ticker")
        fig2.update_layout(template="plotly_dark", height=max(300, len(pnl_by_ticker) * 25))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        if "strategy" in exits.columns:
            pnl_by_strat = (
                exits.groupby("strategy")["realized_pnl"]
                .agg(["sum", "count", "mean"])
                .reset_index()
                .rename(columns={"sum": "Total P&L", "count": "Trades", "mean": "Avg P&L"})
            )
            st.dataframe(style_generic(pnl_by_strat), use_container_width=True, hide_index=True)

    st.divider()

# ---------------------------------------------------------------------------#
# Section 3: Signal breakdown on last trade
# ---------------------------------------------------------------------------#

if has_journal and not df_journal.empty and "signals" in df_journal.columns:
    st.subheader("Signals — Last 5 Trades")
    recent = df_journal.head(5)
    for _, row in recent.iterrows():
        with st.expander(
            f"{row.get('side','?')} {row.get('ticker','?')} "
            f"@ ${float(row.get('price',0)):.2f}  |  {str(row.get('timestamp',''))[:19]}"
        ):
            sig_json = row.get("signals", "{}")
            try:
                sigs = json.loads(sig_json) if isinstance(sig_json, str) else sig_json
                if sigs:
                    sig_df = pd.DataFrame(
                        [{"Signal": k, "Value": round(v, 4)} for k, v in sigs.items()]
                    )
                    st.dataframe(sig_df, use_container_width=True, hide_index=True)
                else:
                    st.write("No signal data recorded.")
            except Exception:
                st.write(sig_json)
    st.divider()

# ---------------------------------------------------------------------------#
# Section 4: Tax lots
# ---------------------------------------------------------------------------#

st.subheader("Tax Lots")

open_lots, closed_lots, harvest_df, tax_err = _load_tax_lots()

if tax_err:
    st.warning(f"Could not load tax lots: {tax_err}")
else:
    tab_open, tab_closed, tab_harvest = st.tabs(["Open Lots", "Closed (Realized)", "Harvest Candidates"])

    with tab_open:
        if open_lots.empty:
            st.info("No open tax lots.")
        else:
            st.metric("Open lots", len(open_lots))
            st.dataframe(style_generic(open_lots), use_container_width=True, hide_index=True)

    with tab_closed:
        if closed_lots.empty:
            st.info("No closed lots yet.")
        else:
            has_gain_type = "gain_type" in closed_lots.columns
            lt_df = closed_lots[closed_lots["gain_type"] == "long_term"] if has_gain_type else pd.DataFrame()
            st_df = closed_lots[closed_lots["gain_type"] == "short_term"] if has_gain_type else pd.DataFrame()
            c1, c2 = st.columns(2)
            c1.metric("Long-term gains",
                      f"${float(lt_df['realized_pnl'].sum()):+,.2f}" if not lt_df.empty and 'realized_pnl' in lt_df.columns else "$0")
            c2.metric("Short-term gains",
                      f"${float(st_df['realized_pnl'].sum()):+,.2f}" if not st_df.empty and 'realized_pnl' in st_df.columns else "$0")
            st.dataframe(style_generic(closed_lots), use_container_width=True, hide_index=True)

    with tab_harvest:
        if harvest_df.empty:
            st.info("No harvest candidates (no positions with >= 5% unrealized loss).")
        else:
            st.warning(
                f"**{len(harvest_df)} positions** have unrealized losses ≥ 5% and could be "
                "harvested to offset realized gains."
            )
            st.dataframe(style_generic(harvest_df), use_container_width=True, hide_index=True)
