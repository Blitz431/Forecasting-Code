"""Page 10 — Portfolio Analytics.

Portfolio analytics, correlation matrix, performance attribution.
Full implementation in Phase 9 (backtesting & analytics).
This page shows a placeholder with any live Alpaca data available now.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import correlation_matrix, _empty_fig
from dashboard.components.tables import style_generic

st.set_page_config(page_title="Portfolio", page_icon="💼", layout="wide")
st.title("💼 Portfolio Analytics")
st.caption("Portfolio tracking, correlation matrix, and performance attribution. Full analytics arrive in Phase 9.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Phase note
# ---------------------------------------------------------------------------#

st.info(
    "**Phase 9** will add: Sharpe ratio, Sortino, max drawdown, beta, "
    "sector concentration, peer relative strength, correlation matrix, "
    "and per-stock P&L attribution.\n\n"
    "This page currently shows live Alpaca portfolio data (if connected)."
)

st.divider()

# ---------------------------------------------------------------------------#
# Alpaca portfolio summary
# ---------------------------------------------------------------------------#

st.subheader("Live Portfolio (Alpaca)")

@st.cache_data(ttl=60, show_spinner="Fetching Alpaca portfolio …")
def _fetch_alpaca_portfolio():
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(
            settings.alpaca.api_key,
            settings.alpaca.secret_key,
            settings.alpaca.base_url,
        )
        account    = api.get_account()
        positions  = api.list_positions()
        return account, positions
    except Exception as e:
        return None, str(e)

if settings.alpaca.api_key:
    account, positions = _fetch_alpaca_portfolio()

    if account is None:
        st.error(f"Alpaca connection failed: {positions}")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portfolio Value",  f"${float(account.portfolio_value):,.2f}")
        col2.metric("Buying Power",     f"${float(account.buying_power):,.2f}")
        col3.metric("Cash",             f"${float(account.cash):,.2f}")
        unrealized = sum(float(p.unrealized_pl) for p in positions) if positions else 0
        col4.metric("Unrealized P&L",   f"${unrealized:,.2f}",
                    delta=f"{unrealized:+,.2f}")

        if positions:
            st.subheader("Open Positions")
            pos_rows = []
            for p in positions:
                pos_rows.append({
                    "Ticker":         p.symbol,
                    "Qty":            float(p.qty),
                    "Avg Entry":      f"${float(p.avg_entry_price):.2f}",
                    "Current Price":  f"${float(p.current_price):.2f}",
                    "Market Value":   f"${float(p.market_value):.2f}",
                    "Unrealized P&L": f"${float(p.unrealized_pl):.2f}",
                    "Unrealized %":   f"{float(p.unrealized_plpc)*100:.2f}%",
                    "Side":           p.side,
                })
            pos_df = pd.DataFrame(pos_rows)
            st.dataframe(style_generic(pos_df), use_container_width=True)

            # Simple correlation heatmap from daily data
            st.subheader("Holding Correlation (30-day)")
            tickers = [p.symbol for p in positions]
            frames = []
            for t in tickers:
                fp = settings.raw_daily_dir / f"{t}.parquet"
                if fp.exists():
                    df = pd.read_parquet(fp)
                    if "Close" in df.columns:
                        frames.append(df["Close"].tail(30).rename(t))
            if frames:
                price_df = pd.concat(frames, axis=1).dropna()
                corr = price_df.pct_change().dropna().corr()
                st.plotly_chart(correlation_matrix(corr), use_container_width=True)
        else:
            st.info("No open positions.")
else:
    st.warning(
        "Alpaca API keys not configured.\n\n"
        "Add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` to your `.env` file to "
        "see live portfolio data."
    )

st.divider()
st.subheader("Coming in Phase 9")
cols = st.columns(3)
with cols[0]:
    st.markdown("- Sharpe ratio\n- Sortino ratio\n- Max drawdown\n- Beta vs S&P 500")
with cols[1]:
    st.markdown("- Sector concentration\n- Peer relative strength\n- Attribution per stock")
with cols[2]:
    st.markdown("- Full correlation matrix\n- Equity curve chart\n- Rolling metrics")
