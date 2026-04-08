"""Page 17 — Trading.

Live / paper trading controls, circuit breaker, and multi-strategy management.
Full implementation in Phase 10.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.tables import style_generic

st.set_page_config(page_title="Trading", page_icon="🔧", layout="wide")
st.title("🔧 Trading Controls")
st.caption("Paper and live trading via Alpaca — multi-strategy execution, circuit breaker, order management.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Phase note
# ---------------------------------------------------------------------------#

st.info(
    "**Phase 10** implements:\n\n"
    "- `src/trading/alpaca_client.py` — paper-first Alpaca wrapper\n"
    "- `src/trading/strategy.py` — entry/exit rules + 5% trailing stop\n"
    "- `src/trading/multi_strategy.py` — value/momentum/mean-reversion simultaneously\n"
    "- `src/trading/circuit_breaker.py` — kill switch at -10% portfolio / -15% single stock\n"
    "- `src/trading/risk.py` — position sizing, max 5% per stock\n\n"
    "CLI: `python cli/trade.py --mode paper`"
)

st.divider()

# ---------------------------------------------------------------------------#
# Trading mode indicator
# ---------------------------------------------------------------------------#

mode_color = "🟡 Paper" if settings.trading_mode == "paper" else "🔴 LIVE"
st.subheader(f"Trading Mode: {mode_color}")

if settings.trading_mode == "live":
    st.error(
        "⚠️ **LIVE TRADING MODE ACTIVE** — real money at risk. "
        "Circuit breaker armed. Set `TRADING_MODE=paper` in `.env` to switch to paper trading."
    )
else:
    st.success("✅ Paper trading mode — no real money at risk.")

st.divider()

# ---------------------------------------------------------------------------#
# Alpaca account status
# ---------------------------------------------------------------------------#

st.subheader("Alpaca Account Status")

@st.cache_data(ttl=30)
def _fetch_account():
    if not settings.alpaca.api_key:
        return None, "API key not configured"
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(
            settings.alpaca.api_key,
            settings.alpaca.secret_key,
            settings.alpaca.base_url,
        )
        return api.get_account(), None
    except Exception as e:
        return None, str(e)

account, err = _fetch_account()

if err:
    st.warning(f"Could not connect to Alpaca: {err}")
    if not settings.alpaca.api_key:
        st.markdown(
            "Add API keys to `.env`:\n```\nALPACA_API_KEY=your_key\nALPACA_SECRET_KEY=your_secret\n```"
        )
else:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Status",         account.status.upper())
    col2.metric("Portfolio Value",f"${float(account.portfolio_value):,.2f}")
    col3.metric("Cash",           f"${float(account.cash):,.2f}")
    col4.metric("Buying Power",   f"${float(account.buying_power):,.2f}")
    col5.metric("Day Trades Left",account.daytrade_count if hasattr(account, "daytrade_count") else "—")

    if account.trading_blocked:
        st.error("⚠️ Trading is blocked on this account.")
    if account.account_blocked:
        st.error("⚠️ Account is blocked.")

st.divider()

# ---------------------------------------------------------------------------#
# Circuit breaker status
# ---------------------------------------------------------------------------#

st.subheader("🚨 Circuit Breaker")

st.markdown(f"""
| Trigger | Threshold | Action |
|---------|-----------|--------|
| Portfolio daily drop | {settings.circuit_breaker_daily_pct:.0f}% | Halt all new trades |
| Single stock loss | {settings.circuit_breaker_single_stock_pct:.0f}% | Force-sell position |
| Manual emergency | Button below | Immediately flatten all positions |
""")

col_arm, col_halt = st.columns(2)

with col_arm:
    st.metric("Circuit Breaker", "Armed ✅ (Phase 10)")

with col_halt:
    if st.button("🛑 EMERGENCY STOP — Flatten All Positions",
                 type="primary",
                 help="Phase 10: Will immediately submit sell orders for all open positions."):
        st.error(
            "Emergency stop will be implemented in Phase 10. "
            "For now, log in to Alpaca directly to close positions."
        )

st.divider()

# ---------------------------------------------------------------------------#
# Planned strategy overview
# ---------------------------------------------------------------------------#

st.subheader("Multi-Strategy Configuration (Phase 10)")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    **Value Strategy**
    - Low P/S ratio
    - High dividend yield
    - Buy-and-hold horizon
    - Capital allocation: 33%
    """)
with col2:
    st.markdown("""
    **Momentum Strategy**
    - Top 20 composite picks
    - Entry on breakout
    - 5% trailing stop
    - Capital allocation: 33%
    """)
with col3:
    st.markdown("""
    **Mean-Reversion Strategy**
    - Bollinger Band touches
    - RSI oversold bounces
    - Short holding period
    - Capital allocation: 33%
    """)

st.divider()

st.subheader("Risk Parameters")

risk_df = pd.DataFrame([
    {"Parameter": "Max position size", "Value": f"{settings.max_position_pct:.0f}% of portfolio"},
    {"Parameter": "Trailing stop",     "Value": f"{settings.trailing_stop_pct:.0f}% after {settings.trailing_stop_days} profitable days"},
    {"Parameter": "Circuit breaker",   "Value": f"Halt at {settings.circuit_breaker_daily_pct:.0f}% portfolio drop"},
    {"Parameter": "Force-sell",        "Value": f"At {settings.circuit_breaker_single_stock_pct:.0f}% single-stock loss"},
    {"Parameter": "Trading mode",      "Value": settings.trading_mode.upper()},
])
st.dataframe(style_generic(risk_df), use_container_width=True, hide_index=True)
