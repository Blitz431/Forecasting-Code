"""Page 17 — Trading Controls.

Live / paper trading dashboard using Phase 10 modules:
- Alpaca account status and positions
- Circuit breaker status with live arming and emergency stop
- Multi-strategy capital allocation (Sharpe-weighted)
- Risk parameters
- One-click dry-run to preview what the trading loop would do
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import get_settings
from dashboard.components.tables import style_generic

st.set_page_config(page_title="Trading", page_icon="🔧", layout="wide")
st.title("🔧 Trading Controls")
st.caption("Paper and live trading via Alpaca — account status, circuit breaker, multi-strategy management.")
st.divider()

settings = get_settings()


# ---------------------------------------------------------------------------#
# Trading mode banner
# ---------------------------------------------------------------------------#

import os
live_env = os.getenv("ALPACA_LIVE_TRADING", "").strip().lower() == "true"
mode_label = "🔴 LIVE TRADING" if live_env else "🟡 PAPER TRADING"

if live_env:
    st.error(
        f"**{mode_label} MODE ACTIVE** — real money at risk.  "
        "Unset `ALPACA_LIVE_TRADING` or set it to `false` in `.env` to return to paper."
    )
else:
    st.success(f"**{mode_label} MODE** — no real money at risk.")

st.divider()

# ---------------------------------------------------------------------------#
# Alpaca account status
# ---------------------------------------------------------------------------#

st.subheader("Alpaca Account")

@st.cache_data(ttl=30, show_spinner="Fetching account …")
def _fetch_account():
    if not settings.alpaca.api_key:
        return None, "API key not configured"
    try:
        from src.trading.alpaca_client import AlpacaClient
        client = AlpacaClient(settings)
        account = client.get_account()
        return account, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=30, show_spinner="Fetching positions …")
def _fetch_positions():
    if not settings.alpaca.api_key:
        return []
    try:
        from src.trading.alpaca_client import AlpacaClient
        client = AlpacaClient(settings)
        return client.list_positions()
    except Exception:
        return []


account, acct_err = _fetch_account()
positions = _fetch_positions()

if acct_err:
    st.warning(f"Could not connect to Alpaca: {acct_err}")
    if not settings.alpaca.api_key:
        st.code("ALPACA_API_KEY=your_key\nALPACA_SECRET_KEY=your_secret", language="bash")
elif account is None:
    st.warning("Could not connect to Alpaca: get_account returned no data")
else:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Status",          account.status.upper())
    col2.metric("Portfolio Value",  f"${account.portfolio_value:,.2f}")
    col3.metric("Cash",             f"${account.cash:,.2f}")
    col4.metric("Buying Power",     f"${account.buying_power:,.2f}")
    col5.metric("Daily P&L",        f"${account.daily_pnl:+,.2f}",
                delta=f"{account.daily_pnl_pct:+.2f}%",
                delta_color="normal" if account.daily_pnl >= 0 else "inverse")

    if account.trading_blocked:
        st.error("⚠️ Trading is blocked on this account.")
    if account.account_blocked:
        st.error("⚠️ Account is blocked.")
    if account.pattern_day_trader:
        st.warning("⚠️ Pattern Day Trader (PDT) flag is set.")

st.divider()

# ---------------------------------------------------------------------------#
# Open positions
# ---------------------------------------------------------------------------#

st.subheader(f"Open Positions ({len(positions)})")

if positions:
    pos_rows = []
    for p in positions:
        pos_rows.append({
            "Ticker":       p.ticker,
            "Qty":          p.qty,
            "Avg Entry":    f"${p.avg_entry_price:.2f}",
            "Current":      f"${p.current_price:.2f}",
            "Market Value": f"${p.market_value:,.2f}",
            "Unrealized P&L": f"${p.unrealized_pl:+,.2f}",
            "Unrealized %": f"{p.unrealized_plpc:+.2f}%",
        })
    pos_df = pd.DataFrame(pos_rows)
    st.dataframe(style_generic(pos_df), use_container_width=True, hide_index=True)

    # Mini chart: allocation by ticker
    fig = px.pie(
        values=[p.market_value for p in positions],
        names=[p.ticker for p in positions],
        title="Position Allocation",
        hole=0.4,
    )
    fig.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No open positions.")

st.divider()

# ---------------------------------------------------------------------------#
# Circuit breaker
# ---------------------------------------------------------------------------#

st.subheader("🚨 Circuit Breaker")

@st.cache_data(ttl=10)
def _cb_status():
    try:
        from src.trading.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(settings)
        return cb.status(), None
    except Exception as e:
        return None, str(e)


cb_status, cb_err = _cb_status()

if cb_err:
    st.warning(f"Circuit breaker unavailable: {cb_err}")
elif cb_status:
    halted = cb_status["halted"]
    open_val = cb_status.get("portfolio_open_value", 0.0)

    # Status indicator
    if halted:
        st.error(
            f"🛑 **HALTED** — {cb_status.get('halt_reason', 'Unknown reason')}  "
            f"(at {cb_status.get('halt_time', '?')} UTC)"
        )
    else:
        st.success("✅ Armed and monitoring")

    col1, col2, col3 = st.columns(3)
    col1.metric("Opening Portfolio Value", f"${open_val:,.2f}" if open_val else "Not armed")
    col2.metric("Daily Drop Limit",  f"{cb_status['daily_drop_threshold']:.0f}%")
    col3.metric("Single-Stock Limit", f"{cb_status['single_stock_threshold']:.0f}%")

    if cb_status.get("force_sells_today"):
        st.warning(f"Force-sells today: {', '.join(cb_status['force_sells_today'])}")

    # Thresholds table
    st.markdown(f"""
| Trigger | Threshold | Action |
|---------|-----------|--------|
| Portfolio daily drop | {cb_status['daily_drop_threshold']:.0f}% | Halt all new orders |
| Single stock loss from entry | {cb_status['single_stock_threshold']:.0f}% | Force-sell position |
| Manual emergency | Button below | Flatten all positions immediately |
""")

# Emergency stop button
st.subheader("Emergency Stop")
col_warn, col_btn = st.columns([3, 1])
with col_warn:
    st.warning(
        "**Clicking EMERGENCY STOP will immediately close ALL open positions and cancel ALL orders.**  "
        "This cannot be undone."
    )
with col_btn:
    if st.button("🛑 EMERGENCY STOP", type="primary",
                 help="Close every open position immediately via Alpaca"):
        if not settings.alpaca.api_key:
            st.error("Alpaca API key not configured.")
        else:
            with st.spinner("Executing emergency stop …"):
                try:
                    from src.trading.alpaca_client import AlpacaClient
                    from src.trading.circuit_breaker import CircuitBreaker
                    client = AlpacaClient(settings)
                    cb = CircuitBreaker(settings)
                    closed = cb.emergency_stop(client)
                    if closed:
                        st.error(f"Emergency stop executed — closed: {', '.join(closed)}")
                    else:
                        st.info("No positions were open at emergency stop.")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Emergency stop failed: {e}")

st.divider()

# ---------------------------------------------------------------------------#
# Multi-strategy capital allocation
# ---------------------------------------------------------------------------#

st.subheader("Multi-Strategy Capital Allocation")

@st.cache_data(ttl=300)
def _strategy_weights():
    try:
        from src.trading.multi_strategy import MultiStrategyManager
        mgr = MultiStrategyManager(settings)
        alloc = mgr.capital_weights()
        return alloc, None
    except Exception as e:
        return None, str(e)


alloc, alloc_err = _strategy_weights()

if alloc_err:
    st.warning(f"Could not compute strategy weights: {alloc_err}")
else:
    method_label = "Sharpe-weighted (rolling 12 weeks)" if alloc.method == "sharpe" else "Equal weight (< 4 weeks history)"
    st.caption(f"Method: {method_label}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Value Strategy",         f"{alloc.value*100:.1f}%")
    col2.metric("Momentum Strategy",      f"{alloc.momentum*100:.1f}%")
    col3.metric("Mean-Reversion Strategy",f"{alloc.mean_reversion*100:.1f}%")

    # Pie chart
    fig_alloc = px.pie(
        values=[alloc.value, alloc.momentum, alloc.mean_reversion],
        names=["Value", "Momentum", "Mean-Reversion"],
        title="Capital Split by Strategy",
        hole=0.4,
        color_discrete_sequence=["#26a69a", "#42a5f5", "#ffa726"],
    )
    fig_alloc.update_layout(template="plotly_dark", height=280)
    st.plotly_chart(fig_alloc, use_container_width=True)

    if account:
        dollar_alloc = alloc.dollar_allocations(account.portfolio_value)
        alloc_df = pd.DataFrame([
            {"Strategy": "Value",         "Allocation %": f"{alloc.value*100:.1f}%",    "Capital ($)": f"${dollar_alloc['value']:,.0f}"},
            {"Strategy": "Momentum",      "Allocation %": f"{alloc.momentum*100:.1f}%", "Capital ($)": f"${dollar_alloc['momentum']:,.0f}"},
            {"Strategy": "Mean-Reversion","Allocation %": f"{alloc.mean_reversion*100:.1f}%","Capital ($)": f"${dollar_alloc['mean_reversion']:,.0f}"},
        ])
        st.dataframe(style_generic(alloc_df), use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------#
# Risk parameters
# ---------------------------------------------------------------------------#

st.subheader("Risk Parameters")

regime_label = "Unknown"
try:
    from src.analytics.market_regime import MarketRegimeAnalyzer
    snap = MarketRegimeAnalyzer(settings).current_snapshot(compute_breadth=False)
    regime_label = snap.regime.value
    effective_max = settings.max_position_pct * (
        settings.bear_regime_position_scale
        if snap.regime.value in ("Bear", "High-Volatility") else 1.0
    )
except Exception:
    effective_max = settings.max_position_pct

risk_rows = [
    {"Parameter": "Regime",                  "Value": regime_label},
    {"Parameter": "Max position (normal)",   "Value": f"{settings.max_position_pct:.0f}%"},
    {"Parameter": "Max position (Bear/HV)",  "Value": f"{settings.max_position_pct * settings.bear_regime_position_scale:.1f}%"},
    {"Parameter": "Effective max now",        "Value": f"{effective_max:.1f}%"},
    {"Parameter": "Trailing stop",           "Value": f"{settings.trailing_stop_pct:.0f}% after {settings.trailing_stop_days} profitable days"},
    {"Parameter": "Circuit breaker (port.)", "Value": f"Halt at {settings.circuit_breaker_daily_pct:.0f}% daily drop"},
    {"Parameter": "Force-sell (single)",     "Value": f"At {settings.circuit_breaker_single_stock_pct:.0f}% loss from entry"},
    {"Parameter": "Kelly Criterion",         "Value": "Enabled" if settings.kelly_criterion_enabled else "Disabled (fixed % sizing)"},
    {"Parameter": "Trading mode",            "Value": "LIVE" if live_env else "PAPER"},
]
st.dataframe(style_generic(pd.DataFrame(risk_rows)), use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------#
# Dry-run preview
# ---------------------------------------------------------------------------#

st.subheader("Dry-Run Signal Preview")
st.caption("Preview what the trading loop would do right now without placing any real orders.")

if st.button("▶ Run Signal Preview (dry-run)"):
    with st.spinner("Generating signals …"):
        try:
            from src.ranking.ranker import top_picks as get_top_picks
            from src.utils.tickers import get_tickers
            from src.trading.multi_strategy import MultiStrategyManager
            from src.trading.risk import RiskManager

            tickers = get_tickers(settings.ticker_source)
            picks = get_top_picks(n=settings.top_n_picks, settings=settings,
                                  include_ml=False, include_indicators=True,
                                  tickers=tickers)
            held = {p.ticker for p in positions}
            pv = account.portfolio_value if account else 100_000.0

            mgr = MultiStrategyManager(settings)
            risk_mgr = RiskManager(settings)
            all_sigs = mgr.generate_all_signals(picks, positions, pv, tickers=tickers)

            rows = []
            for strategy_name, signals in all_sigs.items():
                for sig in signals:
                    rows.append({
                        "Strategy": strategy_name,
                        "Action":   sig.action,
                        "Ticker":   sig.ticker,
                        "Score":    round(sig.score, 4),
                        "Reason":   sig.reason,
                    })
            if rows:
                sig_df = pd.DataFrame(rows)
                st.metric("Signals generated", len(sig_df))
                st.dataframe(style_generic(sig_df), use_container_width=True, hide_index=True)
            else:
                st.info("No signals generated (no qualifying picks or exits).")
        except Exception as e:
            st.error(f"Dry-run failed: {e}")

st.divider()

# ---------------------------------------------------------------------------#
# Options Positions
# ---------------------------------------------------------------------------#

st.subheader("Options Positions")

@st.cache_data(ttl=60, show_spinner="Loading options positions …")
def _fetch_options_state():
    try:
        from src.trading.options_strategy import OptionsStrategy
        from src.trading.alpaca_client import AlpacaClient
        opt = OptionsStrategy(settings)
        state_positions = opt.load_positions()
        if settings.alpaca.api_key:
            client = AlpacaClient(settings)
            live_positions = client.list_option_positions()
        else:
            live_positions = []
        return state_positions, live_positions
    except Exception as e:
        return [], []


state_opts, live_opts = _fetch_options_state()

# Budget utilization
import math as _math
open_notional = sum(p.market_value for p in live_opts) if live_opts else 0.0
committed = sum(p.entry_premium * p.qty * 100 for p in state_opts) if state_opts else 0.0
budget_display = open_notional if live_opts else committed

opt_col1, opt_col2, opt_col3 = st.columns(3)
opt_col1.metric("Options Budget", f"${settings.options_capital:,.0f}")
opt_col2.metric("Open Notional", f"${budget_display:,.2f}")
opt_col3.metric(
    "Budget Used",
    f"{budget_display / settings.options_capital * 100:.1f}%"
    if settings.options_capital > 0 else "—",
)
st.progress(
    min(budget_display / settings.options_capital, 1.0) if settings.options_capital > 0 else 0.0,
    text=f"${budget_display:,.2f} / ${settings.options_capital:,.0f}",
)

if state_opts:
    import datetime as _dt
    live_map = {p.symbol: p for p in live_opts}
    today_dt = _dt.date.today()

    opt_rows = []
    for s_pos in state_opts:
        live = live_map.get(s_pos.symbol)
        try:
            exp_date = _dt.date.fromisoformat(s_pos.expiration)
            dte = (exp_date - today_dt).days
        except Exception:
            dte = "?"

        # Determine active exit triggers
        triggers = []
        if live:
            if live.unrealized_plpc >= settings.options_profit_target * 100:
                triggers.append("profit target")
            elif live.unrealized_plpc <= -(settings.options_stop_loss * 100):
                triggers.append("stop loss")
        if isinstance(dte, int) and dte <= settings.options_exit_dte:
            triggers.append(f"{dte} DTE guard")

        opt_rows.append({
            "Symbol":        s_pos.symbol,
            "Underlying":    s_pos.underlying,
            "Type":          s_pos.contract_type.upper(),
            "Strike":        f"${s_pos.strike:.0f}",
            "Expiry":        s_pos.expiration,
            "DTE":           dte,
            "Entry Premium": f"${s_pos.entry_premium:.2f}",
            "Current":       f"${live.current_price:.2f}" if live else "—",
            "Unreal. P&L":   f"${live.unrealized_pl:+,.2f}" if live else "—",
            "Unreal. %":     f"{live.unrealized_plpc:+.1f}%" if live else "—",
            "Exit Trigger":  ", ".join(triggers) if triggers else "—",
        })

    opt_df = pd.DataFrame(opt_rows)
    st.dataframe(style_generic(opt_df), use_container_width=True, hide_index=True)
else:
    st.info("No open options positions tracked in data/options_positions.json.")

st.divider()

# ---------------------------------------------------------------------------#
# Portfolio equity history
# ---------------------------------------------------------------------------#

st.subheader("Portfolio Equity History")

@st.cache_data(ttl=300)
def _equity_history():
    try:
        from src.trading.portfolio import PortfolioTracker
        tracker = PortfolioTracker(settings)
        return tracker.equity_history(), None
    except Exception as e:
        return pd.DataFrame(), str(e)


equity_df, eq_err = _equity_history()
if eq_err:
    st.warning(f"Could not load equity history: {eq_err}")
elif equity_df.empty:
    st.info("No EOD portfolio snapshots yet. Snapshots are saved after each `cli/trade.py` run.")
else:
    fig_eq = px.line(equity_df, y="portfolio_value",
                     title="Portfolio Value Over Time",
                     labels={"portfolio_value": "Value ($)", "date": "Date"})
    fig_eq.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_eq, use_container_width=True)
