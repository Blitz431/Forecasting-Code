"""Page 10 — Portfolio Analytics.

Sharpe, Sortino, max drawdown, beta, sector allocation, correlation matrix,
per-stock P&L attribution. Uses src/analytics/ modules from Phase 9.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import correlation_matrix as corr_chart, sector_donut, _empty_fig
from dashboard.components.tables import style_generic

st.set_page_config(page_title="Portfolio", page_icon="💼", layout="wide")
st.title("💼 Portfolio Analytics")
st.caption("Sharpe, Sortino, drawdown, beta, correlation matrix, and per-stock attribution.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Load backtest equity curve (if available) OR Alpaca live portfolio
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=60)
def _load_backtest_equity() -> tuple[pd.Series, pd.Series]:
    """Try to load previously saved backtest equity + SPY curves."""
    out_dir = settings.data_dir / "backtest_results"
    eq_fp  = out_dir / "equity_curve.parquet"
    spy_fp = out_dir / "spy_curve.parquet"

    equity = pd.Series(dtype=float)
    spy    = pd.Series(dtype=float)

    if eq_fp.exists():
        try:
            df = pd.read_parquet(eq_fp)
            equity = df.iloc[:, 0]
        except Exception:
            pass
    if spy_fp.exists():
        try:
            df = pd.read_parquet(spy_fp)
            spy = df.iloc[:, 0]
        except Exception:
            pass
    return equity, spy


@st.cache_data(ttl=60)
def _fetch_alpaca_positions() -> tuple[object | None, list, str]:
    try:
        import alpaca_trade_api as tradeapi
        # The legacy SDK appends its own /v2, so strip it from a user-supplied base URL
        # to avoid hitting /v2/v2/account → 404.
        base_url = settings.alpaca.base_url.rstrip("/")
        if base_url.endswith("/v2"):
            base_url = base_url[:-3]
        api = tradeapi.REST(
            settings.alpaca.api_key,
            settings.alpaca.secret_key,
            base_url,
        )
        return api.get_account(), api.list_positions(), ""
    except Exception as e:
        return None, [], str(e)


# Attempt to load saved backtest
equity_curve, spy_curve = _load_backtest_equity()

# ---------------------------------------------------------------------------#
# Performance metrics section
# ---------------------------------------------------------------------------#

st.subheader("📈 Performance Metrics")

if not equity_curve.empty:
    from src.analytics.portfolio_metrics import compute_metrics, drawdown_series

    spy_bench = spy_curve if not spy_curve.empty else None
    m = compute_metrics(equity_curve, benchmark=spy_bench)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Return",    f"{m.total_return*100:+.1f}%")
    col2.metric("CAGR",            f"{m.cagr*100:+.1f}%")
    col3.metric("Sharpe Ratio",    f"{m.sharpe:.2f}")
    col4.metric("Sortino Ratio",   f"{m.sortino:.2f}")
    col5.metric("Max Drawdown",    f"{m.max_drawdown*100:.1f}%",
                delta_color="inverse")

    col6, col7, col8, col9, col10 = st.columns(5)
    col6.metric("Beta vs SPY",     f"{m.beta:.2f}")
    col7.metric("Alpha (ann.)",    f"{m.alpha*100:+.1f}%")
    col8.metric("Volatility",      f"{m.annualised_vol*100:.1f}%")
    col9.metric("Win Rate",        f"{m.win_rate*100:.1f}%")
    col10.metric("Calmar Ratio",   f"{m.calmar:.2f}")

    # Equity curve chart
    st.divider()
    st.subheader("Equity Curve")
    import plotly.graph_objects as go
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values,
        mode="lines", name="Strategy",
        line=dict(color="#5c7cfa", width=2),
    ))
    if not spy_curve.empty:
        fig_eq.add_trace(go.Scatter(
            x=spy_curve.index, y=spy_curve.values,
            mode="lines", name="SPY Buy & Hold",
            line=dict(color="#ffa726", width=2, dash="dash"),
        ))
    fig_eq.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title="Strategy vs SPY Buy & Hold",
        yaxis_title="Portfolio Value ($)",
        height=400,
        xaxis=dict(gridcolor="#2a2a3a"), yaxis=dict(gridcolor="#2a2a3a"),
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # Drawdown chart
    dd = drawdown_series(equity_curve)
    fig_dd = go.Figure(go.Scatter(
        x=dd.index, y=dd.values * 100,
        mode="lines", fill="tozeroy",
        name="Drawdown %",
        line=dict(color="#ef5350", width=1),
        fillcolor="rgba(239,83,80,0.2)",
    ))
    fig_dd.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title="Drawdown (%)", yaxis_ticksuffix="%",
        height=220,
        xaxis=dict(gridcolor="#2a2a3a"), yaxis=dict(gridcolor="#2a2a3a"),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    if not spy_curve.empty:
        col_ret1, col_ret2, col_ret3 = st.columns(3)
        col_ret1.metric("Strategy Total Return", f"{m.total_return*100:+.1f}%")
        col_ret2.metric("SPY Total Return",       f"{m.benchmark_total_return*100:+.1f}%")
        col_ret3.metric("Excess Return",          f"{m.excess_return*100:+.1f}%",
                        delta_color="normal")
else:
    st.info(
        "No backtest equity curve found.\n\n"
        "Run the backtester first:\n\n"
        "```\npython cli/backtest.py --mode full-sim --save\n```"
    )

st.divider()

# ---------------------------------------------------------------------------#
# Live Alpaca portfolio (if connected)
# ---------------------------------------------------------------------------#

st.subheader("🔌 Live Portfolio (Alpaca)")

if settings.alpaca.api_key:
    account, positions, err = _fetch_alpaca_positions()
    if account is None:
        st.error(f"Alpaca connection failed: {err}")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portfolio Value",  f"${float(account.portfolio_value):,.2f}")
        col2.metric("Buying Power",     f"${float(account.buying_power):,.2f}")
        col3.metric("Cash",             f"${float(account.cash):,.2f}")
        unrealized = sum(float(p.unrealized_pl) for p in positions) if positions else 0
        col4.metric("Unrealized P&L",   f"${unrealized:,.2f}", delta=f"{unrealized:+,.2f}")

        if positions:
            # Build live positions dict for analytics
            live_positions: dict[str, dict] = {}
            for p in positions:
                fp = settings.raw_daily_dir / f"{p.symbol}.parquet"
                cur_price = float(p.current_price)
                live_positions[p.symbol] = {
                    "shares": float(p.qty),
                    "entry_price": float(p.avg_entry_price),
                    "current_price": cur_price,
                }

            # Attribution table
            st.subheader("P&L Attribution")
            from src.analytics.attribution import compute_attribution, attribution_dataframe
            from src.analytics.sector_analysis import _STATIC_SECTORS
            attr = compute_attribution(live_positions,
                                       portfolio_value=float(account.portfolio_value),
                                       sector_map=_STATIC_SECTORS)
            if attr:
                attr_df = attribution_dataframe(attr)
                st.dataframe(style_generic(attr_df), use_container_width=True)

            # Sector allocation
            st.subheader("Sector Allocation")
            from src.analytics.sector_analysis import SectorAnalyzer
            sa = SectorAnalyzer(settings)
            pos_values = {p.symbol: float(p.market_value) for p in positions}
            sw = sa.sector_weights_series(pos_values)
            if not sw.empty:
                col_donut, col_warn = st.columns([2, 1])
                with col_donut:
                    st.plotly_chart(sector_donut(sw), use_container_width=True)
                with col_warn:
                    warnings = sa.concentration_warnings(pos_values)
                    if warnings:
                        st.warning("**Concentration Warnings**")
                        for w in warnings:
                            st.write(f"- {w}")
                    else:
                        st.success("No concentration warnings — portfolio is well diversified.")

            # Correlation matrix
            st.subheader("Holding Correlation (63-day)")
            from src.analytics.correlation import CorrelationAnalyzer
            ca = CorrelationAnalyzer(settings)
            tickers = [p.symbol for p in positions]
            corr_df = ca.compute(tickers, lookback_days=63)
            if not corr_df.empty:
                st.plotly_chart(corr_chart(corr_df), use_container_width=True)
                pairs = ca.highly_correlated(corr_df)
                if pairs:
                    st.warning("**High Correlation Pairs**")
                    for pair in pairs[:5]:
                        st.write(f"- {pair.warning}")
            else:
                st.info("Insufficient daily data to compute correlations.")
        else:
            st.info("No open positions.")
else:
    st.warning(
        "Alpaca API keys not configured. "
        "Add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` to your `.env` to see live data."
    )

st.divider()

# ---------------------------------------------------------------------------#
# Sector rotation
# ---------------------------------------------------------------------------#

st.subheader("🔄 Sector Rotation (63-day momentum)")

@st.cache_data(ttl=1800, show_spinner="Computing sector rotation …")
def _sector_rotation():
    from src.analytics.sector_analysis import SectorAnalyzer
    sa = SectorAnalyzer(settings)
    return sa.detect_rotation(lookback_days=63)

rot_df = _sector_rotation()
if not rot_df.empty:
    import plotly.express as px
    fig_rot = px.bar(
        rot_df.sort_values("Momentum"),
        x="Momentum", y="Sector",
        orientation="h",
        color="Momentum",
        color_continuous_scale="RdYlGn",
        title="Sector Rotation — 63-day Momentum Score",
        labels={"Momentum": "Momentum (recent - prior period return)"},
    )
    fig_rot.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=400,
    )
    st.plotly_chart(fig_rot, use_container_width=True)
    st.dataframe(style_generic(rot_df), use_container_width=True)
else:
    st.info("Sector rotation data unavailable. Requires sector ETF data (XLK, XLF, etc.).")
