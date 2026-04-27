"""Page 11 — Backtesting.

Full paper-trading simulation results dashboard.
Displays equity curve, drawdown, trade log, regime overlay, and performance metrics.
Run the backtester first: python cli/backtest.py --mode full-sim --save
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.tables import style_generic

st.set_page_config(page_title="Backtesting", page_icon="⏮️", layout="wide")
st.title("⏮️ Backtesting")
st.caption("Paper-trading simulation 2020–2026 — equity curve, trade log, regime overlay, Sharpe, max drawdown vs SPY.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Helper: load saved results
# ---------------------------------------------------------------------------#

_RESULTS_DIR = settings.data_dir / "backtest_results"


@st.cache_data(ttl=60)
def _load_results():
    results = {}

    for fname in ["equity_curve.parquet", "spy_curve.parquet",
                  "drawdown_curve.parquet", "trade_log.parquet"]:
        fp = _RESULTS_DIR / fname
        if fp.exists():
            try:
                results[fname.replace(".parquet", "")] = pd.read_parquet(fp)
            except Exception:
                pass

    metrics_fp = _RESULTS_DIR / "metrics.json"
    if metrics_fp.exists():
        try:
            results["metrics"] = json.loads(metrics_fp.read_text())
        except Exception:
            pass

    return results


res = _load_results()

# ---------------------------------------------------------------------------#
# Run controls
# ---------------------------------------------------------------------------#

with st.expander("⚙️ Run Backtest", expanded=not bool(res)):
    col1, col2 = st.columns(2)
    with col1:
        train_w = st.text_input("Train window", "2015-2020",
                                help="YYYY-YYYY (signals trained on this period)")
        test_w  = st.text_input("Test window",  "2020-2026",
                                help="YYYY-YYYY (simulation runs this period)")
        capital = st.number_input("Initial capital ($)", value=100_000, step=10_000)
    with col2:
        top_n      = st.slider("Max concurrent positions", 5, 50, 20)
        slippage   = st.slider("Slippage (bps)",  0, 20, 5)
        circuit    = st.slider("Circuit breaker daily drop %", 5, 20, 10)

    run_btn = st.button("▶ Run Full Simulation", type="primary",
                        help="May take several minutes for a 6-year window with 500 tickers")

    if run_btn:
        from src.trading.backtester import Backtester, BacktestConfig

        def _parse_yrange(s):
            parts = s.split("-")
            return f"{parts[0]}-01-01", f"{parts[1]}-12-31"

        train_s, train_e = _parse_yrange(train_w)
        test_s,  test_e  = _parse_yrange(test_w)

        config = BacktestConfig(
            train_start=train_s, train_end=train_e,
            test_start=test_s,   test_end=test_e,
            initial_capital=float(capital),
            max_position_pct=1.0 / max(top_n, 1),
            top_n=top_n,
            slippage_bps=float(slippage),
            circuit_breaker_pct=float(circuit) / 100.0,
        )

        progress_bar = st.progress(0, text="Simulating …")

        def _cb(day: int, total: int) -> None:
            progress_bar.progress(min(day / max(total, 1), 1.0),
                                  text=f"Day {day}/{total}")

        with st.spinner("Running simulation …"):
            bt = Backtester(config, settings)
            result = bt.run(progress_callback=_cb)

        # Save results
        out_dir = settings.data_dir / "backtest_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_frame("portfolio_value").to_parquet(
            out_dir / "equity_curve.parquet")
        if not result.spy_curve.empty:
            result.spy_curve.to_frame("spy_value").to_parquet(
                out_dir / "spy_curve.parquet")
        if not result.trade_log_df.empty:
            result.trade_log_df.to_parquet(out_dir / "trade_log.parquet", index=False)
        if not result.drawdown_curve.empty:
            result.drawdown_curve.to_frame("drawdown").to_parquet(
                out_dir / "drawdown_curve.parquet")
        m = result.metrics
        metrics_dict = m.to_dict()
        metrics_dict["spy"] = result.spy_metrics.to_dict()
        metrics_dict["total_trades"] = result.total_trades
        metrics_dict["winning_trades"] = result.winning_trades
        metrics_dict["losing_trades"] = result.losing_trades
        metrics_dict["circuit_breaker_hits"] = result.circuit_breaker_hits
        (out_dir / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))

        st.success(f"Simulation complete! {result.total_trades} trades executed.")
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------------------------#
# Display results (if available)
# ---------------------------------------------------------------------------#

if not res:
    st.info(
        "No backtest results yet.\n\n"
        "Click **Run Full Simulation** above, or run from CLI:\n\n"
        "```\npython cli/backtest.py --mode full-sim --train 2015-2020 --test 2020-2026 --save\n```"
    )
    st.stop()

# Performance summary
st.subheader("📊 Performance Summary")
metrics = res.get("metrics", {})
spy_m   = metrics.get("spy", {})

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Return",
            f"{metrics.get('total_return',0)*100:+.1f}%",
            delta=f"vs SPY {spy_m.get('total_return',0)*100:+.1f}%" if spy_m else None)
col2.metric("CAGR",         f"{metrics.get('cagr',0)*100:+.1f}%")
col3.metric("Sharpe Ratio", f"{metrics.get('sharpe',0):.2f}")
col4.metric("Max Drawdown", f"{metrics.get('max_drawdown',0)*100:.1f}%",
            delta_color="inverse")
col5.metric("Win Rate",     f"{metrics.get('win_rate',0)*100:.1f}%")

col6, col7, col8, col9, col10 = st.columns(5)
col6.metric("Sortino",        f"{metrics.get('sortino',0):.2f}")
col7.metric("Beta",           f"{metrics.get('beta',1.0):.2f}")
col8.metric("Alpha (ann.)",   f"{metrics.get('alpha',0)*100:+.1f}%")
col9.metric("Total Trades",   str(metrics.get("total_trades", 0)))
col10.metric("Circuit Breaks", str(metrics.get("circuit_breaker_hits", 0)))

st.divider()

# Equity curve
st.subheader("📈 Equity Curve")
eq_df  = res.get("equity_curve")
spy_df = res.get("spy_curve")

import plotly.graph_objects as go

if eq_df is not None and not eq_df.empty:
    eq_series  = eq_df.iloc[:, 0]
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=eq_series.index, y=eq_series.values,
        mode="lines", name="Strategy",
        line=dict(color="#5c7cfa", width=2),
        fill="tozeroy", fillcolor="rgba(92,124,250,0.07)",
    ))
    if spy_df is not None and not spy_df.empty:
        spy_series = spy_df.iloc[:, 0]
        fig_eq.add_trace(go.Scatter(
            x=spy_series.index, y=spy_series.values,
            mode="lines", name="SPY Buy & Hold",
            line=dict(color="#ffa726", width=2, dash="dash"),
        ))

    # Regime events annotation
    events = {"COVID Crash": "2020-03-23", "Recovery": "2020-11-01",
              "2022 Bear": "2022-01-03", "2023 Rally": "2023-01-01"}
    for label, date_str in events.items():
        ts = pd.Timestamp(date_str)
        if eq_series.index.min() <= ts <= eq_series.index.max():
            fig_eq.add_vline(x=ts.value // 10**6, line_dash="dash", line_color="#555",
                             annotation_text=label, annotation_position="top right",
                             annotation_font_color="#999")

    fig_eq.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title="Strategy Equity Curve vs SPY Buy & Hold",
        yaxis_title="Portfolio Value ($)", height=450,
        xaxis=dict(gridcolor="#2a2a3a"), yaxis=dict(gridcolor="#2a2a3a"),
    )
    st.plotly_chart(fig_eq, use_container_width=True)

# Drawdown chart
dd_df = res.get("drawdown_curve")
if dd_df is not None and not dd_df.empty:
    dd_series = dd_df.iloc[:, 0] * 100
    fig_dd = go.Figure(go.Scatter(
        x=dd_series.index, y=dd_series.values,
        mode="lines", fill="tozeroy",
        name="Drawdown %",
        line=dict(color="#ef5350", width=1),
        fillcolor="rgba(239,83,80,0.2)",
    ))
    fig_dd.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title="Portfolio Drawdown (%)",
        yaxis_ticksuffix="%", height=220,
        xaxis=dict(gridcolor="#2a2a3a"), yaxis=dict(gridcolor="#2a2a3a"),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

st.divider()

# Trade log
st.subheader("📋 Trade Log")
trade_df = res.get("trade_log")
if trade_df is not None and not trade_df.empty:
    n_trades = len(trade_df)

    tab_all, tab_wins, tab_losses, tab_stops = st.tabs([
        f"All ({n_trades})",
        f"Wins ({metrics.get('winning_trades', 0)})",
        f"Losses ({metrics.get('losing_trades', 0)})",
        "Trailing Stops & Circuit Breaks",
    ])

    with tab_all:
        st.dataframe(style_generic(trade_df), use_container_width=True)

    with tab_wins:
        wins = trade_df[trade_df["P&L ($)"] > 0]
        if wins.empty:
            st.info("No winning trades.")
        else:
            st.dataframe(style_generic(wins), use_container_width=True)

    with tab_losses:
        losses = trade_df[trade_df["P&L ($)"] < 0]
        if losses.empty:
            st.info("No losing trades.")
        else:
            st.dataframe(style_generic(losses), use_container_width=True)

    with tab_stops:
        stops = trade_df[trade_df["Exit Reason"].isin(["trailing_stop", "circuit_breaker"])]
        if stops.empty:
            st.info("No trailing stops or circuit breaker triggers.")
        else:
            st.dataframe(style_generic(stops), use_container_width=True)

    # Distribution of P&L %
    import plotly.express as px
    fig_hist = px.histogram(
        trade_df, x="P&L %", nbins=40,
        title="Trade P&L % Distribution",
        color_discrete_sequence=["#5c7cfa"],
    )
    fig_hist.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=300,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Best / worst trades
    col_best, col_worst = st.columns(2)
    with col_best:
        st.markdown("**Top 5 Best Trades**")
        st.dataframe(
            style_generic(trade_df.nlargest(5, "P&L ($)")),
            use_container_width=True,
        )
    with col_worst:
        st.markdown("**Top 5 Worst Trades**")
        st.dataframe(
            style_generic(trade_df.nsmallest(5, "P&L ($)")),
            use_container_width=True,
        )
else:
    st.info("No trade log available. Run the simulation to generate results.")
