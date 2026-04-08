"""Page 11 — Backtesting.

Strategy simulation 2020-2026 with equity curve and trade log.
Full implementation in Phase 9.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import price_line, _empty_fig
from dashboard.components.tables import style_generic

st.set_page_config(page_title="Backtesting", page_icon="⏮️", layout="wide")
st.title("⏮️ Backtesting")
st.caption("Full paper-trading simulation 2020–2026 across COVID crash, bear market, and recovery.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Phase note
# ---------------------------------------------------------------------------#

st.info(
    "**Phase 9** will implement the full backtester:\n\n"
    "- Replay each trading day 2020–2026 as if running live\n"
    "- Signals computed with no future data leakage\n"
    "- Full slippage & spread modeling\n"
    "- COVID crash (Mar 2020), 2022 bear, 2023-2025 bull run\n"
    "- Equity curve, Sharpe, max drawdown, comparison vs S&P 500 buy-and-hold\n\n"
    "Coming soon: `python cli/backtest.py --mode full-sim --train 2015-2020 --test 2020-2026`"
)

st.divider()

# ---------------------------------------------------------------------------#
# Preview: show S&P 500 (SPY) price history as reference
# ---------------------------------------------------------------------------#

st.subheader("S&P 500 Reference Chart (SPY)")

@st.cache_data(ttl=300)
def _load_spy():
    fp = settings.raw_daily_dir / "SPY.parquet"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_parquet(fp)

spy_df = _load_spy()

if not spy_df.empty:
    # Filter to 2020-present to match the planned backtest window
    spy_2020 = spy_df[spy_df.index >= "2020-01-01"]
    if not spy_2020.empty:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=spy_2020.index,
            y=spy_2020["Close"],
            mode="lines",
            name="SPY (Buy & Hold)",
            line=dict(color="#5c7cfa", width=2),
            fill="tozeroy",
            fillcolor="rgba(92,124,250,0.1)",
        ))

        # Mark regime events
        events = {
            "COVID Crash": "2020-03-23",
            "Recovery":    "2020-11-01",
            "2022 Bear":   "2022-01-03",
            "2023 Rally":  "2023-01-01",
        }
        for label, date_str in events.items():
            ts = pd.Timestamp(date_str)
            if ts in spy_2020.index or (spy_2020.index.min() <= ts <= spy_2020.index.max()):
                fig.add_vline(x=ts, line_dash="dash", line_color="#888",
                              annotation_text=label, annotation_position="top right",
                              annotation_font_color="#aaa")

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            title="SPY 2020-Present — Backtest Benchmark",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cumulative return
        first_price = float(spy_2020["Close"].iloc[0])
        last_price  = float(spy_2020["Close"].iloc[-1])
        total_return = (last_price / first_price - 1) * 100
        st.metric("SPY Total Return (2020–present)", f"{total_return:.1f}%",
                  help="This is the buy-and-hold benchmark the backtester will compare against.")
else:
    st.info("SPY data not available. Run `python cli/scrape.py` with SPY included.")

st.divider()

# ---------------------------------------------------------------------------#
# Backtest configuration preview
# ---------------------------------------------------------------------------#

st.subheader("Planned Backtest Configuration")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Simulation Settings**
    | Parameter | Value |
    |-----------|-------|
    | Train window | 2015-01-01 → 2019-12-31 |
    | Test window | 2020-01-01 → 2026-12-31 |
    | Rebalance | Weekly (every Friday) |
    | Max position size | 5% of portfolio |
    | Trailing stop | 5% after 3 profitable days |
    | Circuit breaker | Halt at -10% portfolio / day |
    """)
with col2:
    st.markdown("""
    **Performance Metrics Planned**
    - Cumulative return vs SPY
    - Sharpe ratio (annualized)
    - Sortino ratio
    - Max drawdown & recovery
    - Win rate, avg win/loss
    - Calmar ratio
    - Regime-breakdown analysis
    """)
