"""Page 13 — Market Regime.

Bull / Bear / Sideways / High-Volatility detection using VIX, yield curve,
and market breadth.  Full analytics in Phase 9 (market_regime.py).
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import regime_gauge, price_line, _empty_fig
from dashboard.components.tables import style_generic

st.set_page_config(page_title="Market Regime", page_icon="🌡️", layout="wide")
st.title("🌡️ Market Regime")
st.caption("Real-time Bull/Bear/Sideways/High-Vol classification using VIX, yield curve, and breadth.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Load FRED macro data
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=600)
def _load_fred(series_id: str) -> pd.DataFrame:
    fp = settings.raw_macro_dir / f"{series_id}.parquet"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_parquet(fp)

@st.cache_data(ttl=300)
def _load_spy() -> pd.DataFrame:
    fp = settings.raw_daily_dir / "SPY.parquet"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_parquet(fp)

vix_df      = _load_fred("VIXCLS")
t10y_df     = _load_fred("DGS10")
t2y_df      = _load_fred("DGS2")
t10y2y_df   = _load_fred("T10Y2Y")
spy_df      = _load_spy()

# ---------------------------------------------------------------------------#
# Compute regime (simple heuristic — Phase 9 will use src/analytics/market_regime.py)
# ---------------------------------------------------------------------------#

def _classify_regime(vix_val: float | None, spread: float | None, spy_50d_ret: float | None) -> str:
    if vix_val is None:
        return "Unknown"
    if vix_val > 30:
        return "High-Volatility / Crisis"
    if vix_val > 20:
        if spy_50d_ret is not None and spy_50d_ret < -0.10:
            return "Bear"
        return "Elevated Volatility"
    if spread is not None and spread < 0:
        return "Inverted Yield Curve"
    if spy_50d_ret is not None and spy_50d_ret > 0.05:
        return "Bull"
    if spy_50d_ret is not None and spy_50d_ret < -0.05:
        return "Bear"
    return "Sideways / Neutral"

vix_val     = float(vix_df.iloc[-1].iloc[0])    if not vix_df.empty    else None
spread_val  = float(t10y2y_df.iloc[-1].iloc[0]) if not t10y2y_df.empty else None

spy_50d_ret = None
if not spy_df.empty and "Close" in spy_df.columns and len(spy_df) >= 50:
    last   = float(spy_df["Close"].iloc[-1])
    ago50  = float(spy_df["Close"].iloc[-50])
    spy_50d_ret = (last / ago50) - 1

regime_label = _classify_regime(vix_val, spread_val, spy_50d_ret)

# ---------------------------------------------------------------------------#
# Summary metrics
# ---------------------------------------------------------------------------#

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Regime",       regime_label)
col2.metric("VIX",          f"{vix_val:.1f}" if vix_val else "—",
            delta="High" if vix_val and vix_val > 25 else None,
            delta_color="inverse")
col3.metric("10Y-2Y Spread",f"{spread_val:.2f}%" if spread_val is not None else "—",
            delta="Inverted" if spread_val is not None and spread_val < 0 else "Normal",
            delta_color="inverse" if spread_val is not None and spread_val < 0 else "normal")
col4.metric("SPY 50d Return",f"{spy_50d_ret*100:.1f}%" if spy_50d_ret is not None else "—")
t10_val = float(t10y_df.iloc[-1].iloc[0]) if not t10y_df.empty else None
col5.metric("10Y Treasury",  f"{t10_val:.2f}%" if t10_val else "—")

st.divider()

# ---------------------------------------------------------------------------#
# Regime gauge + VIX chart
# ---------------------------------------------------------------------------#

col_gauge, col_chart = st.columns([1, 2])

with col_gauge:
    st.plotly_chart(regime_gauge(vix_val, regime_label), use_container_width=True)

with col_chart:
    if not vix_df.empty:
        import plotly.graph_objects as go
        vix_ts = vix_df.copy()
        vix_col = vix_ts.columns[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vix_ts.index, y=vix_ts[vix_col],
            mode="lines", name="VIX",
            line=dict(color="#ffa726", width=2),
            fill="tozeroy", fillcolor="rgba(255,167,38,0.1)",
        ))
        for level, color, label in [(15, "#26a69a", "Low Risk"),
                                      (25, "#ffa726", "Elevated"),
                                      (35, "#ef5350", "High Risk")]:
            fig.add_hline(y=level, line_dash="dot", line_color=color,
                          annotation_text=label, annotation_position="right")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            title="VIX — Volatility Index (Full History)", height=300,
            xaxis=dict(gridcolor="#2a2a3a"),
            yaxis=dict(gridcolor="#2a2a3a"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("VIX data not available. Run `python cli/scrape.py` to fetch FRED data.")

st.divider()

# ---------------------------------------------------------------------------#
# Yield curve
# ---------------------------------------------------------------------------#

st.subheader("Yield Curve")

if not t10y2y_df.empty:
    import plotly.graph_objects as go
    fig2 = go.Figure()
    spread_col = t10y2y_df.columns[0]
    fig2.add_trace(go.Scatter(
        x=t10y2y_df.index, y=t10y2y_df[spread_col],
        mode="lines", name="10Y-2Y Spread",
        line=dict(color="#5c7cfa", width=2),
    ))
    fig2.add_hline(y=0, line_dash="solid", line_color="#ef5350",
                   annotation_text="Inversion threshold")
    fig2.add_hrect(y0=-5, y1=0, fillcolor="rgba(239,83,80,0.1)", line_width=0)
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title="10Y-2Y Treasury Spread (Yield Curve)", height=280,
        xaxis=dict(gridcolor="#2a2a3a"), yaxis=dict(gridcolor="#2a2a3a"),
        yaxis_ticksuffix="%",
    )
    st.plotly_chart(fig2, use_container_width=True)
    if spread_val is not None and spread_val < 0:
        st.error(f"⚠️ Inverted yield curve ({spread_val:.2f}%). "
                 "Historically a recession indicator — consider defensive positioning.")
else:
    st.info("Yield curve data not available (requires FRED API). Run `python cli/scrape.py`.")

st.divider()

# ---------------------------------------------------------------------------#
# Phase 9 note
# ---------------------------------------------------------------------------#

st.info(
    "**Phase 9** will add the full `src/analytics/market_regime.py` module:\n\n"
    "- Automated Bull/Bear/Sideways classification with configurable thresholds\n"
    "- Market breadth (% of S&P 500 stocks above 200-day MA)\n"
    "- Regime-aware strategy aggressiveness adjustment\n"
    "- Historical regime overlay on backtest equity curve"
)
