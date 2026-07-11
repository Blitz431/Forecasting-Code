"""Page 13 — Market Regime.

Bull / Bear / Sideways / High-Volatility detection using:
  - VIX level
  - 10Y-2Y yield curve spread
  - SPY 50-day and 200-day momentum
  - Market breadth (% of S&P 500 above 200-day MA)

Uses src/analytics/market_regime.py from Phase 9.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import regime_gauge, _empty_fig
from dashboard.components.tables import style_generic

st.set_page_config(page_title="Market Regime", page_icon="🌡️", layout="wide")

from dashboard.components.market_clock import render_market_clock
render_market_clock()

st.title("🌡️ Market Regime")
st.caption("Real-time Bull/Bear/Sideways/High-Vol classification using VIX, yield curve, SPY momentum, and market breadth.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Compute current regime snapshot
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=600, show_spinner="Analysing market regime …")
def _current_snapshot(compute_breadth: bool):
    from src.analytics.market_regime import MarketRegimeAnalyzer
    mra = MarketRegimeAnalyzer(settings)
    return mra.current_snapshot(compute_breadth=compute_breadth)


compute_breadth = st.sidebar.toggle(
    "Compute market breadth",
    value=False,
    help="Scans all tickers — slow on first run (~30 sec)",
)

snapshot = _current_snapshot(compute_breadth)

# ---------------------------------------------------------------------------#
# Top-row: regime banner + key metrics
# ---------------------------------------------------------------------------#

regime_color = snapshot.regime.color()

st.markdown(
    f"""<div style="background:{regime_color}22; border-left:4px solid {regime_color};
        padding:12px 16px; border-radius:4px; margin-bottom:16px;">
        <span style="font-size:1.4rem; font-weight:700; color:{regime_color};">
        {snapshot.regime.value}</span>
        &nbsp;&nbsp;<span style="color:#ccc;">Current Market Regime</span><br>
        <small style="color:#aaa;">{snapshot.regime.strategy_hint()}</small>
    </div>""",
    unsafe_allow_html=True,
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("VIX",
            f"{snapshot.vix:.1f}" if snapshot.vix else "—",
            delta="High" if snapshot.vix and snapshot.vix > 25 else None,
            delta_color="inverse")
col2.metric("10Y-2Y Spread",
            f"{snapshot.yield_spread:.2f}%" if snapshot.yield_spread is not None else "—",
            delta="Inverted" if snapshot.yield_spread is not None and snapshot.yield_spread < 0 else "Normal",
            delta_color="inverse" if snapshot.yield_spread is not None and snapshot.yield_spread < 0 else "normal")
col3.metric("SPY 50d Return",
            f"{snapshot.spy_50d_return*100:.1f}%" if snapshot.spy_50d_return is not None else "—")
col4.metric("SPY 200d Return",
            f"{snapshot.spy_200d_return*100:.1f}%" if snapshot.spy_200d_return is not None else "—")
col5.metric("Breadth (>200d MA)",
            f"{snapshot.breadth_pct:.0f}%" if snapshot.breadth_pct is not None else "—",
            help="% of S&P 500 stocks above their 200-day MA")

st.divider()

# ---------------------------------------------------------------------------#
# Gauge + VIX chart
# ---------------------------------------------------------------------------#

col_gauge, col_chart = st.columns([1, 2])

with col_gauge:
    st.plotly_chart(regime_gauge(snapshot.vix, snapshot.regime.value),
                    use_container_width=True)
    st.markdown("**Factor breakdown**")
    for factor, desc in snapshot.details.items():
        st.markdown(f"- **{factor.replace('_', ' ').title()}**: {desc}")
    st.markdown(f"**Aggregate score**: `{snapshot.score:+.2f}`")

with col_chart:
    @st.cache_data(ttl=600)
    def _load_fred(series_id: str) -> pd.DataFrame:
        fp = settings.raw_macro_dir / f"{series_id}.parquet"
        return pd.read_parquet(fp) if fp.exists() else pd.DataFrame()

    vix_df = _load_fred("VIXCLS")
    if not vix_df.empty:
        import plotly.graph_objects as go
        vix_col = vix_df.columns[0]
        fig_vix = go.Figure()
        fig_vix.add_trace(go.Scatter(
            x=vix_df.index, y=vix_df[vix_col],
            mode="lines", name="VIX",
            line=dict(color="#ffa726", width=2),
            fill="tozeroy", fillcolor="rgba(255,167,38,0.1)",
        ))
        for level, color, label in [(15, "#26a69a", "Low Risk"),
                                     (25, "#ffa726", "Elevated"),
                                     (35, "#ef5350", "High Risk")]:
            fig_vix.add_hline(y=level, line_dash="dot", line_color=color,
                              annotation_text=label, annotation_position="right")
        # Shade high-vol periods
        fig_vix.add_vrect(x0="2020-02-01", x1="2020-05-01",
                          fillcolor="rgba(239,83,80,0.07)", line_width=0,
                          annotation_text="COVID", annotation_position="top left")
        fig_vix.add_vrect(x0="2022-01-01", x1="2022-12-31",
                          fillcolor="rgba(239,83,80,0.07)", line_width=0,
                          annotation_text="2022 Bear", annotation_position="top left")
        fig_vix.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            title="VIX — Full History",
            height=320,
            xaxis=dict(gridcolor="#2a2a3a"), yaxis=dict(gridcolor="#2a2a3a"),
        )
        st.plotly_chart(fig_vix, use_container_width=True)
    else:
        st.info("VIX data not available. Run `python cli/scrape.py` to fetch FRED data.")

st.divider()

# ---------------------------------------------------------------------------#
# Yield curve
# ---------------------------------------------------------------------------#

st.subheader("Yield Curve (10Y - 2Y Spread)")

@st.cache_data(ttl=600)
def _load_spread():
    fp = settings.raw_macro_dir / "T10Y2Y.parquet"
    return pd.read_parquet(fp) if fp.exists() else pd.DataFrame()

spread_df = _load_spread()
if not spread_df.empty:
    import plotly.graph_objects as go
    sc = spread_df.columns[0]
    fig_spread = go.Figure()
    fig_spread.add_trace(go.Scatter(
        x=spread_df.index, y=spread_df[sc],
        mode="lines", name="10Y-2Y",
        line=dict(color="#5c7cfa", width=2),
    ))
    fig_spread.add_hline(y=0, line_dash="solid", line_color="#ef5350",
                         annotation_text="Inversion line")
    fig_spread.add_hrect(y0=-5, y1=0, fillcolor="rgba(239,83,80,0.08)", line_width=0)
    fig_spread.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        title="10Y-2Y Treasury Spread",
        height=260, yaxis_ticksuffix="%",
        xaxis=dict(gridcolor="#2a2a3a"), yaxis=dict(gridcolor="#2a2a3a"),
    )
    st.plotly_chart(fig_spread, use_container_width=True)

    spread_val = snapshot.yield_spread
    if spread_val is not None and spread_val < 0:
        st.error(
            f"⚠️ Yield curve inverted at {spread_val:.2f}%. "
            "Historically a leading recession indicator."
        )
else:
    st.info("Yield curve data unavailable (requires FRED API). Run `python cli/scrape.py`.")

st.divider()

# ---------------------------------------------------------------------------#
# Historical regime overlay
# ---------------------------------------------------------------------------#

st.subheader("Historical Regime Timeline")

@st.cache_data(ttl=1800, show_spinner="Building regime history …")
def _historical_regimes():
    from src.analytics.market_regime import MarketRegimeAnalyzer
    mra = MarketRegimeAnalyzer(settings)
    return mra.historical_regimes(window=63)

if st.button("Load historical regime timeline"):
    regime_hist = _historical_regimes()

    if not regime_hist.empty:
        import plotly.express as px
        import plotly.graph_objects as go

        COLORS = {
            "Bull": "#26a69a", "Bear": "#ef5350",
            "Sideways": "#ffa726", "Recovery": "#66bb6a",
            "High-Volatility": "#ab47bc", "Unknown": "#9e9e9e",
        }

        fig_hist = go.Figure()
        # Regime score line
        fig_hist.add_trace(go.Scatter(
            x=regime_hist.index, y=regime_hist["score"],
            mode="lines", name="Regime Score",
            line=dict(color="#5c7cfa", width=2),
        ))
        fig_hist.add_hline(y=0, line_dash="dash", line_color="#888")
        fig_hist.add_hline(y=2.5,  line_dash="dot", line_color="#26a69a",
                           annotation_text="Bull threshold")
        fig_hist.add_hline(y=-2.5, line_dash="dot", line_color="#ef5350",
                           annotation_text="Bear threshold")
        fig_hist.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            title="Market Regime Score (rolling 63-day window)",
            height=320,
            xaxis=dict(gridcolor="#2a2a3a"), yaxis=dict(gridcolor="#2a2a3a"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Regime distribution pie
        regime_counts = regime_hist["regime"].value_counts()
        import plotly.express as px
        fig_pie = px.pie(
            values=regime_counts.values,
            names=regime_counts.index,
            title="Regime Distribution (historical)",
            color=regime_counts.index,
            color_discrete_map=COLORS,
        )
        fig_pie.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Not enough data to build regime history.")
