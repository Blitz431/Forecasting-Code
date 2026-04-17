"""Page 16 — Peer Comparison.

Stock vs sector peers: relative strength, price performance, signal comparison.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import price_line, _empty_fig
from dashboard.components.tables import style_ranking_table, style_generic
from src.utils.input_sanitize import clean_ticker_list

st.set_page_config(page_title="Peer Comparison", page_icon="⚖️", layout="wide")
st.title("⚖️ Peer Comparison")
st.caption("Compare a ticker against its sector peers — relative strength, signals, and performance.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Sector peer map (curated top-10 by sector)
# ---------------------------------------------------------------------------#

SECTOR_PEERS: dict[str, list[str]] = {
    "Technology":          ["AAPL", "MSFT", "NVDA", "AVGO", "AMD", "INTC", "QCOM", "TXN", "CRM", "ORCL"],
    "Healthcare":          ["JNJ", "UNH", "LLY", "ABBV", "ABT", "MRK", "TMO", "DHR", "AMGN", "ISRG"],
    "Financials":          ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "SCHW", "USB"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TGT", "BKNG", "GM"],
    "Communication":       ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "SNAP", "TTWO"],
    "Industrials":         ["UNP", "HON", "RTX", "CAT", "BA", "DE", "MMM", "GE", "FDX", "UPS"],
    "Energy":              ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "PXD"],
    "Consumer Staples":    ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "GIS"],
    "Utilities":           ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PEG", "XEL", "ED"],
    "Real Estate":         ["PLD", "AMT", "EQIX", "CCI", "PSA", "WELL", "SPG", "O", "DLR", "EQR"],
    "Materials":           ["LIN", "APD", "ECL", "SHW", "FCX", "NEM", "NUE", "VMC", "MLM", "ALB"],
}

# ---------------------------------------------------------------------------#
# Controls
# ---------------------------------------------------------------------------#

col1, col2 = st.columns([2, 2])

with col1:
    sector = st.selectbox("Sector", list(SECTOR_PEERS.keys()))
    peers  = SECTOR_PEERS[sector]

with col2:
    selected = st.selectbox("Primary ticker", peers)
    custom_peers = st.text_input("Custom peer list (comma-sep)", placeholder="AAPL,MSFT,NVDA")

if custom_peers.strip():
    parsed_peers, rejected = clean_ticker_list(custom_peers)
    if rejected:
        st.warning(f"Ignored invalid peer(s): {', '.join(rejected[:10])}")
    if parsed_peers:
        peers = parsed_peers

# Ensure primary ticker is in list
if selected not in peers:
    peers = [selected] + peers

lookback = st.slider("Lookback for relative performance (days)", 30, 756, 252)

# ---------------------------------------------------------------------------#
# Load price data for all peers
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300)
def _load_peers(tickers: tuple, lookback_days: int) -> pd.DataFrame:
    """Load Close prices for all peer tickers, forward-fill, return aligned DataFrame."""
    frames = {}
    for t in tickers:
        fp = settings.raw_daily_dir / f"{t}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        if "Close" in df.columns and not df.empty:
            frames[t] = df["Close"].tail(lookback_days)

    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).dropna(how="all").ffill()

peer_prices = _load_peers(tuple(peers), lookback)

if peer_prices.empty:
    st.warning(
        "No price data for selected peers. "
        "Run `python cli/scrape.py --tickers " + ",".join(peers[:5]) + "` to scrape data."
    )
    st.stop()

# ---------------------------------------------------------------------------#
# Relative strength (indexed to 100)
# ---------------------------------------------------------------------------#

st.divider()
st.subheader("Relative Performance (Indexed to 100)")

import plotly.graph_objects as go
import plotly.express as px

# Index all series to 100 at start
indexed = peer_prices.div(peer_prices.iloc[0]) * 100

fig = go.Figure()
colors = px.colors.qualitative.Plotly

for i, col in enumerate(indexed.columns):
    width = 3 if col == selected else 1.5
    dash  = "solid" if col == selected else "dot"
    fig.add_trace(go.Scatter(
        x=indexed.index, y=indexed[col],
        mode="lines",
        name=col,
        line=dict(color=colors[i % len(colors)], width=width, dash=dash),
    ))

fig.add_hline(y=100, line_dash="dash", line_color="#555")
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    title=f"{selected} vs {sector} Peers — Indexed Performance",
    xaxis=dict(gridcolor="#2a2a3a"),
    yaxis=dict(gridcolor="#2a2a3a", ticksuffix=""),
    height=400,
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------#
# Return comparison table
# ---------------------------------------------------------------------------#

st.subheader("Return Comparison")

returns_data = []
for t in peer_prices.columns:
    s = peer_prices[t].dropna()
    if len(s) < 2:
        continue
    total  = (s.iloc[-1] / s.iloc[0] - 1) * 100
    w1     = (s.iloc[-1] / s.iloc[min(-5,  -len(s))] - 1) * 100
    m1     = (s.iloc[-1] / s.iloc[min(-21, -len(s))] - 1) * 100
    m3     = (s.iloc[-1] / s.iloc[min(-63, -len(s))] - 1) * 100
    returns_data.append({
        "Ticker":    t,
        f"{lookback}d Return (%)": round(total, 2),
        "1W (%)":    round(w1, 2),
        "1M (%)":    round(m1, 2),
        "3M (%)":    round(m3, 2),
        "Primary":   "⭐" if t == selected else "",
    })

ret_df = pd.DataFrame(returns_data).sort_values(f"{lookback}d Return (%)", ascending=False)
st.dataframe(style_generic(ret_df), use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------#
# Signal comparison (ranking scores)
# ---------------------------------------------------------------------------#

st.subheader("Signal Comparison")

avail_peers = [t for t in peers if (settings.raw_daily_dir / f"{t}.parquet").exists()]

if avail_peers:
    @st.cache_data(ttl=600, show_spinner="Computing peer signals …")
    def _peer_signals(tickers: tuple) -> pd.DataFrame:
        from src.ranking.ranker import rank_tickers, to_dataframe
        entries = rank_tickers(list(tickers), include_ml=False, include_indicators=True)
        return to_dataframe(entries)

    peer_sig_df = _peer_signals(tuple(avail_peers))

    if not peer_sig_df.empty:
        score_cols = [c for c in ["Rank", "Ticker", "Score", "Indicators",
                                   "Forecast", "News", "Options"] if c in peer_sig_df.columns]
        st.dataframe(style_ranking_table(peer_sig_df[score_cols]),
                     use_container_width=True)
else:
    st.info("Score comparison requires scraped data for peer tickers.")
