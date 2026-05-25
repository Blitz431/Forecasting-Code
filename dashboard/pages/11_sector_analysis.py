"""Page 9 — Sector Analysis.

Sector breakdown of ranked picks, concentration view, and rotation indicators.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import sector_donut, ranking_bar, _empty_fig
from dashboard.components.tables import style_ranking_table, style_generic

st.set_page_config(page_title="Sector Analysis", page_icon="🗂️", layout="wide")
st.title("🗂️ Sector Analysis")
st.caption("Sector breakdown of top picks, concentration warnings, and rotation view.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Load top picks ranking
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=900, show_spinner="Computing rankings …")
def _load_top_picks(n: int = 30):
    from src.ranking.ranker import top_picks, to_dataframe
    picks = top_picks(n=n, include_ml=False, include_indicators=True)
    return to_dataframe(picks), picks

with st.sidebar:
    top_n = st.slider("Tickers to analyse", 10, 100, value=30)

# Check data
if not settings.raw_daily_dir.exists() or not any(settings.raw_daily_dir.glob("*.parquet")):
    st.warning("No price data found. Run `python cli/scrape.py --backfill 2015` first.")
    st.stop()

with st.spinner("Loading rankings …"):
    try:
        df, picks = _load_top_picks(top_n)
    except Exception as e:
        st.error(f"Failed to load rankings: {e}")
        st.stop()

if df.empty:
    st.warning("No picks available.")
    st.stop()

# ---------------------------------------------------------------------------#
# Fetch sector info via yfinance
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=3600, show_spinner="Fetching sector info …")
def _get_sectors(tickers: tuple) -> pd.DataFrame:
    """Fetch sector/industry data for a list of tickers via yfinance."""
    import yfinance as yf
    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            rows.append({
                "Ticker":   ticker,
                "Sector":   info.get("sector",   "Unknown"),
                "Industry": info.get("industry", "Unknown"),
            })
        except Exception:
            rows.append({"Ticker": ticker, "Sector": "Unknown", "Industry": "Unknown"})
    return pd.DataFrame(rows)

sector_df = _get_sectors(tuple(df["Ticker"].tolist()))

# Merge with ranking df
merged = df.merge(sector_df, on="Ticker", how="left")
merged["Sector"] = merged["Sector"].fillna("Unknown")

# ---------------------------------------------------------------------------#
# Sector summary
# ---------------------------------------------------------------------------#

sector_counts = merged["Sector"].value_counts()
sector_scores = merged.groupby("Sector")["Score"].mean().sort_values(ascending=False)

col1, col2, col3 = st.columns(3)
col1.metric("Sectors Represented",   sector_counts.nunique())
col2.metric("Top Sector (by count)", sector_counts.index[0] if not sector_counts.empty else "—")
col3.metric("Top Sector (by score)", sector_scores.index[0] if not sector_scores.empty else "—")

st.divider()

# ---------------------------------------------------------------------------#
# Charts
# ---------------------------------------------------------------------------#

tab_donut, tab_bar, tab_table = st.tabs(["🍩 Sector Donut", "📊 Sector Scores", "📋 Full Table"])

with tab_donut:
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(sector_donut(sector_counts), use_container_width=True)
        st.caption("Count of top picks per sector")
    with col_r:
        st.plotly_chart(sector_donut(sector_scores.clip(lower=0)),
                        use_container_width=True)
        st.caption("Average composite score per sector (positive picks only)")

with tab_bar:
    # Picks grouped by sector
    import plotly.graph_objects as go
    import plotly.express as px

    fig = px.bar(
        merged.sort_values("Score", ascending=False),
        x="Ticker", y="Score",
        color="Sector",
        title="Top Picks Coloured by Sector",
        template="plotly_dark",
        height=400,
    )
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(fig, use_container_width=True)

with tab_table:
    disp_cols = ["Rank", "Ticker", "Sector", "Industry", "Score"]
    avail = [c for c in disp_cols if c in merged.columns]
    st.dataframe(style_ranking_table(merged[avail]), use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------#
# Concentration check
# ---------------------------------------------------------------------------#

st.subheader("Concentration Check")

top_sector = sector_counts.index[0] if not sector_counts.empty else None
top_pct    = sector_counts.iloc[0] / len(merged) * 100 if not sector_counts.empty else 0

if top_pct > 40:
    st.warning(
        f"⚠️ Concentration risk: **{top_pct:.0f}%** of top picks are in **{top_sector}**. "
        "Consider diversifying sector exposure."
    )
else:
    st.success(f"Sector concentration looks healthy — top sector ({top_sector}) is {top_pct:.0f}% of picks.")

st.dataframe(
    pd.DataFrame({
        "Sector":  sector_counts.index,
        "Count":   sector_counts.values,
        "% of Picks": (sector_counts.values / len(merged) * 100).round(1),
        "Avg Score": [round(sector_scores.get(s, 0), 3) for s in sector_counts.index],
    }),
    use_container_width=True,
    hide_index=True,
)
