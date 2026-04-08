"""AutoStockAnalyzer — Streamlit Dashboard entry point.

Run with:
    streamlit run dashboard/app.py

This is the landing page (home / overview).  All 17 feature pages live in
``dashboard/pages/`` and are automatically listed in the sidebar by Streamlit.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when running as `streamlit run dashboard/app.py`
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(
    page_title="AutoStockAnalyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------#
# Header
# ---------------------------------------------------------------------------#

st.title("📈 AutoStockAnalyzer")
st.markdown(
    """
    **Automated stock analysis, forecasting, and ranking for the S&P 500.**
    Use the sidebar to navigate between modules.
    """,
    unsafe_allow_html=False,
)
st.divider()

# ---------------------------------------------------------------------------#
# Quick status cards
# ---------------------------------------------------------------------------#

from config.settings import get_settings
settings = get_settings()

col1, col2, col3, col4 = st.columns(4)

def _count_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.parquet")))

with col1:
    n_daily = _count_files(settings.raw_daily_dir)
    st.metric("Daily Price Files", f"{n_daily:,}", help="Tickers scraped to data/raw/daily/")

with col2:
    n_forecast = _count_files(settings.forecasts_dir)
    st.metric("Forecast Files", f"{n_forecast:,}", help="Tickers with forecasts in data/forecasts/")

with col3:
    n_ml = _count_files(settings.data_dir / "ml" / "results")
    st.metric("ML Result Files", f"{n_ml:,}", help="Tickers with trained ML models")

with col4:
    n_news = _count_files(settings.news_articles_dir)
    st.metric("News Article Files", f"{n_news:,}", help="Tickers with news sentiment data")

st.divider()

# ---------------------------------------------------------------------------#
# Top picks quick view
# ---------------------------------------------------------------------------#

st.subheader("⚡ Quick Top 10 (Fast Ranking)")

@st.cache_data(ttl=1800, show_spinner="Computing fast rankings …")
def _fast_top10():
    from src.ranking.ranker import top_picks, to_dataframe
    picks = top_picks(n=10, include_ml=False, include_indicators=True)
    return to_dataframe(picks)

if st.button("Run Fast Ranking", type="primary"):
    with st.spinner("Ranking stocks …"):
        df = _fast_top10()
    if df.empty:
        st.warning(
            "No picks found. Ensure you have scraped data first:\n\n"
            "`python cli/scrape.py --backfill 2015`"
        )
    else:
        from dashboard.components.charts import ranking_bar
        from dashboard.components.tables import style_ranking_table
        st.plotly_chart(ranking_bar(df, top_n=10), use_container_width=True)
        st.dataframe(style_ranking_table(df), use_container_width=True)
else:
    st.info("Click **Run Fast Ranking** to compute composite scores for all tickers.")

st.divider()

# ---------------------------------------------------------------------------#
# Navigation guide
# ---------------------------------------------------------------------------#

st.subheader("🗺️ Dashboard Pages")

pages = [
    ("📊", "1 — Data Overview",         "Browse available ticker data, date ranges, and data quality stats."),
    ("🔭", "2 — Long-Term Forecast",     "All 12 forecast methods per ticker with RMSE comparison and overlay chart."),
    ("📡", "3 — Short-Term Signals",     "9 technical indicators, composite signal, and heatmap view."),
    ("🤖", "4 — ML Predictions",         "XGBoost / LightGBM / LSTM / Transformer results with feature importance."),
    ("🏆", "5 — Stock Rankings",         "Full top-20 composite ranking with per-signal breakdown."),
    ("📰", "6 — News & Sentiment",       "Recent headlines, FinBERT sentiment scores, and short interest."),
    ("🏛️", "7 — Political & Insider",   "Congressional STOCK Act trades and SEC Form 4 insider filings."),
    ("🎯", "8 — Options Flow",           "Put/Call ratio, unusual volume, implied vs historical volatility."),
    ("🗂️", "9 — Sector Analysis",       "Sector rotation view, concentration, and top picks by sector."),
    ("💼", "10 — Portfolio",             "Portfolio analytics, correlation matrix, and attribution (Phase 9)."),
    ("⏮️", "11 — Backtesting",          "Strategy simulation 2020-2026 with equity curve (Phase 9)."),
    ("📅", "12 — Calendar",             "Upcoming earnings dates and economic events (FOMC, CPI, jobs)."),
    ("🌡️", "13 — Market Regime",        "Bull/Bear/Sideways detection, VIX dashboard, breadth indicators."),
    ("👁️", "14 — Watchlist",            "Custom watchlist with multi-timeframe analysis."),
    ("📓", "15 — Trade Journal",         "Trade history, audit log, and tax lot view (Phase 10)."),
    ("⚖️", "16 — Peer Comparison",      "Stock vs sector peers relative strength and ranking."),
    ("🔧", "17 — Trading",              "Live/paper trading controls and circuit breaker (Phase 10)."),
]

for icon, name, desc in pages:
    st.markdown(f"**{icon} {name}** — {desc}")

st.divider()
st.caption("AutoStockAnalyzer — Phase 8 Dashboard | Built with Streamlit + Plotly")
