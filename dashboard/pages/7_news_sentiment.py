"""Page 6 — News & Sentiment.

Recent headlines with FinBERT sentiment scores, sentiment trend chart,
and short interest data.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import sentiment_timeline, _empty_fig
from dashboard.components.tables import style_sentiment_table, style_generic
from dashboard.components.ticker_selector import render_ticker_sidebar

st.set_page_config(page_title="News & Sentiment", page_icon="📰", layout="wide")
st.title("📰 News & Sentiment")
st.caption("FinBERT-scored headlines, 7-day rolling sentiment, and short interest data.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Ticker selector (global — persists across pages)
# ---------------------------------------------------------------------------#

selected = render_ticker_sidebar()
if not selected:
    st.warning("No data found. Run `python cli/news.py --premarket` first.")
    st.stop()

from dashboard.components.session_cache import format_freshness
st.caption(f"📅 News last fetched: {format_freshness(settings.news_articles_dir / f'{selected}.parquet')}")

window_days = st.slider("Sentiment window (days)", 1, 30, value=7)

# ---------------------------------------------------------------------------#
# Load & display sentiment
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300)
def _load_articles(ticker: str) -> pd.DataFrame:
    fp = settings.news_articles_dir / f"{ticker}.parquet"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_parquet(fp)

@st.cache_data(ttl=300)
def _sentiment_summary(ticker: str, days: int) -> dict:
    from src.news.aggregator import get_sentiment_summary
    return get_sentiment_summary(ticker, settings.news_articles_dir, days=days)

@st.cache_data(ttl=300)
def _daily_sentiment(ticker: str, days: int) -> pd.DataFrame:
    from src.news.aggregator import get_daily_sentiment
    return get_daily_sentiment(ticker, settings.news_articles_dir, days=days)

articles = _load_articles(selected)
summary  = _sentiment_summary(selected, window_days)
daily_s  = _daily_sentiment(selected, window_days)

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sentiment",     summary.get("sentiment", "neutral").title())
col2.metric("Avg Score",     f"{summary.get('avg_score', 0.0):+.4f}")
col3.metric("Articles (window)", summary.get("article_count", 0))
col4.metric("Total articles", len(articles))

st.divider()

# Sentiment timeline
st.subheader("Daily Sentiment Trend")
if not daily_s.empty:
    st.plotly_chart(sentiment_timeline(daily_s, selected), use_container_width=True)
else:
    st.info("No daily sentiment data available for the selected window.")

# Fetch fresh news button
st.divider()
if st.button("🔄 Fetch Latest News"):
    with st.spinner(f"Fetching news for {selected} …"):
        try:
            from src.news.runner import run_news_pipeline
            result = run_news_pipeline(
                [selected],
                max_articles=settings.news_max_articles,
                sentiment_window_days=window_days,
            )
            st.success(f"Done — sentiment: {result.get(selected, {}).get('sentiment', '?')}")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"News fetch failed: {e}")

# ---------------------------------------------------------------------------#
# Recent articles table
# ---------------------------------------------------------------------------#

st.subheader("Recent Articles")

if articles.empty:
    st.info(
        f"No articles for **{selected}**.\n\n"
        "Run `python cli/news.py --premarket` to fetch news."
    )
else:
    # Show most recent articles
    disp_cols = [c for c in ["headline", "source", "sentiment_label", "sentiment_score",
                              "confidence", "summary"] if c in articles.columns]
    recent = articles.sort_index(ascending=False).head(50)
    if disp_cols:
        st.dataframe(style_sentiment_table(recent[disp_cols].reset_index(names=["Published"])),
                     use_container_width=True, height=400)
    else:
        st.dataframe(style_generic(recent.reset_index()), use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------#
# Short interest
# ---------------------------------------------------------------------------#

st.subheader("Short Interest")

@st.cache_data(ttl=300)
def _load_short_interest(ticker: str) -> dict:
    from src.news.short_interest import get_short_interest_signal
    return get_short_interest_signal(ticker, settings.news_short_interest_dir)

si = _load_short_interest(selected)
ratio = si.get("short_ratio")
pct   = si.get("short_pct_float")
high  = si.get("high_short_interest", False)

col1, col2, col3 = st.columns(3)
col1.metric("Short Ratio (days)",  f"{ratio:.1f}" if ratio else "—")
col2.metric("Short % of Float",    f"{pct*100:.1f}%" if pct else "—")
col3.metric("High Short Interest", "⚠️ YES" if high else "No")

if high:
    st.warning(f"⚠️ {selected} has high short interest ({ratio:.1f} days to cover). "
               "This may indicate bearish sentiment or potential short-squeeze risk.")
