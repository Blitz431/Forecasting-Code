"""Phase 5: News, Sentiment & Short Interest."""

from src.news.runner import run_news_pipeline, run_sentiment_only, run_short_interest_only
from src.news.aggregator import (
    load_news,
    get_daily_sentiment,
    get_sentiment_summary,
    build_sentiment_feature,
)
from src.news.scraper import fetch_ticker_news, fetch_batch_news
from src.news.short_interest import (
    fetch_short_interest,
    load_short_interest,
    get_short_interest_signal,
)

__all__ = [
    # Runners
    "run_news_pipeline",
    "run_sentiment_only",
    "run_short_interest_only",
    # Aggregation
    "load_news",
    "get_daily_sentiment",
    "get_sentiment_summary",
    "build_sentiment_feature",
    # Scraping
    "fetch_ticker_news",
    "fetch_batch_news",
    # Short Interest
    "fetch_short_interest",
    "load_short_interest",
    "get_short_interest_signal",
]
