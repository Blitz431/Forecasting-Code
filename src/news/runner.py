from __future__ import annotations

import time
from pathlib import Path

from config.settings import get_settings
from src.news.aggregator import build_news_dataframe, save_news, get_sentiment_summary
from src.news.scraper import fetch_ticker_news
from src.news.sentiment import FinBERTScorer
from src.news.short_interest import (
    fetch_short_interest,
    save_short_interest,
    get_short_interest_signal,
)
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Orchestrate the full Phase 5 pipeline — fetch news, score with FinBERT, store articles and short interest.

Connections:
  - src/news/scraper.py: fetch_ticker_news() per ticker
  - src/news/sentiment.py: FinBERTScorer (singleton, cached to avoid reloading model)
  - src/news/aggregator.py: build_news_dataframe(), save_news(), get_sentiment_summary()
  - src/news/short_interest.py: fetch/save/get_short_interest_signal()
  - config/settings.py: articles dir, short interest dir, max articles, sentiment window
  - cli/news.py: calls run_news_pipeline(), run_sentiment_only(), run_short_interest_only()

In:  list of ticker symbols
Out: dict[ticker, result] with sentiment (label), avg_score, article_count, short_ratio, high_short_interest
"""

# Module-level scorer cache — avoids reloading model weights on repeated calls
_scorer: FinBERTScorer | None = None


def _get_scorer(model_name: str) -> FinBERTScorer:
    global _scorer
    if _scorer is None:
        _scorer = FinBERTScorer(model_name)
    return _scorer


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #

def run_news_pipeline(
    tickers: list[str],
    articles_dir: Path | None = None,
    short_interest_dir: Path | None = None,
    max_articles: int | None = None,
    sentiment_window_days: int = 7,
    request_delay: float = 1.0,
) -> dict[str, dict]:
    """Run the full Phase 5 pipeline for a list of tickers.

    For each ticker:
    - Fetches news from RSS feeds
    - Scores sentiment with FinBERT
    - Stores articles + scores to Parquet
    - Fetches current short interest
    - Stores short interest to Parquet

    Args:
        tickers: List of ticker symbols.
        articles_dir: Where to store article Parquet files.
                      Defaults to settings.news_articles_dir.
        short_interest_dir: Where to store short interest Parquet files.
                            Defaults to settings.news_short_interest_dir.
        max_articles: Max articles to fetch per ticker.
                      Defaults to settings.news_max_articles.
        sentiment_window_days: Rolling window for sentiment summary.
        request_delay: Seconds to sleep between tickers (rate limiting).

    Returns:
        Dict mapping ticker -> result dict with keys:
        - sentiment (str): "positive" / "negative" / "neutral"
        - avg_score (float)
        - article_count (int)
        - short_ratio (float | None)
        - high_short_interest (bool)
        - error (str | None): set only when the ticker failed
    """
    settings = get_settings()
    articles_dir = articles_dir or settings.news_articles_dir
    short_interest_dir = short_interest_dir or settings.news_short_interest_dir
    max_articles = max_articles or settings.news_max_articles

    articles_dir.mkdir(parents=True, exist_ok=True)
    short_interest_dir.mkdir(parents=True, exist_ok=True)

    scorer = _get_scorer(settings.finbert_model)
    results: dict[str, dict] = {}

    for i, ticker in enumerate(tickers):
        try:
            # --- News & Sentiment ---
            logger.info(f"[{ticker}] Fetching news ({i + 1}/{len(tickers)})")
            articles = fetch_ticker_news(ticker, max_articles=max_articles)

            if articles:
                texts = [
                    f"{a['headline']}. {a['summary']}".strip() if a.get("summary") else a["headline"]
                    for a in articles
                ]
                scored = scorer.score(texts)
                df = build_news_dataframe(articles, scored)
                save_news(df, ticker, articles_dir)
            else:
                logger.warning(f"[{ticker}] No articles found")

            sentiment = get_sentiment_summary(ticker, articles_dir, days=sentiment_window_days)

            # --- Short Interest ---
            si_record = fetch_short_interest(ticker)
            save_short_interest([si_record], short_interest_dir)
            si_signal = get_short_interest_signal(ticker, short_interest_dir)

            results[ticker] = {
                **sentiment,
                "short_ratio": si_signal["short_ratio"],
                "short_pct_float": si_signal["short_pct_float"],
                "high_short_interest": si_signal["high_short_interest"],
                "error": None,
            }

            logger.info(
                f"[{ticker}] sentiment={sentiment['sentiment']} "
                f"score={sentiment['avg_score']} "
                f"articles={sentiment['article_count']} "
                f"short_ratio={si_signal['short_ratio']}"
            )

        except Exception as exc:
            logger.error(f"[{ticker}] Pipeline failed: {exc}")
            results[ticker] = {"ticker": ticker, "error": str(exc)}

        if request_delay > 0 and i < len(tickers) - 1:
            time.sleep(request_delay)

    logger.info(f"Phase 5 pipeline complete: {len(results)} tickers processed")
    return results


# --------------------------------------------------------------------------- #
# Partial runners
# --------------------------------------------------------------------------- #

def run_sentiment_only(
    tickers: list[str],
    articles_dir: Path | None = None,
    max_articles: int | None = None,
    sentiment_window_days: int = 7,
    request_delay: float = 1.0,
) -> dict[str, dict]:
    """Run news fetch + sentiment scoring only (skip short interest).

    Useful for mid-day refreshes or when short interest data is already fresh.
    """
    settings = get_settings()
    articles_dir = articles_dir or settings.news_articles_dir
    max_articles = max_articles or settings.news_max_articles

    articles_dir.mkdir(parents=True, exist_ok=True)
    scorer = _get_scorer(settings.finbert_model)
    results: dict[str, dict] = {}

    for i, ticker in enumerate(tickers):
        try:
            articles = fetch_ticker_news(ticker, max_articles=max_articles)
            if articles:
                texts = [
                    f"{a['headline']}. {a['summary']}".strip() if a.get("summary") else a["headline"]
                    for a in articles
                ]
                scored = scorer.score(texts)
                df = build_news_dataframe(articles, scored)
                save_news(df, ticker, articles_dir)

            results[ticker] = get_sentiment_summary(ticker, articles_dir, days=sentiment_window_days)

        except Exception as exc:
            logger.error(f"[{ticker}] Sentiment pipeline failed: {exc}")
            results[ticker] = {"ticker": ticker, "error": str(exc)}

        if request_delay > 0 and i < len(tickers) - 1:
            time.sleep(request_delay)

    return results


def run_short_interest_only(
    tickers: list[str],
    short_interest_dir: Path | None = None,
) -> dict[str, dict]:
    """Fetch and persist short interest data only (skip news/sentiment).

    Args:
        tickers: List of ticker symbols.
        short_interest_dir: Storage directory.

    Returns:
        Dict mapping ticker -> short interest signal dict.
    """
    settings = get_settings()
    short_interest_dir = short_interest_dir or settings.news_short_interest_dir
    short_interest_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    for i, ticker in enumerate(tickers):
        try:
            record = fetch_short_interest(ticker)
            save_short_interest([record], short_interest_dir)
            results[ticker] = get_short_interest_signal(ticker, short_interest_dir)
            logger.info(
                f"[{ticker}] short_ratio={results[ticker]['short_ratio']} "
                f"high={results[ticker]['high_short_interest']}"
            )
        except Exception as exc:
            logger.error(f"[{ticker}] Short interest fetch failed: {exc}")
            results[ticker] = {"ticker": ticker, "error": str(exc)}

    return results
