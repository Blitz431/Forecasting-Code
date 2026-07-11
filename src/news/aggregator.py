from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.scraper.storage import upsert_dataframe, load_dataframe
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Combine scraped articles with FinBERT scores into per-ticker parquets and compute rolling sentiment summaries.

Connections:
  - src/scraper/storage.py: upsert_dataframe(), load_dataframe() for parquet persistence
  - src/news/runner.py: calls build_news_dataframe(), save_news(), get_sentiment_summary()
  - src/ranking/ranker.py: reads sentiment summaries as one ranking signal
  - src/alerts/triggers.py: reads sentiment to detect sentiment flips

In:  article dicts + FinBERT scored dicts from scraper.py / sentiment.py
Out: data/news/articles/{ticker}.parquet (headline, sentiment_label, sentiment_score, confidence)
"""


# --------------------------------------------------------------------------- #
# Build + persist
# --------------------------------------------------------------------------- #

def build_news_dataframe(articles: list[dict], scored: list[dict]) -> pd.DataFrame:
    """Combine raw articles and sentiment scores into a DataFrame.

    Args:
        articles: Output of :func:`src.news.scraper.fetch_ticker_news`.
        scored: Output of :meth:`src.news.sentiment.FinBERTScorer.score`,
                same length and order as ``articles``.

    Returns:
        DataFrame indexed by UTC publish timestamp with columns:
        headline, summary, url, source, sentiment_label, sentiment_score,
        confidence.  Empty DataFrame if inputs are empty.
    """
    if not articles or not scored:
        return pd.DataFrame()

    rows = []
    for article, sentiment in zip(articles, scored):
        rows.append({
            "published": article["published"],
            "headline": article["headline"],
            "summary": article.get("summary", ""),
            "url": article["url"],
            "source": article["source"],
            "sentiment_label": sentiment["label"],
            "sentiment_score": sentiment["score"],
            "confidence": sentiment["confidence"],
        })

    df = pd.DataFrame(rows)
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df = df.set_index("published").sort_index(ascending=False)
    return df


def save_news(df: pd.DataFrame, ticker: str, articles_dir: Path) -> None:
    """Upsert a news DataFrame for a ticker into Parquet storage.

    Args:
        df: DataFrame returned by :func:`build_news_dataframe`.
        ticker: Stock ticker symbol.
        articles_dir: Base directory (e.g. data/news/articles/).
    """
    if df.empty:
        return
    filepath = articles_dir / f"{ticker}.parquet"
    upsert_dataframe(df, filepath)


def load_news(ticker: str, articles_dir: Path) -> pd.DataFrame:
    """Load stored news for a ticker.

    Args:
        ticker: Stock ticker symbol.
        articles_dir: Directory containing article Parquet files.

    Returns:
        DataFrame with UTC timestamp index, or empty DataFrame if none stored.
    """
    filepath = articles_dir / f"{ticker}.parquet"
    return load_dataframe(filepath)


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #

def get_daily_sentiment(ticker: str, articles_dir: Path, days: int = 30) -> pd.DataFrame:
    """Compute daily aggregate sentiment for a ticker.

    Args:
        ticker: Stock ticker symbol.
        articles_dir: Directory containing article Parquet files.
        days: How many calendar days back to include.

    Returns:
        DataFrame with date index (newest first) and columns:
        - avg_score       : mean FinBERT sentiment score (-1 to +1)
        - article_count   : number of articles that day
        - positive_pct    : fraction labelled "positive"
        - negative_pct    : fraction labelled "negative"
        Empty DataFrame when no data is available.
    """
    df = load_news(ticker, articles_dir)
    if df.empty:
        return pd.DataFrame()

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    df = df[df.index >= cutoff].copy()
    if df.empty:
        return pd.DataFrame()

    df["date"] = df.index.normalize()  # UTC date

    daily = (
        df.groupby("date")
        .agg(
            avg_score=("sentiment_score", "mean"),
            article_count=("sentiment_score", "count"),
            positive_pct=("sentiment_label", lambda x: (x == "positive").mean()),
            negative_pct=("sentiment_label", lambda x: (x == "negative").mean()),
        )
        .sort_index(ascending=False)
    )
    daily.index = pd.to_datetime(daily.index)
    return daily


def get_sentiment_summary(ticker: str, articles_dir: Path, days: int = 7) -> dict:
    """Return a high-level sentiment summary for the last N days.

    Used by the runner, ranker, and dashboard to get a quick read on a
    ticker's news tone without loading the full article history.

    Args:
        ticker: Stock ticker symbol.
        articles_dir: Directory containing article Parquet files.
        days: Rolling window in calendar days.

    Returns:
        Dict with keys: ticker, avg_score, article_count, sentiment
        ("positive" / "negative" / "neutral"), days.
    """
    daily = get_daily_sentiment(ticker, articles_dir, days=days)

    if daily.empty:
        return {
            "ticker": ticker,
            "avg_score": 0.0,
            "article_count": 0,
            "sentiment": "neutral",
            "days": days,
        }

    avg = float(daily["avg_score"].mean())
    count = int(daily["article_count"].sum())

    if avg > 0.1:
        label = "positive"
    elif avg < -0.1:
        label = "negative"
    else:
        label = "neutral"

    return {
        "ticker": ticker,
        "avg_score": round(avg, 4),
        "article_count": count,
        "sentiment": label,
        "days": days,
    }


def build_sentiment_feature(tickers: list[str], articles_dir: Path, days: int = 7) -> pd.DataFrame:
    """Build a sentiment feature row for each ticker for use in the ML pipeline.

    Args:
        tickers: List of ticker symbols.
        articles_dir: Directory containing article Parquet files.
        days: Rolling window passed to :func:`get_sentiment_summary`.

    Returns:
        DataFrame indexed by ticker with columns:
        sentiment_score, sentiment_article_count, sentiment_label_encoded
        (positive=1, neutral=0, negative=-1).
    """
    rows = []
    for ticker in tickers:
        summary = get_sentiment_summary(ticker, articles_dir, days=days)
        label_enc = {"positive": 1, "neutral": 0, "negative": -1}.get(summary["sentiment"], 0)
        rows.append({
            "ticker": ticker,
            "sentiment_score": summary["avg_score"],
            "sentiment_article_count": summary["article_count"],
            "sentiment_label_encoded": label_enc,
        })

    df = pd.DataFrame(rows).set_index("ticker")
    return df
