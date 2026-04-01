"""Phase 5: News scraper using RSS feeds and yfinance fallback.

Fetches headlines and summaries for a given ticker from:
  1. Yahoo Finance RSS
  2. Google News RSS
  3. yfinance .news (fallback when RSS yields too few articles)

All articles are normalized to the same dict schema and deduplicated by URL.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import feedparser
import yfinance as yf
from bs4 import BeautifulSoup

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

_YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"

# Minimum articles before falling back to yfinance.news
_RSS_FALLBACK_THRESHOLD = 5


def _parse_entry(entry: Any, source: str) -> dict | None:
    """Normalize a feedparser entry into a standard article dict."""
    try:
        headline = entry.get("title", "").strip()
        url = entry.get("link", "").strip()
        summary_raw = entry.get("summary", "") or ""

        parsed = entry.get("published_parsed")
        if parsed:
            published = datetime(*parsed[:6], tzinfo=timezone.utc)
        else:
            published = datetime.now(tz=timezone.utc)

        if not headline or not url:
            return None

        # Strip HTML tags from summary
        summary = BeautifulSoup(summary_raw, "html.parser").get_text()[:500] if summary_raw else ""

        return {
            "published": published,
            "headline": headline,
            "summary": summary.strip(),
            "url": url,
            "source": source,
        }
    except Exception as exc:
        logger.debug(f"Failed to parse feed entry: {exc}")
        return None


def _fetch_feed(url: str, source: str, seen_urls: set[str]) -> list[dict]:
    """Fetch and parse a single RSS feed, skipping already-seen URLs."""
    articles: list[dict] = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            article = _parse_entry(entry, source)
            if article and article["url"] not in seen_urls:
                seen_urls.add(article["url"])
                articles.append(article)
    except Exception as exc:
        logger.warning(f"RSS fetch failed for {source} ({url}): {exc}")
    return articles


def _fetch_yfinance_news(ticker: str, seen_urls: set[str]) -> list[dict]:
    """Fallback: pull articles from yfinance Ticker.news."""
    articles: list[dict] = []
    try:
        raw = yf.Ticker(ticker).news or []
        for item in raw:
            url = item.get("link", "").strip()
            headline = item.get("title", "").strip()
            if not headline or not url or url in seen_urls:
                continue
            ts = item.get("providerPublishTime", 0)
            published = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(tz=timezone.utc)
            articles.append({
                "published": published,
                "headline": headline,
                "summary": "",
                "url": url,
                "source": item.get("publisher", "yfinance"),
            })
            seen_urls.add(url)
    except Exception as exc:
        logger.warning(f"[{ticker}] yfinance news fallback failed: {exc}")
    return articles


def fetch_ticker_news(ticker: str, max_articles: int = 50) -> list[dict]:
    """Fetch news articles for a single ticker.

    Tries Yahoo Finance RSS and Google News RSS first. Falls back to
    yfinance.news when RSS returns fewer than ``_RSS_FALLBACK_THRESHOLD``
    articles.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        max_articles: Maximum number of articles to return.

    Returns:
        List of article dicts with keys:
        - published (datetime, UTC-aware)
        - headline (str)
        - summary (str, may be empty)
        - url (str)
        - source (str)
        Sorted newest-first, capped at ``max_articles``.
    """
    seen_urls: set[str] = set()
    articles: list[dict] = []

    feeds = [
        (_YAHOO_RSS.format(ticker=ticker), "Yahoo Finance"),
        (_GOOGLE_NEWS_RSS.format(ticker=ticker), "Google News"),
    ]

    for feed_url, source in feeds:
        articles.extend(_fetch_feed(feed_url, source, seen_urls))

    if len(articles) < _RSS_FALLBACK_THRESHOLD:
        logger.debug(f"[{ticker}] Only {len(articles)} RSS articles — trying yfinance fallback")
        articles.extend(_fetch_yfinance_news(ticker, seen_urls))

    articles.sort(key=lambda a: a["published"], reverse=True)
    logger.info(f"[{ticker}] Fetched {min(len(articles), max_articles)} articles")
    return articles[:max_articles]


def fetch_batch_news(
    tickers: list[str],
    max_articles: int = 50,
    delay: float = 1.0,
) -> dict[str, list[dict]]:
    """Fetch news for multiple tickers with a polite delay between requests.

    Args:
        tickers: List of ticker symbols.
        max_articles: Max articles per ticker.
        delay: Seconds to sleep between tickers.

    Returns:
        Dict mapping ticker -> list of article dicts.
    """
    results: dict[str, list[dict]] = {}
    for i, ticker in enumerate(tickers):
        results[ticker] = fetch_ticker_news(ticker, max_articles=max_articles)
        if delay > 0 and i < len(tickers) - 1:
            time.sleep(delay)
    return results
