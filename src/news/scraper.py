from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

import feedparser
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from config.settings import get_settings
from src.utils.logging import setup_logger
from src.utils.parallel import thread_map

logger = setup_logger(__name__)

"""
Purpose: Fetch news articles for a ticker from Yahoo Finance RSS, Google News RSS, and yfinance fallback.

Connections:
  - src/news/runner.py: calls fetch_ticker_news() per ticker
  - src/news/aggregator.py: receives article dicts and scores them
  - src/utils/logging.py: logger

In:  ticker symbol string
Out: list[dict] with headline, url, summary, source, published — deduplicated by URL
"""

_YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"

# Minimum articles before falling back to yfinance.news
_RSS_FALLBACK_THRESHOLD = 5

_REQUEST_TIMEOUT = 10
_USER_AGENT = (
    "Mozilla/5.0 (compatible; AutoStockAnalyzer/1.0; +https://github.com/)"
)

# Per-thread requests.Session — requests.Session isn't documented
# thread-safe, so each worker thread gets its own for connection pooling
# without cross-thread state sharing.
_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Return the current thread's requests.Session, creating it on first use."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": _USER_AGENT})
        _thread_local.session = session
    return session


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


def _fetch_feed(url: str, source: str) -> list[dict]:
    """Fetch and parse a single RSS feed.

    Dedup across feeds is NOT done here — it happens in the caller
    (fetch_ticker_news) after all feeds have returned, since mutating a
    shared set from pool threads isn't safe.
    """
    articles: list[dict] = []
    try:
        response = _get_session().get(url, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        for entry in feed.entries:
            article = _parse_entry(entry, source)
            if article:
                articles.append(article)
    except Exception as exc:
        logger.warning(f"RSS fetch failed for {source} ({url}): {exc}")
    return articles


def _fetch_feed_unit(spec: tuple[str, str]) -> list[dict]:
    """thread_map-friendly wrapper: (url, source) -> articles."""
    url, source = spec
    return _fetch_feed(url, source)


def _parse_yfinance_item(item: dict) -> tuple[str, str, str, int]:
    """Extract (headline, url, publisher, timestamp) from a yfinance news item.

    Handles both the old flat format (yfinance < 0.2.50) and the new nested
    content format (yfinance >= 0.2.50).
    """
    # New nested format: {"id": ..., "content": {"title": ..., "canonicalUrl": {"url": ...}, ...}}
    content = item.get("content", {})
    if content:
        headline = content.get("title", "").strip()
        url = (content.get("canonicalUrl") or {}).get("url", "").strip()
        if not url:
            url = (content.get("clickThroughUrl") or {}).get("url", "").strip()
        publisher = (content.get("provider") or {}).get("displayName", "yfinance")
        pub_date = content.get("pubDate", "")
        try:
            ts = int(datetime.fromisoformat(pub_date.replace("Z", "+00:00")).timestamp()) if pub_date else 0
        except Exception:
            ts = 0
    else:
        # Old flat format
        headline = item.get("title", "").strip()
        url = item.get("link", "").strip()
        publisher = item.get("publisher", "yfinance")
        ts = item.get("providerPublishTime", 0)

    return headline, url, publisher, ts


def _fetch_yfinance_news(ticker: str, seen_urls: set[str]) -> list[dict]:
    """Fallback: pull articles from yfinance Ticker.news."""
    articles: list[dict] = []
    try:
        raw = yf.Ticker(ticker).news or []
        for item in raw:
            headline, url, publisher, ts = _parse_yfinance_item(item)
            if not headline or not url or url in seen_urls:
                continue
            published = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(tz=timezone.utc)
            articles.append({
                "published": published,
                "headline": headline,
                "summary": "",
                "url": url,
                "source": publisher,
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
    feeds = [
        (_YAHOO_RSS.format(ticker=ticker), "Yahoo Finance"),
        (_GOOGLE_NEWS_RSS.format(ticker=ticker), "Google News"),
    ]

    # Fetch both feeds concurrently. thread_map preserves input order, so
    # feed_results[0] is Yahoo and feed_results[1] is Google.
    feed_results = thread_map(_fetch_feed_unit, feeds, max_workers=2, label="news-feed")

    # Dedup by URL AFTER both feeds return — Yahoo first, then Google, so
    # a Yahoo article wins over a Google duplicate.
    seen_urls: set[str] = set()
    articles: list[dict] = []
    for feed_articles in feed_results:
        for article in feed_articles or []:
            if article["url"] not in seen_urls:
                seen_urls.add(article["url"])
                articles.append(article)

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
    max_workers: int | None = None,
) -> dict[str, list[dict]]:
    """Fetch news for multiple tickers concurrently on a bounded thread pool.

    Args:
        tickers: List of ticker symbols.
        max_articles: Max articles per ticker.
        delay: NOTE — semantic change: this used to be a hard per-ticker
            serialization delay (time.sleep(delay) between each ticker,
            processed one at a time). It is now the stagger (seconds)
            between task *submissions* on the bounded pool, which only
            bounds the average submission rate (~max_workers / delay per
            second) and no longer guarantees strict serialization. If a
            caller needs the old hard-serialization guarantee, pass
            max_workers=1 instead of relying on `delay`.
        max_workers: Concurrent ticker fetches. Defaults to
            settings.news_max_workers when None.

    Returns:
        Dict mapping ticker -> list of article dicts.
    """
    if max_workers is None:
        max_workers = get_settings().news_max_workers

    results = thread_map(
        lambda t: fetch_ticker_news(t, max_articles),
        tickers,
        max_workers=max_workers,
        label="news-batch",
        stagger=delay / max(max_workers, 1),
    )
    return {ticker: (articles or []) for ticker, articles in zip(tickers, results)}
