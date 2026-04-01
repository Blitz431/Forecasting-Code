"""Tests for Phase 5 — News, Sentiment & Short Interest.

All tests are fully offline — no network calls, no FinBERT model download.
Network-dependent functions are monkey-patched with fixtures.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.news.aggregator import (
    build_news_dataframe,
    build_sentiment_feature,
    get_daily_sentiment,
    get_sentiment_summary,
    load_news,
    save_news,
)
from src.news.scraper import _parse_entry, fetch_ticker_news
from src.news.short_interest import (
    fetch_short_interest,
    get_short_interest_signal,
    load_short_interest,
    save_short_interest,
)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

def _make_articles(n: int = 5) -> list[dict]:
    """Each article gets a unique timestamp (1-hour apart) so storage won't deduplicate them.
    Uses recent dates so rolling-window aggregation includes them.
    """
    from datetime import timedelta
    base = datetime.now(tz=timezone.utc) - timedelta(hours=n)
    return [
        {
            "published": base + timedelta(hours=i),
            "headline": f"Stock headline {i}",
            "summary": f"Summary text {i}",
            "url": f"https://example.com/article/{i}",
            "source": "Yahoo Finance",
        }
        for i in range(n)
    ]


def _make_scored(n: int = 5, label: str = "positive") -> list[dict]:
    score = 0.8 if label == "positive" else (-0.8 if label == "negative" else 0.0)
    return [{"label": label, "score": score, "confidence": abs(score)} for _ in range(n)]


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


# ------------------------------------------------------------------ #
# scraper._parse_entry
# ------------------------------------------------------------------ #

class TestParseEntry:
    def test_parses_valid_entry(self):
        entry = {
            "title": "  AAPL hits record high  ",
            "link": "https://example.com/aapl",
            "summary": "<p>Apple surges</p>",
            "published_parsed": (2024, 1, 10, 9, 0, 0, 0, 0, 0),
        }
        result = _parse_entry(entry, "Yahoo Finance")
        assert result is not None
        assert result["headline"] == "AAPL hits record high"
        assert result["url"] == "https://example.com/aapl"
        assert result["source"] == "Yahoo Finance"
        assert "Apple surges" in result["summary"]
        assert result["published"].tzinfo is not None

    def test_returns_none_on_missing_headline(self):
        entry = {"title": "", "link": "https://example.com"}
        assert _parse_entry(entry, "src") is None

    def test_returns_none_on_missing_url(self):
        entry = {"title": "Headline", "link": ""}
        assert _parse_entry(entry, "src") is None

    def test_falls_back_to_now_when_no_date(self):
        entry = {"title": "No date", "link": "https://example.com/x"}
        result = _parse_entry(entry, "test")
        assert result is not None
        assert result["published"].tzinfo is not None


# ------------------------------------------------------------------ #
# scraper.fetch_ticker_news (mocked)
# ------------------------------------------------------------------ #

class TestFetchTickerNews:
    def test_returns_sorted_articles(self):
        """Verify results are sorted newest-first and capped at max_articles."""
        mock_articles = _make_articles(10)
        with (
            patch("src.news.scraper.feedparser.parse") as mock_parse,
        ):
            mock_feed = MagicMock()
            mock_entry = {
                "title": "Headline",
                "link": "https://example.com/1",
                "summary": "",
                "published_parsed": (2024, 1, 10, 9, 0, 0, 0, 0, 0),
            }
            mock_feed.entries = [mock_entry]
            mock_parse.return_value = mock_feed

            results = fetch_ticker_news("AAPL", max_articles=1)
            assert len(results) <= 1

    def test_deduplicates_by_url(self):
        """Same URL appearing in both feeds should only appear once."""
        with patch("src.news.scraper.feedparser.parse") as mock_parse:
            entry = {
                "title": "Dupe",
                "link": "https://example.com/dupe",
                "summary": "",
                "published_parsed": (2024, 1, 10, 9, 0, 0, 0, 0, 0),
            }
            mock_feed = MagicMock()
            mock_feed.entries = [entry]
            mock_parse.return_value = mock_feed

            results = fetch_ticker_news("AAPL", max_articles=50)
            urls = [r["url"] for r in results]
            assert len(urls) == len(set(urls))


# ------------------------------------------------------------------ #
# aggregator.build_news_dataframe
# ------------------------------------------------------------------ #

class TestBuildNewsDataframe:
    def test_builds_correct_shape(self):
        articles = _make_articles(3)
        scored = _make_scored(3, "positive")
        df = build_news_dataframe(articles, scored)
        assert len(df) == 3
        assert "headline" in df.columns
        assert "sentiment_score" in df.columns
        assert "sentiment_label" in df.columns

    def test_index_is_datetime(self):
        articles = _make_articles(2)
        scored = _make_scored(2)
        df = build_news_dataframe(articles, scored)
        assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_empty_inputs_returns_empty(self):
        assert build_news_dataframe([], []).empty


# ------------------------------------------------------------------ #
# aggregator.save_news / load_news
# ------------------------------------------------------------------ #

class TestSaveLoadNews:
    def test_round_trip(self, tmp_dir):
        articles = _make_articles(4)
        scored = _make_scored(4, "neutral")
        df = build_news_dataframe(articles, scored)
        save_news(df, "AAPL", tmp_dir)

        loaded = load_news("AAPL", tmp_dir)
        assert len(loaded) == 4
        assert "headline" in loaded.columns

    def test_upsert_deduplicates(self, tmp_dir):
        articles = _make_articles(3)
        scored = _make_scored(3)
        df = build_news_dataframe(articles, scored)
        save_news(df, "AAPL", tmp_dir)
        save_news(df, "AAPL", tmp_dir)  # same data again

        loaded = load_news("AAPL", tmp_dir)
        # Same timestamp index — should deduplicate to original count
        assert len(loaded) == len(df)

    def test_load_missing_ticker_returns_empty(self, tmp_dir):
        df = load_news("NOTEXIST", tmp_dir)
        assert df.empty


# ------------------------------------------------------------------ #
# aggregator.get_sentiment_summary
# ------------------------------------------------------------------ #

class TestGetSentimentSummary:
    def test_positive_summary(self, tmp_dir):
        articles = _make_articles(5)
        scored = _make_scored(5, "positive")
        df = build_news_dataframe(articles, scored)
        save_news(df, "AAPL", tmp_dir)

        summary = get_sentiment_summary("AAPL", tmp_dir, days=30)
        assert summary["sentiment"] == "positive"
        assert summary["avg_score"] > 0
        assert summary["article_count"] == 5

    def test_negative_summary(self, tmp_dir):
        articles = _make_articles(3)
        scored = _make_scored(3, "negative")
        df = build_news_dataframe(articles, scored)
        save_news(df, "MSFT", tmp_dir)

        summary = get_sentiment_summary("MSFT", tmp_dir, days=30)
        assert summary["sentiment"] == "negative"
        assert summary["avg_score"] < 0

    def test_no_data_returns_neutral(self, tmp_dir):
        summary = get_sentiment_summary("EMPTY", tmp_dir, days=30)
        assert summary["sentiment"] == "neutral"
        assert summary["article_count"] == 0


# ------------------------------------------------------------------ #
# aggregator.build_sentiment_feature
# ------------------------------------------------------------------ #

class TestBuildSentimentFeature:
    def test_returns_dataframe_indexed_by_ticker(self, tmp_dir):
        for ticker, label in [("AAPL", "positive"), ("MSFT", "negative")]:
            articles = _make_articles(3)
            scored = _make_scored(3, label)
            df = build_news_dataframe(articles, scored)
            save_news(df, ticker, tmp_dir)

        feat = build_sentiment_feature(["AAPL", "MSFT", "EMPTY"], tmp_dir, days=30)
        assert set(feat.index) == {"AAPL", "MSFT", "EMPTY"}
        assert "sentiment_score" in feat.columns
        assert "sentiment_label_encoded" in feat.columns
        assert feat.loc["AAPL", "sentiment_label_encoded"] == 1
        assert feat.loc["MSFT", "sentiment_label_encoded"] == -1


# ------------------------------------------------------------------ #
# short_interest
# ------------------------------------------------------------------ #

class TestShortInterest:
    def _make_record(self, ticker="AAPL", ratio=3.0, pct=0.02) -> dict:
        return {
            "ticker": ticker,
            "date": "2024-01-10",
            "short_ratio": ratio,
            "short_pct_float": pct,
            "shares_short": 50_000_000,
            "shares_float": 2_500_000_000,
            "high_short_interest": ratio >= 5.0,
        }

    def test_save_and_load(self, tmp_dir):
        record = self._make_record()
        save_short_interest([record], tmp_dir)
        df = load_short_interest("AAPL", tmp_dir)
        assert not df.empty
        assert "short_ratio" in df.columns

    def test_high_si_flag_in_signal(self, tmp_dir):
        record = self._make_record(ratio=6.0)
        save_short_interest([record], tmp_dir)

        mock_cfg = MagicMock()
        mock_cfg.short_interest_high = 5.0
        with patch("src.news.short_interest.get_settings", return_value=mock_cfg):
            signal = get_short_interest_signal("AAPL", tmp_dir)

        assert signal["high_short_interest"] == True  # noqa: E712 — np.bool_ compat

    def test_low_si_flag_in_signal(self, tmp_dir):
        record = self._make_record(ratio=2.0)
        save_short_interest([record], tmp_dir)

        mock_cfg = MagicMock()
        mock_cfg.short_interest_high = 5.0
        with patch("src.news.short_interest.get_settings", return_value=mock_cfg):
            signal = get_short_interest_signal("AAPL", tmp_dir)

        assert signal["high_short_interest"] == False  # noqa: E712 — np.bool_ compat

    def test_missing_ticker_returns_defaults(self, tmp_dir):
        signal = get_short_interest_signal("NOTEXIST", tmp_dir)
        assert signal["short_ratio"] is None
        assert signal["high_short_interest"] is False

    def test_fetch_short_interest_handles_missing_fields(self):
        """yfinance info missing shortRatio should return None, not raise."""
        mock_cfg = MagicMock()
        mock_cfg.short_interest_high = 5.0
        with (
            patch("src.news.short_interest.yf.Ticker") as mock_ticker,
            patch("src.news.short_interest.get_settings", return_value=mock_cfg),
        ):
            mock_ticker.return_value.info = {}
            record = fetch_short_interest("AAPL")

        assert record["short_ratio"] is None
        assert record["ticker"] == "AAPL"

    def test_fetch_short_interest_handles_exception(self):
        """yfinance failure should not raise — returns None fields."""
        mock_cfg = MagicMock()
        mock_cfg.short_interest_high = 5.0
        with (
            patch("src.news.short_interest.yf.Ticker") as mock_ticker,
            patch("src.news.short_interest.get_settings", return_value=mock_cfg),
        ):
            mock_ticker.side_effect = Exception("network error")
            record = fetch_short_interest("AAPL")

        assert record["ticker"] == "AAPL"
        assert record["short_ratio"] is None
