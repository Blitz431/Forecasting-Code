"""Tests for src/scraper/price_scraper.py.

Covers:
  - MultiIndex column-level extraction (both yfinance orderings), via
    src/scraper/_yf.extract_ticker_frame (used internally by download_batch).
  - Batch/grouping logic in scrape_prices (backfill vs incremental vs
    already-up-to-date skip), with download_batch fully mocked — no
    network calls.
  - aggregate_to_quarterly's mtime-based skip/recompute logic.
  - Numeric correctness of the quarterly resample (mean OHLC, sum Volume).

No live yfinance calls are made anywhere in this file.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.scraper._yf import extract_ticker_frame
from src.scraper.price_scraper import aggregate_to_quarterly, scrape_prices
from src.scraper.storage import load_dataframe, save_dataframe


def _ohlcv(dates: list[str], **overrides) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="Date")
    n = len(dates)
    df = pd.DataFrame(
        {
            "Open": [10.0 + i for i in range(n)],
            "High": [11.0 + i for i in range(n)],
            "Low": [9.0 + i for i in range(n)],
            "Close": [10.5 + i for i in range(n)],
            "Volume": [1000.0 + i for i in range(n)],
        },
        index=idx,
    )
    for k, v in overrides.items():
        df[k] = v
    return df


# ------------------------------------------------------------------ #
# extract_ticker_frame — both MultiIndex column orderings
# ------------------------------------------------------------------ #

class TestExtractTickerFrame:
    def test_ticker_first_ordering(self):
        idx = pd.date_range("2020-01-01", periods=3)
        cols = pd.MultiIndex.from_product(
            [["AAPL", "MSFT"], ["Open", "Close"]], names=["Ticker", "Price"]
        )
        data = pd.DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], index=idx, columns=cols
        )
        out = extract_ticker_frame(data, "AAPL")
        assert list(out.columns) == ["Open", "Close"]
        assert out.loc[idx[0], "Open"] == 1
        assert out.loc[idx[0], "Close"] == 2

    def test_price_first_ordering(self):
        idx = pd.date_range("2020-01-01", periods=3)
        cols = pd.MultiIndex.from_product(
            [["Open", "Close"], ["AAPL"]], names=["Price", "Ticker"]
        )
        data = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]], index=idx, columns=cols
        )
        out = extract_ticker_frame(data, "AAPL")
        assert list(out.columns) == ["Open", "Close"]
        assert out.loc[idx[0], "Open"] == 1
        assert out.loc[idx[0], "Close"] == 2

    def test_ticker_not_present_returns_empty(self):
        idx = pd.date_range("2020-01-01", periods=2)
        cols = pd.MultiIndex.from_product(
            [["AAPL"], ["Open", "Close"]], names=["Ticker", "Price"]
        )
        data = pd.DataFrame([[1, 2], [3, 4]], index=idx, columns=cols)
        out = extract_ticker_frame(data, "GOOG")
        assert out.empty

    def test_all_nan_rows_dropped(self):
        idx = pd.date_range("2020-01-01", periods=3)
        cols = pd.MultiIndex.from_product(
            [["AAPL"], ["Open", "Close"]], names=["Ticker", "Price"]
        )
        data = pd.DataFrame(
            [[1, 2], [None, None], [5, 6]], index=idx, columns=cols
        )
        out = extract_ticker_frame(data, "AAPL")
        assert len(out) == 2


# ------------------------------------------------------------------ #
# scrape_prices — batch grouping logic (download_batch fully mocked)
# ------------------------------------------------------------------ #

class TestScrapePricesGrouping:
    def test_new_ticker_goes_to_backfill_group(self, tmp_path):
        """Ticker with no existing file must be downloaded from
        backfill_start_year, not incrementally."""
        calls = []

        def fake_download_batch(tickers, start, end=None, retries=2):
            calls.append((tuple(tickers), start))
            return {t: _ohlcv(["2024-01-02"]) for t in tickers}

        with patch("src.scraper.price_scraper.download_batch", side_effect=fake_download_batch):
            results = scrape_prices(["NEWTICK"], data_dir=tmp_path, max_workers=1)

        assert results == {"NEWTICK": 1}
        assert len(calls) == 1
        tickers, start = calls[0]
        assert tickers == ("NEWTICK",)
        from config.settings import get_settings
        assert start == f"{get_settings().backfill_start_year}-01-01"

    def test_existing_ticker_goes_incremental_from_day_after_latest(self, tmp_path):
        p = tmp_path / "OLD.parquet"
        save_dataframe(_ohlcv(["2024-01-01", "2024-01-02"]), p)

        calls = []

        def fake_download_batch(tickers, start, end=None, retries=2):
            calls.append((tuple(tickers), start))
            return {t: _ohlcv(["2024-01-03"]) for t in tickers}

        with patch("src.scraper.price_scraper.download_batch", side_effect=fake_download_batch):
            results = scrape_prices(["OLD"], data_dir=tmp_path, max_workers=1)

        assert len(calls) == 1
        tickers, start = calls[0]
        assert tickers == ("OLD",)
        assert start == "2024-01-03"  # day after last stored date
        merged = load_dataframe(p)
        assert len(merged) == 3  # 2 existing + 1 new

    def test_already_up_to_date_ticker_is_skipped_no_download_call(self, tmp_path):
        """If latest stored date is today, computed start > today and the
        unit is skipped entirely — download_batch must not be called for it."""
        p = tmp_path / "UPTODATE.parquet"
        today = pd.Timestamp.today().normalize()
        save_dataframe(_ohlcv([today.strftime("%Y-%m-%d")]), p)

        calls = []

        def fake_download_batch(tickers, start, end=None, retries=2):
            calls.append((tuple(tickers), start))
            return {}

        with patch("src.scraper.price_scraper.download_batch", side_effect=fake_download_batch):
            results = scrape_prices(["UPTODATE"], data_dir=tmp_path, max_workers=1)

        assert calls == []
        assert results == {}

    def test_backfill_flag_forces_backfill_even_with_existing_data(self, tmp_path):
        p = tmp_path / "HASDATA.parquet"
        save_dataframe(_ohlcv(["2024-01-01"]), p)

        calls = []

        def fake_download_batch(tickers, start, end=None, retries=2):
            calls.append((tuple(tickers), start))
            return {t: _ohlcv(["2015-01-01"]) for t in tickers}

        with patch("src.scraper.price_scraper.download_batch", side_effect=fake_download_batch):
            scrape_prices(["HASDATA"], data_dir=tmp_path, backfill=True, max_workers=1)

        from config.settings import get_settings
        assert len(calls) == 1
        assert calls[0][1] == f"{get_settings().backfill_start_year}-01-01"

    def test_tickers_with_same_start_date_grouped_into_one_batch(self, tmp_path):
        """Two tickers with no existing data (both go to backfill) should be
        combined into shared batch(es), not one download_batch call per
        ticker — this is the core grouping optimization under test."""
        calls = []

        def fake_download_batch(tickers, start, end=None, retries=2):
            calls.append((tuple(sorted(tickers)), start))
            return {t: _ohlcv(["2024-01-02"]) for t in tickers}

        with patch("src.scraper.price_scraper.download_batch", side_effect=fake_download_batch):
            results = scrape_prices(["AAA", "BBB"], data_dir=tmp_path, max_workers=1)

        assert len(calls) == 1  # single shared batch call, not 2 individual calls
        assert calls[0][0] == ("AAA", "BBB")
        assert set(results.keys()) == {"AAA", "BBB"}

    def test_invalid_ohlcv_from_download_is_not_stored(self, tmp_path):
        """download_batch returning data that fails validate_ohlcv (e.g.
        missing columns) must not be upserted or counted in results."""
        def fake_download_batch(tickers, start, end=None, retries=2):
            bad = pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex(["2024-01-02"]))
            return {t: bad for t in tickers}

        with patch("src.scraper.price_scraper.download_batch", side_effect=fake_download_batch):
            results = scrape_prices(["BADTICK"], data_dir=tmp_path, max_workers=1)

        assert results == {}
        assert not (tmp_path / "BADTICK.parquet").exists()

    def test_empty_ticker_list(self, tmp_path):
        with patch("src.scraper.price_scraper.download_batch") as mock_dl:
            results = scrape_prices([], data_dir=tmp_path, max_workers=1)
        assert results == {}
        mock_dl.assert_not_called()


# ------------------------------------------------------------------ #
# aggregate_to_quarterly — mtime skip logic
# ------------------------------------------------------------------ #

class TestAggregateToQuarterlyMtimeSkip:
    def _make_daily(self, path: Path) -> None:
        save_dataframe(
            _ohlcv(["2024-01-02", "2024-02-01", "2024-03-01", "2024-04-01"]),
            path,
        )

    def test_skips_when_quarterly_is_up_to_date(self, tmp_path):
        daily_dir = tmp_path / "daily"
        quarterly_dir = tmp_path / "quarterly"
        daily_dir.mkdir()
        quarterly_dir.mkdir()

        daily_path = daily_dir / "TICK.parquet"
        self._make_daily(daily_path)

        quarterly_path = quarterly_dir / "TICK.parquet"
        save_dataframe(_ohlcv(["2024-03-31"]), quarterly_path)
        # Ensure quarterly mtime >= daily mtime.
        now = time.time()
        os.utime(daily_path, (now, now))
        os.utime(quarterly_path, (now + 5, now + 5))

        with patch("src.scraper.price_scraper.upsert_dataframe") as mock_upsert:
            aggregate_to_quarterly(daily_dir, quarterly_dir, tickers=["TICK"])

        mock_upsert.assert_not_called()

    def test_recomputes_when_daily_is_newer(self, tmp_path):
        daily_dir = tmp_path / "daily"
        quarterly_dir = tmp_path / "quarterly"
        daily_dir.mkdir()
        quarterly_dir.mkdir()

        daily_path = daily_dir / "TICK.parquet"
        self._make_daily(daily_path)

        quarterly_path = quarterly_dir / "TICK.parquet"
        save_dataframe(_ohlcv(["2024-03-31"]), quarterly_path)

        now = time.time()
        os.utime(quarterly_path, (now, now))
        os.utime(daily_path, (now + 5, now + 5))  # daily newer -> must recompute

        with patch("src.scraper.price_scraper.upsert_dataframe") as mock_upsert:
            aggregate_to_quarterly(daily_dir, quarterly_dir, tickers=["TICK"])

        mock_upsert.assert_called_once()

    def test_force_always_recomputes(self, tmp_path):
        daily_dir = tmp_path / "daily"
        quarterly_dir = tmp_path / "quarterly"
        daily_dir.mkdir()
        quarterly_dir.mkdir()

        daily_path = daily_dir / "TICK.parquet"
        self._make_daily(daily_path)

        quarterly_path = quarterly_dir / "TICK.parquet"
        save_dataframe(_ohlcv(["2024-03-31"]), quarterly_path)

        now = time.time()
        os.utime(daily_path, (now, now))
        os.utime(quarterly_path, (now + 100, now + 100))  # quarterly much newer

        with patch("src.scraper.price_scraper.upsert_dataframe") as mock_upsert:
            aggregate_to_quarterly(daily_dir, quarterly_dir, tickers=["TICK"], force=True)

        mock_upsert.assert_called_once()

    def test_missing_daily_file_is_skipped_silently(self, tmp_path):
        daily_dir = tmp_path / "daily"
        quarterly_dir = tmp_path / "quarterly"
        daily_dir.mkdir()
        quarterly_dir.mkdir()

        with patch("src.scraper.price_scraper.upsert_dataframe") as mock_upsert:
            # Should not raise even though NOFILE.parquet doesn't exist.
            aggregate_to_quarterly(daily_dir, quarterly_dir, tickers=["NOFILE"])

        mock_upsert.assert_not_called()


# ------------------------------------------------------------------ #
# aggregate_to_quarterly — numeric correctness of the resample
# ------------------------------------------------------------------ #

class TestAggregateToQuarterlyNumericCorrectness:
    def test_mean_ohlc_sum_volume(self, tmp_path):
        daily_dir = tmp_path / "daily"
        quarterly_dir = tmp_path / "quarterly"
        daily_dir.mkdir()
        quarterly_dir.mkdir()

        # Three known days, all within Q1 2024.
        idx = pd.DatetimeIndex(
            ["2024-01-02", "2024-02-01", "2024-03-01"], name="Date"
        )
        daily = pd.DataFrame(
            {
                "Open": [10.0, 20.0, 30.0],
                "High": [15.0, 25.0, 35.0],
                "Low": [5.0, 15.0, 25.0],
                "Close": [12.0, 22.0, 32.0],
                "Volume": [100.0, 200.0, 300.0],
            },
            index=idx,
        )
        daily_path = daily_dir / "KNOWN.parquet"
        save_dataframe(daily, daily_path)

        aggregate_to_quarterly(daily_dir, quarterly_dir, tickers=["KNOWN"])

        quarterly_path = quarterly_dir / "KNOWN.parquet"
        result = load_dataframe(quarterly_path)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["Open"] == pytest.approx((10.0 + 20.0 + 30.0) / 3)
        assert row["High"] == pytest.approx((15.0 + 25.0 + 35.0) / 3)
        assert row["Low"] == pytest.approx((5.0 + 15.0 + 25.0) / 3)
        assert row["Close"] == pytest.approx((12.0 + 22.0 + 32.0) / 3)
        assert row["Volume"] == pytest.approx(100.0 + 200.0 + 300.0)
        # Quarter-end label for a QE resample of Q1 2024.
        assert result.index[0] == pd.Timestamp("2024-03-31")

    def test_multiple_quarters_produce_multiple_rows(self, tmp_path):
        daily_dir = tmp_path / "daily"
        quarterly_dir = tmp_path / "quarterly"
        daily_dir.mkdir()
        quarterly_dir.mkdir()

        idx = pd.DatetimeIndex(
            ["2024-01-02", "2024-04-02", "2024-07-02"], name="Date"
        )
        daily = pd.DataFrame(
            {
                "Open": [10.0, 20.0, 30.0],
                "High": [10.0, 20.0, 30.0],
                "Low": [10.0, 20.0, 30.0],
                "Close": [10.0, 20.0, 30.0],
                "Volume": [1.0, 2.0, 3.0],
            },
            index=idx,
        )
        save_dataframe(daily, daily_dir / "MULTI.parquet")

        aggregate_to_quarterly(daily_dir, quarterly_dir, tickers=["MULTI"])

        result = load_dataframe(quarterly_dir / "MULTI.parquet")
        assert len(result) == 3
        assert result["Close"].tolist() == [10.0, 20.0, 30.0]
