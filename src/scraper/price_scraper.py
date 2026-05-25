"""Stock price data scraper using yfinance.

Supports both batch backfill (from 2015) and incremental daily updates.
Uses batched yf.download() for efficiency.
"""

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import get_settings
from src.scraper.storage import get_latest_date, get_ticker_filepath, upsert_dataframe
from src.utils.logging import setup_logger
from src.utils.validation import validate_ohlcv

logger = setup_logger(__name__)

_BATCH_DELAY_SECONDS = 3      # pause between batches to avoid rate limits
_MAX_RETRIES = 3              # retry a failed batch this many times
_RETRY_DELAY_SECONDS = 10     # wait before retrying a rate-limited batch


def download_batch(
    tickers: list[str],
    start: str | date,
    end: str | date | None = None,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for multiple tickers in one batch.

    Args:
        tickers: List of ticker symbols.
        start: Start date for data.
        end: End date (defaults to today).

    Returns:
        Dict mapping ticker -> DataFrame with OHLCV data.
    """
    if end is None:
        end = date.today().isoformat()

    if isinstance(start, date):
        start = start.isoformat()
    if isinstance(end, date):
        end = end.isoformat()

    logger.info(f"Downloading {len(tickers)} tickers from {start} to {end}")

    data = pd.DataFrame()
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            data = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
            )
            if not data.empty:
                break
            logger.warning(f"Batch returned empty data (attempt {attempt}/{_MAX_RETRIES})")
        except Exception as e:
            logger.warning(f"Batch download error (attempt {attempt}/{_MAX_RETRIES}): {e}")
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY_SECONDS)

    if data.empty:
        logger.error(f"Batch failed after {_MAX_RETRIES} attempts — skipping {len(tickers)} tickers")
        return {}

    results = {}

    if data.empty:
        return results

    # yfinance always returns MultiIndex columns now.
    # Single ticker: (Price, Ticker) — e.g., ('Close', 'AAPL')
    # Multi ticker with group_by="ticker": (Ticker, Price) — e.g., ('AAPL', 'Close')
    if isinstance(data.columns, pd.MultiIndex):
        level_names = data.columns.names  # ['Price', 'Ticker'] or ['Ticker', 'Price']

        # Find which level contains the ticker symbols
        if level_names[0] == "Ticker":
            ticker_level, price_level = 0, 1
        else:
            ticker_level, price_level = 1, 0

        available_tickers = data.columns.get_level_values(ticker_level).unique()

        for ticker in tickers:
            if ticker in available_tickers:
                try:
                    ticker_data = data.xs(ticker, level=ticker_level, axis=1).dropna(how="all")
                    if not ticker_data.empty:
                        results[ticker] = ticker_data
                except (KeyError, Exception) as e:
                    logger.warning(f"[{ticker}] No data in batch result: {e}")
    else:
        # Flat columns (very old yfinance versions)
        if len(tickers) == 1:
            results[tickers[0]] = data

    logger.info(f"Successfully downloaded data for {len(results)}/{len(tickers)} tickers")
    return results


def scrape_prices(
    tickers: list[str],
    data_dir: Path | None = None,
    backfill: bool = False,
) -> dict[str, int]:
    """Scrape and store price data for a list of tickers.

    For each ticker:
    - If backfill=True or no existing data: download from backfill_start_year
    - Otherwise: download from the day after the last stored date

    Args:
        tickers: List of ticker symbols.
        data_dir: Directory to store Parquet files. Defaults to settings.
        backfill: Force full historical download.

    Returns:
        Dict mapping ticker -> number of new rows added.
    """
    settings = get_settings()
    if data_dir is None:
        data_dir = settings.raw_daily_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    batch_size = settings.yfinance_batch_size
    results = {}

    # Group tickers by whether they need backfill or incremental
    backfill_tickers = []
    incremental_groups: dict[str, list[str]] = {}  # start_date -> [tickers]

    for ticker in tickers:
        filepath = get_ticker_filepath(ticker, data_dir)
        latest = get_latest_date(filepath)

        if backfill or latest is None:
            backfill_tickers.append(ticker)
        else:
            # Start from the day after the last stored date
            start = (latest + timedelta(days=1)).strftime("%Y-%m-%d")
            if start not in incremental_groups:
                incremental_groups[start] = []
            incremental_groups[start].append(ticker)

    # Process backfill tickers in batches
    if backfill_tickers:
        start_date = f"{settings.backfill_start_year}-01-01"
        total_batches = (len(backfill_tickers) + batch_size - 1) // batch_size
        logger.info(f"Backfilling {len(backfill_tickers)} tickers in {total_batches} batches from {start_date}")

        for i in range(0, len(backfill_tickers), batch_size):
            batch = backfill_tickers[i : i + batch_size]
            batch_num = i // batch_size + 1
            logger.info(f"Batch {batch_num}/{total_batches} — tickers {i+1}–{min(i+batch_size, len(backfill_tickers))}")
            data = download_batch(batch, start=start_date)

            for ticker, df in data.items():
                if validate_ohlcv(df, ticker):
                    filepath = get_ticker_filepath(ticker, data_dir)
                    upsert_dataframe(df, filepath)
                    results[ticker] = len(df)

            logger.info(f"Batch {batch_num}/{total_batches} done — {len(results)} tickers saved so far")
            if i + batch_size < len(backfill_tickers):
                time.sleep(_BATCH_DELAY_SECONDS)

    # Process incremental tickers
    for start, ticker_group in incremental_groups.items():
        if start > date.today().isoformat():
            logger.debug(f"Skipping {len(ticker_group)} tickers — already up to date")
            continue

        for i in range(0, len(ticker_group), batch_size):
            batch = ticker_group[i : i + batch_size]
            data = download_batch(batch, start=start)

            for ticker, df in data.items():
                if not df.empty and validate_ohlcv(df, ticker):
                    filepath = get_ticker_filepath(ticker, data_dir)
                    upsert_dataframe(df, filepath)
                    results[ticker] = len(df)

            if i + batch_size < len(ticker_group):
                time.sleep(_BATCH_DELAY_SECONDS)

    logger.info(f"Price scrape complete: {len(results)} tickers updated")
    return results


def aggregate_to_quarterly(daily_dir: Path, quarterly_dir: Path, tickers: list[str] | None = None) -> None:
    """Aggregate daily OHLCV data to quarterly averages.

    For each ticker, computes quarterly:
    - Average Close price
    - Total Volume
    - Average Open/High/Low

    Args:
        daily_dir: Directory with daily Parquet files.
        quarterly_dir: Directory to store quarterly Parquet files.
        tickers: Specific tickers to process. None = all available.
    """
    quarterly_dir.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        from src.scraper.storage import list_stored_tickers
        tickers = list_stored_tickers(daily_dir)

    for ticker in tickers:
        daily_path = get_ticker_filepath(ticker, daily_dir)
        df = pd.read_parquet(daily_path) if daily_path.exists() else pd.DataFrame()

        if df.empty:
            continue

        # Resample to quarterly
        quarterly = df.resample("QE").agg({
            "Open": "mean",
            "High": "mean",
            "Low": "mean",
            "Close": "mean",
            "Volume": "sum",
        }).dropna()

        if not quarterly.empty:
            quarterly_path = get_ticker_filepath(ticker, quarterly_dir)
            upsert_dataframe(quarterly, quarterly_path)

    logger.info(f"Aggregated {len(tickers)} tickers to quarterly data")
