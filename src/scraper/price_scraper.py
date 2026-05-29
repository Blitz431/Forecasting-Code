"""Stock price data scraper using yfinance.

Supports both batch backfill (from 2015) and incremental daily updates.
Uses batched yf.download() for efficiency.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import get_settings
from src.scraper.storage import get_latest_date, get_ticker_filepath, upsert_dataframe
from src.utils.logging import setup_logger
from src.utils.validation import validate_ohlcv

logger = setup_logger(__name__)


def _extract_ticker_from_multiindex(
    data: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Pull a single ticker out of a yfinance MultiIndex DataFrame.

    yfinance 1.x is inconsistent about column level order:
      - Multi-ticker download  → ['Ticker', 'Price']  e.g. ('AAPL', 'Close')
      - Single-ticker download → ['Price', 'Ticker']  e.g. ('Close', 'AAPL')

    We detect which level holds the ticker names at call time so neither
    layout causes a KeyError.
    """
    level_names = data.columns.names  # e.g. ['Ticker', 'Price'] or ['Price', 'Ticker']

    if level_names[0] == "Ticker":
        ticker_level = 0
    else:
        ticker_level = 1

    available = data.columns.get_level_values(ticker_level).unique()
    if ticker not in available:
        return pd.DataFrame()

    df = data.xs(ticker, level=ticker_level, axis=1).dropna(how="all")
    # Normalise column names to title-case (Open/High/Low/Close/Volume)
    df.columns = [str(c).title() for c in df.columns]
    return df


def download_batch(
    tickers: list[str],
    start: str | date,
    end: str | date | None = None,
    retries: int = 2,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for multiple tickers in one yfinance batch.

    Args:
        tickers: List of ticker symbols.
        start: Start date for data.
        end: End date (defaults to today).
        retries: Number of retry attempts on failure.

    Returns:
        Dict mapping ticker -> DataFrame with OHLCV data.
    """
    import time

    if end is None:
        end = date.today().isoformat()
    if isinstance(start, date):
        start = start.isoformat()
    if isinstance(end, date):
        end = end.isoformat()

    logger.info(f"Downloading {len(tickers)} tickers from {start} to {end}")

    data = pd.DataFrame()
    for attempt in range(retries + 1):
        try:
            data = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            break   # success
        except Exception as exc:
            if attempt < retries:
                wait = 5 * (attempt + 1)
                logger.warning(f"Batch download attempt {attempt+1} failed: {exc}. Retrying in {wait}s …")
                time.sleep(wait)
            else:
                logger.error(f"Batch download failed after {retries+1} attempts: {exc}")
                return {}

    if data.empty:
        return {}

    results: dict[str, pd.DataFrame] = {}

    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                df = _extract_ticker_from_multiindex(data, ticker)
                if not df.empty:
                    results[ticker] = df
            except Exception as exc:
                logger.warning(f"[{ticker}] Could not extract from batch: {exc}")
    else:
        # Flat columns — only possible when a single ticker was passed
        if len(tickers) == 1:
            df = data.copy()
            df.columns = [str(c).title() for c in df.columns]
            if not df.empty:
                results[tickers[0]] = df

    logger.info(f"Downloaded data for {len(results)}/{len(tickers)} tickers")
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
        logger.info(f"Backfilling {len(backfill_tickers)} tickers from {start_date}")

        for i in range(0, len(backfill_tickers), batch_size):
            batch = backfill_tickers[i : i + batch_size]
            data = download_batch(batch, start=start_date)

            for ticker, df in data.items():
                if validate_ohlcv(df, ticker):
                    filepath = get_ticker_filepath(ticker, data_dir)
                    upsert_dataframe(df, filepath)
                    results[ticker] = len(df)

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
