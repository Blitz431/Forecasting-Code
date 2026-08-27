from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.scraper._yf import extract_ticker_frame, yf_download
from src.scraper.storage import get_latest_dates, get_ticker_filepath, upsert_dataframe
from src.utils.logging import setup_logger
from src.utils.parallel import thread_map
from src.utils.validation import validate_ohlcv

logger = setup_logger(__name__)

"""
Purpose: Download and store daily OHLCV prices from yfinance — supports incremental updates and full backfill.

Connections:
  - src/scraper/_yf.py: yf_download(), extract_ticker_frame() — shared, rate-limit-aware yfinance access
  - src/scraper/storage.py: upsert_dataframe(), get_latest_dates(), get_ticker_filepath()
  - src/utils/parallel.py: thread_map() for concurrent per-batch downloads
  - src/utils/validation.py: validate_ohlcv() after each batch download
  - config/settings.py: batch size, backfill start year, directory paths, worker counts
  - cli/scrape.py: calls scrape_prices() and aggregate_to_quarterly()

In:  list of ticker symbols; optional data_dir and backfill flag
Out: data/raw/daily/{ticker}.parquet (daily OHLCV), data/raw/quarterly/{ticker}.parquet (resampled quarterly)
"""


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
    if end is None:
        end = date.today().isoformat()
    if isinstance(start, date):
        start = start.isoformat()
    if isinstance(end, date):
        end = end.isoformat()

    logger.info(f"Downloading {len(tickers)} tickers from {start} to {end}")

    data = yf_download(tickers, start=start, end=end, retries=retries)

    if data.empty:
        return {}

    results: dict[str, pd.DataFrame] = {}

    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                df = extract_ticker_frame(data, ticker)
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
    max_workers: int | None = None,
) -> dict[str, int]:
    """Scrape and store price data for a list of tickers.

    For each ticker:
    - If backfill=True or no existing data: download from backfill_start_year
    - Otherwise: download from the day after the last stored date

    Args:
        tickers: List of ticker symbols.
        data_dir: Directory to store Parquet files. Defaults to settings.
        backfill: Force full historical download.
        max_workers: Max concurrent batch downloads. Defaults to settings.scrape_max_workers.

    Returns:
        Dict mapping ticker -> number of new rows added.
    """
    settings = get_settings()
    if data_dir is None:
        data_dir = settings.raw_daily_dir
    if max_workers is None:
        max_workers = settings.scrape_max_workers

    data_dir.mkdir(parents=True, exist_ok=True)
    batch_size = settings.yfinance_batch_size
    results: dict[str, int] = {}

    # Batched lookup of the latest stored date per ticker (single call instead
    # of a per-ticker loop).
    filepaths = {ticker: get_ticker_filepath(ticker, data_dir) for ticker in tickers}
    latest_dates = get_latest_dates(filepaths, max_workers=settings.scrape_io_workers)

    # Group tickers by whether they need backfill or incremental
    backfill_tickers = []
    incremental_groups: dict[str, list[str]] = {}  # start_date -> [tickers]

    for ticker in tickers:
        latest = latest_dates.get(ticker)

        if backfill or latest is None:
            backfill_tickers.append(ticker)
        else:
            # Start from the day after the last stored date
            start = (latest + timedelta(days=1)).strftime("%Y-%m-%d")
            if start not in incremental_groups:
                incremental_groups[start] = []
            incremental_groups[start].append(ticker)

    # Build a flat list of (ticker_batch, start_date) work units
    units: list[tuple[list[str], str]] = []

    if backfill_tickers:
        start_date = f"{settings.backfill_start_year}-01-01"
        logger.info(f"Backfilling {len(backfill_tickers)} tickers from {start_date}")

        for i in range(0, len(backfill_tickers), batch_size):
            batch = backfill_tickers[i : i + batch_size]
            units.append((batch, start_date))

    for start, ticker_group in incremental_groups.items():
        if start > date.today().isoformat():
            logger.debug(f"Skipping {len(ticker_group)} tickers — already up to date")
            continue

        for i in range(0, len(ticker_group), batch_size):
            batch = ticker_group[i : i + batch_size]
            units.append((batch, start))

    def _download_and_store(unit: tuple[list[str], str]) -> dict[str, int]:
        batch, start = unit
        unit_results: dict[str, int] = {}
        data = download_batch(batch, start=start)

        for ticker, df in data.items():
            if not df.empty and validate_ohlcv(df, ticker):
                filepath = get_ticker_filepath(ticker, data_dir)
                upsert_dataframe(df, filepath)
                unit_results[ticker] = len(df)

        return unit_results

    unit_results_list = thread_map(
        _download_and_store, units, max_workers=max_workers, label="price-batch"
    )

    for unit_results in unit_results_list:
        if unit_results:
            results.update(unit_results)

    logger.info(f"Price scrape complete: {len(results)} tickers updated")
    return results


def aggregate_to_quarterly(
    daily_dir: Path,
    quarterly_dir: Path,
    tickers: list[str] | None = None,
    max_workers: int | None = None,
    force: bool = False,
) -> None:
    """Aggregate daily OHLCV data to quarterly averages.

    For each ticker, computes quarterly:
    - Average Close price
    - Total Volume
    - Average Open/High/Low

    Args:
        daily_dir: Directory with daily Parquet files.
        quarterly_dir: Directory to store quarterly Parquet files.
        tickers: Specific tickers to process. None = all available.
        max_workers: Max concurrent aggregation workers. Defaults to settings.scrape_io_workers.
        force: Bypass the mtime-based skip check and re-aggregate every ticker.
    """
    settings = get_settings()
    if max_workers is None:
        max_workers = settings.scrape_io_workers

    quarterly_dir.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        from src.scraper.storage import list_stored_tickers
        tickers = list_stored_tickers(daily_dir)

    def _aggregate_one(ticker: str) -> None:
        daily_path = get_ticker_filepath(ticker, daily_dir)
        if not daily_path.exists():
            return

        quarterly_path = get_ticker_filepath(ticker, quarterly_dir)
        if (
            not force
            and quarterly_path.exists()
            and quarterly_path.stat().st_mtime >= daily_path.stat().st_mtime
        ):
            return

        df = pd.read_parquet(daily_path, columns=["Open", "High", "Low", "Close", "Volume"])

        if df.empty:
            return

        # Resample to quarterly
        quarterly = df.resample("QE").agg({
            "Open": "mean",
            "High": "mean",
            "Low": "mean",
            "Close": "mean",
            "Volume": "sum",
        }).dropna()

        if not quarterly.empty:
            upsert_dataframe(quarterly, quarterly_path)

    thread_map(_aggregate_one, tickers, max_workers=max_workers, label="quarterly-agg")

    logger.info(f"Aggregated {len(tickers)} tickers to quarterly data")
