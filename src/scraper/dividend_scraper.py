from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import get_settings
from src.scraper._yf import extract_ticker_frame, yf_download
from src.scraper.storage import get_latest_dates, get_ticker_filepath, upsert_dataframe
from src.utils.logging import setup_logger
from src.utils.parallel import thread_map

logger = setup_logger(__name__)

"""
Purpose: Fetch dividend history and current yield from yfinance and store per-ticker parquets.

Connections:
  - src/scraper/_yf.py: extract_ticker_frame(), yf_download() — shared, rate-limit-aware batch access
  - src/scraper/storage.py: upsert_dataframe(), get_ticker_filepath(), get_latest_dates()
  - src/utils/parallel.py: thread_map() for concurrent batched downloads
  - config/settings.py: backfill start year, raw_dividends_dir, batch/worker sizing
  - cli/scrape.py: calls scrape_dividends()
  - src/ml/feature_engineer.py: reads dividends parquets as ML features (yield, ex-date proximity)

In:  list of ticker symbols
Out: data/raw/dividends/{ticker}.parquet (Dividends column indexed by ex-dividend date)
"""

# yf.Ticker().dividends (the legacy per-ticker path) returns a tz-aware
# America/New_York index, and that's what's already stored in existing
# parquet files. yf_download()'s batched index is tz-naive, so we localize
# it to this zone before upserting to stay compatible with existing data
# without needing a migration/backfill.
_DIVIDEND_TZ = "America/New_York"


def download_dividends_batch(tickers: list[str], start: str | None = None) -> dict[str, pd.DataFrame]:
    """Download dividend history for multiple tickers in one (or more) batched calls.

    Args:
        tickers: List of ticker symbols.
        start: Start date (defaults to backfill_start_year).

    Returns:
        Dict mapping ticker -> DataFrame with a single 'Dividends' column,
        indexed by ex-dividend date (tz-aware, America/New_York). Tickers
        with no dividend rows in the window are omitted from the dict.
    """
    settings = get_settings()
    if start is None:
        start = f"{settings.backfill_start_year}-01-01"

    batch_size = settings.yfinance_batch_size
    results: dict[str, pd.DataFrame] = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        data = yf_download(batch, start=start, actions=True)
        if data.empty:
            continue

        for ticker in batch:
            try:
                df = extract_ticker_frame(data, ticker)
                if df.empty or "Dividends" not in df.columns:
                    continue

                divs = df["Dividends"]
                # yf_download's Dividends column is DENSE (0.0-filled on every
                # non-ex-dividend trading day) — filter down to actual ex-dividend
                # rows so we don't balloon storage with mostly-zero rows.
                divs = divs[divs.notna() & (divs != 0)]
                if divs.empty:
                    continue

                divs_df = divs.to_frame(name="Dividends")
                divs_df.index.name = "Date"

                if divs_df.index.tz is None:
                    divs_df.index = divs_df.index.tz_localize(_DIVIDEND_TZ)
                else:
                    divs_df.index = divs_df.index.tz_convert(_DIVIDEND_TZ)

                results[ticker] = divs_df
            except Exception as e:
                logger.warning(f"[{ticker}] Failed to extract dividends: {e}")

    return results


def download_dividends(ticker: str, start: str | None = None) -> pd.DataFrame:
    """Download dividend history for a single ticker.

    Args:
        ticker: Stock ticker symbol.
        start: Start date (defaults to backfill_start_year).

    Returns:
        DataFrame with columns: Dividends, indexed by ex-dividend date.
    """
    return download_dividends_batch([ticker], start).get(ticker, pd.DataFrame())


def get_dividend_yield(ticker: str) -> float | None:
    """Get current dividend yield for a ticker.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dividend yield as a decimal (e.g., 0.025 for 2.5%), or None.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get("dividendYield")
    except Exception:
        return None


def scrape_dividends(
    tickers: list[str],
    data_dir: Path | None = None,
    backfill: bool = False,
    max_workers: int | None = None,
) -> dict[str, int]:
    """Scrape and store dividend data for a list of tickers.

    Args:
        tickers: List of ticker symbols.
        data_dir: Directory to store Parquet files.
        backfill: Force full download.
        max_workers: Max concurrent batched downloads (defaults to settings.scrape_max_workers).

    Returns:
        Dict mapping ticker -> number of dividend records.
    """
    settings = get_settings()
    if data_dir is None:
        data_dir = settings.raw_dividends_dir
    if max_workers is None:
        max_workers = settings.scrape_max_workers

    data_dir.mkdir(parents=True, exist_ok=True)
    batch_size = settings.yfinance_batch_size
    results: dict[str, int] = {}

    if backfill:
        start_date = f"{settings.backfill_start_year}-01-01"
        groups: dict[str, list[str]] = {start_date: list(tickers)}
    else:
        filepaths = {ticker: get_ticker_filepath(ticker, data_dir) for ticker in tickers}
        latest_dates = get_latest_dates(filepaths, max_workers=settings.scrape_io_workers)

        groups = {}
        for ticker in tickers:
            latest = latest_dates.get(ticker)
            if latest is None:
                start = f"{settings.backfill_start_year}-01-01"
            else:
                start = (latest + timedelta(days=1)).strftime("%Y-%m-%d")
            groups.setdefault(start, []).append(ticker)

    # Chunk each start-date group by yfinance_batch_size, same shape as
    # price_scraper's incremental grouping.
    today_str = date.today().isoformat()
    jobs: list[tuple[str, list[str]]] = []
    for start, group_tickers in groups.items():
        if not backfill and start > today_str:
            logger.debug(f"Skipping {len(group_tickers)} tickers — already up to date")
            continue
        for i in range(0, len(group_tickers), batch_size):
            jobs.append((start, group_tickers[i : i + batch_size]))

    def _run_job(job: tuple[str, list[str]]) -> dict[str, pd.DataFrame]:
        start, batch = job
        return download_dividends_batch(batch, start=start)

    job_results = thread_map(_run_job, jobs, max_workers=max_workers, label="dividend-batch")

    for job_result in job_results:
        if not job_result:
            continue
        for ticker, df in job_result.items():
            if df.empty:
                continue
            filepath = get_ticker_filepath(ticker, data_dir)
            upsert_dataframe(df, filepath)
            results[ticker] = len(df)

    logger.info(f"Dividend scrape complete: {len(results)}/{len(tickers)} tickers with dividends")
    return results
