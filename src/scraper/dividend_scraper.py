from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import get_settings
from src.scraper.storage import get_ticker_filepath, upsert_dataframe
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Fetch dividend history and current yield from yfinance and store per-ticker parquets.

Connections:
  - src/scraper/storage.py: upsert_dataframe(), get_ticker_filepath()
  - config/settings.py: backfill start year, raw_dividends_dir
  - cli/scrape.py: calls scrape_dividends()
  - src/ml/feature_engineer.py: reads dividends parquets as ML features (yield, ex-date proximity)

In:  list of ticker symbols
Out: data/raw/dividends/{ticker}.parquet (Dividends column indexed by ex-dividend date)
"""


def download_dividends(ticker: str, start: str | None = None) -> pd.DataFrame:
    """Download dividend history for a single ticker.

    Args:
        ticker: Stock ticker symbol.
        start: Start date (defaults to backfill_start_year).

    Returns:
        DataFrame with columns: Dividends, indexed by ex-dividend date.
    """
    if start is None:
        settings = get_settings()
        start = f"{settings.backfill_start_year}-01-01"

    try:
        stock = yf.Ticker(ticker)
        divs = stock.dividends

        if divs.empty:
            return pd.DataFrame()

        # Filter by start date
        divs = divs[divs.index >= start]

        # Newer yfinance returns a DataFrame directly; older versions return a Series
        if isinstance(divs, pd.DataFrame):
            df = divs.rename(columns={divs.columns[0]: "Dividends"})
        else:
            df = divs.to_frame(name="Dividends")
        df.index.name = "Date"

        return df
    except Exception as e:
        logger.warning(f"[{ticker}] Failed to download dividends: {e}")
        return pd.DataFrame()


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
) -> dict[str, int]:
    """Scrape and store dividend data for a list of tickers.

    Args:
        tickers: List of ticker symbols.
        data_dir: Directory to store Parquet files.
        backfill: Force full download.

    Returns:
        Dict mapping ticker -> number of dividend records.
    """
    settings = get_settings()
    if data_dir is None:
        data_dir = settings.raw_dividends_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for ticker in tickers:
        try:
            df = download_dividends(ticker)

            if df.empty:
                continue

            filepath = get_ticker_filepath(ticker, data_dir)
            upsert_dataframe(df, filepath)
            results[ticker] = len(df)
        except Exception as e:
            logger.warning(f"[{ticker}] Dividend scrape error: {e}")

    logger.info(f"Dividend scrape complete: {len(results)}/{len(tickers)} tickers with dividends")
    return results
