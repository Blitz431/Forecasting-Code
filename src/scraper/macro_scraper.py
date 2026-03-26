"""Macroeconomic data scraper using FRED API.

Pulls series like GDP, Fed Funds Rate, CPI, VIX, etc.
"""

from datetime import date
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.scraper.storage import get_latest_date, get_ticker_filepath, upsert_dataframe
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def _get_fred_client():
    """Get a FRED API client. Raises if no API key configured."""
    from fredapi import Fred

    settings = get_settings()
    if not settings.fred.api_key:
        raise ValueError(
            "FRED_API_KEY not set. Get a free key at: "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return Fred(api_key=settings.fred.api_key)


def download_fred_series(
    series_id: str,
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    """Download a single FRED series.

    Args:
        series_id: FRED series ID (e.g., "GDP", "FEDFUNDS").
        start: Start date.
        end: End date (defaults to today).

    Returns:
        DataFrame with DatetimeIndex and column named after the series.
    """
    fred = _get_fred_client()

    if start is None:
        settings = get_settings()
        start = f"{settings.backfill_start_year}-01-01"

    try:
        series = fred.get_series(series_id, observation_start=start, observation_end=end)
        df = series.to_frame(name=series_id)
        df.index.name = "Date"
        df = df.dropna()
        logger.info(f"Downloaded FRED series '{series_id}': {len(df)} observations")
        return df
    except Exception as e:
        logger.error(f"Failed to download FRED series '{series_id}': {e}")
        return pd.DataFrame()


def scrape_macro(
    series_ids: list[str] | None = None,
    data_dir: Path | None = None,
    backfill: bool = False,
) -> dict[str, int]:
    """Scrape and store FRED macroeconomic series.

    Each series is stored as its own Parquet file in the macro directory.

    Args:
        series_ids: List of FRED series IDs. Defaults to settings.fred_series.
        data_dir: Directory to store Parquet files.
        backfill: Force full download from backfill_start_year.

    Returns:
        Dict mapping series_id -> number of new rows added.
    """
    settings = get_settings()

    if series_ids is None:
        series_ids = settings.fred_series

    if data_dir is None:
        data_dir = settings.raw_macro_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for series_id in series_ids:
        filepath = get_ticker_filepath(series_id, data_dir)

        # Determine start date
        if backfill:
            start = f"{settings.backfill_start_year}-01-01"
        else:
            latest = get_latest_date(filepath)
            if latest is not None:
                start = (latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if start > date.today().isoformat():
                    logger.debug(f"[{series_id}] Already up to date")
                    continue
            else:
                start = f"{settings.backfill_start_year}-01-01"

        df = download_fred_series(series_id, start=start)

        if not df.empty:
            upsert_dataframe(df, filepath)
            results[series_id] = len(df)

    logger.info(f"Macro scrape complete: {len(results)}/{len(series_ids)} series updated")
    return results
