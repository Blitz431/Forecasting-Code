from pathlib import Path

import pandas as pd

from src.utils.logging import setup_logger
from src.utils.validation import remove_duplicates

logger = setup_logger(__name__)

"""
Purpose: Parquet-based upsert storage layer — read, write, and deduplicate per-ticker data files.

Connections:
  - src/utils/validation.py: calls remove_duplicates() on every upsert
  - src/utils/logging.py: logger
  - Used by: price_scraper.py, dividend_scraper.py, macro_scraper.py, congress_tracker.py,
    insider_tracker.py, and all signal modules that persist results

In:  pd.DataFrame + target .parquet file path
Out: merged, deduplicated DataFrame written to disk; returns merged DataFrame
"""


def save_dataframe(df: pd.DataFrame, filepath: Path) -> None:
    """Save a DataFrame to Parquet, creating parent directories if needed.

    Args:
        df: DataFrame to save.
        filepath: Target .parquet file path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, engine="pyarrow")
    logger.debug(f"Saved {len(df)} rows to {filepath}")


def load_dataframe(filepath: Path) -> pd.DataFrame:
    """Load a DataFrame from Parquet.

    Args:
        filepath: Path to .parquet file.

    Returns:
        DataFrame, or empty DataFrame if file doesn't exist.
    """
    if not filepath.exists():
        return pd.DataFrame()

    df = pd.read_parquet(filepath, engine="pyarrow")
    logger.debug(f"Loaded {len(df)} rows from {filepath}")
    return df


def upsert_dataframe(new_data: pd.DataFrame, filepath: Path) -> pd.DataFrame:
    """Upsert new data into an existing Parquet file.

    Loads existing data, concatenates with new data, deduplicates
    by index (keeping newest), sorts by index, and saves.

    Args:
        new_data: New DataFrame to merge in.
        filepath: Target .parquet file path.

    Returns:
        The merged DataFrame.
    """
    existing = load_dataframe(filepath)

    if existing.empty:
        merged = new_data
    elif new_data.empty:
        return existing
    else:
        merged = pd.concat([existing, new_data])

    merged = remove_duplicates(merged)
    merged = merged.sort_index()

    save_dataframe(merged, filepath)
    logger.info(f"Upserted {len(new_data)} new rows -> {len(merged)} total in {filepath.name}")

    return merged


def get_ticker_filepath(ticker: str, data_dir: Path) -> Path:
    """Get the Parquet file path for a specific ticker.

    Args:
        ticker: Stock ticker symbol.
        data_dir: Base directory (e.g., data/raw/daily).

    Returns:
        Path like data/raw/daily/AAPL.parquet
    """
    return data_dir / f"{ticker}.parquet"


def list_stored_tickers(data_dir: Path) -> list[str]:
    """List all tickers that have stored data in a directory.

    Args:
        data_dir: Directory containing .parquet files.

    Returns:
        List of ticker symbols.
    """
    if not data_dir.exists():
        return []
    return sorted(f.stem for f in data_dir.glob("*.parquet"))


def get_latest_date(filepath: Path) -> pd.Timestamp | None:
    """Get the most recent date in a stored Parquet file.

    Args:
        filepath: Path to .parquet file.

    Returns:
        Latest timestamp, or None if file doesn't exist/is empty.
    """
    df = load_dataframe(filepath)
    if df.empty:
        return None
    return pd.Timestamp(df.index.max())


def export_to_excel(df: pd.DataFrame, filepath: Path) -> None:
    """Export a DataFrame to Excel for manual review.

    Args:
        df: DataFrame to export.
        filepath: Target .xlsx file path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(filepath, engine="openpyxl")
    logger.info(f"Exported {len(df)} rows to {filepath}")
