import os
import threading
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.utils.logging import setup_logger
from src.utils.parallel import thread_map
from src.utils.validation import remove_duplicates

logger = setup_logger(__name__)

"""
Purpose: Parquet-based upsert storage layer — read, write, and deduplicate per-ticker data files.

Connections:
  - src/utils/validation.py: calls remove_duplicates() on every upsert
  - src/utils/logging.py: logger
  - src/utils/parallel.py: thread_map() for batched get_latest_dates()
  - Used by: price_scraper.py, dividend_scraper.py, macro_scraper.py, congress_tracker.py,
    insider_tracker.py, and all signal modules that persist results

In:  pd.DataFrame + target .parquet file path
Out: merged, deduplicated DataFrame written to disk; returns merged DataFrame
"""

_file_locks_guard = threading.Lock()
_file_locks: dict[str, threading.Lock] = {}


def _file_lock(filepath: Path) -> threading.Lock:
    """Get (or lazily create) a lock scoped to a specific file path.

    The same resolved path always maps to the same Lock object, so
    concurrent readers/writers of the same file serialize with each other
    while unrelated files are unaffected.
    """
    key = str(filepath.resolve())
    with _file_locks_guard:
        lock = _file_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _file_locks[key] = lock
        return lock


def _write_parquet_atomic(df: pd.DataFrame, filepath: Path) -> None:
    """Write a DataFrame to filepath atomically via a temp file + os.replace.

    Caller is responsible for holding the appropriate `_file_lock`.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_name(filepath.name + ".tmp")
    try:
        df.to_parquet(tmp_path, engine="pyarrow")
        os.replace(tmp_path, filepath)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def save_dataframe(df: pd.DataFrame, filepath: Path) -> None:
    """Save a DataFrame to Parquet, creating parent directories if needed.

    Writes are atomic (write to a .tmp file, then os.replace) and
    serialized per-file so concurrent saves/upserts of the same path
    don't race.

    Args:
        df: DataFrame to save.
        filepath: Target .parquet file path.
    """
    with _file_lock(filepath):
        _write_parquet_atomic(df, filepath)
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

    The full read-modify-write is serialized per-file (see `_file_lock`)
    so concurrent upserts/saves of the same path can't interleave.

    Args:
        new_data: New DataFrame to merge in.
        filepath: Target .parquet file path.

    Returns:
        The merged DataFrame.
    """
    with _file_lock(filepath):
        existing = load_dataframe(filepath)

        if existing.empty:
            merged = new_data
        elif new_data.empty:
            return existing
        else:
            merged = pd.concat([existing, new_data])

        if merged.index.has_duplicates:
            merged = remove_duplicates(merged)
        if not merged.index.is_monotonic_increasing:
            merged = merged.sort_index()

        _write_parquet_atomic(merged, filepath)

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

    Tries to resolve this from Parquet row-group statistics (metadata
    only, no row data read). Falls back to reading just the index column
    if statistics are unavailable or the index can't be confidently
    resolved from metadata.

    Args:
        filepath: Path to .parquet file.

    Returns:
        Latest timestamp, or None if file doesn't exist/is empty.
    """
    if not filepath.exists():
        return None

    try:
        pf = pq.ParquetFile(filepath)
        pandas_metadata = pf.schema_arrow.pandas_metadata or {}
        index_columns = pandas_metadata.get("index_columns", [])

        if len(index_columns) != 1 or not isinstance(index_columns[0], str):
            raise ValueError("Cannot confidently resolve a single index column")

        index_col_name = index_columns[0]
        if index_col_name.startswith("__index_level_"):
            raise ValueError("Index column stored without a name")

        col_idx = pf.schema_arrow.get_field_index(index_col_name)
        if col_idx == -1:
            raise ValueError("Index column missing from schema")

        max_val = None
        for rg in range(pf.metadata.num_row_groups):
            stats = pf.metadata.row_group(rg).column(col_idx).statistics
            if stats is None or not stats.has_min_max:
                raise ValueError("Row group statistics unavailable")
            if max_val is None or stats.max > max_val:
                max_val = stats.max

        if max_val is None:
            raise ValueError("No row groups found")

        return pd.Timestamp(max_val)

    except Exception:
        pass

    try:
        df = pd.read_parquet(filepath, columns=[])
    except Exception:
        return None

    if df.index.empty:
        return None

    latest = df.index.max()
    if pd.isna(latest):
        return None

    return pd.Timestamp(latest)


def get_latest_dates(
    filepaths: dict[str, Path], max_workers: int = 8
) -> dict[str, pd.Timestamp | None]:
    """Get the most recent stored date for multiple Parquet files in parallel.

    Args:
        filepaths: Mapping of key (e.g. ticker) to .parquet file path.
        max_workers: Max concurrent threads for `thread_map`.

    Returns:
        Mapping of every key in `filepaths` to its latest timestamp, or
        None if the file is missing/empty/unreadable.
    """
    keys = list(filepaths.keys())
    results = thread_map(
        lambda key: get_latest_date(filepaths[key]),
        keys,
        max_workers=max_workers,
        label="get_latest_dates",
    )
    return dict(zip(keys, results))


def export_to_excel(df: pd.DataFrame, filepath: Path) -> None:
    """Export a DataFrame to Excel for manual review.

    Args:
        df: DataFrame to export.
        filepath: Target .xlsx file path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(filepath, engine="openpyxl")
    logger.info(f"Exported {len(df)} rows to {filepath}")
