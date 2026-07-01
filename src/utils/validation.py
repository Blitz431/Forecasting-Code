import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Data quality checks for OHLCV DataFrames — column presence, negative prices, and freshness.

Connections:
  - src/scraper/storage.py: calls remove_duplicates() on every upsert
  - src/scraper/price_scraper.py: calls validate_ohlcv() after each batch download
  - src/utils/logging.py: logger

In:  pd.DataFrame or pd.Series
Out: bool pass/fail or cleaned DataFrame (remove_duplicates)
"""


def validate_ohlcv(df: pd.DataFrame, ticker: str = "") -> bool:
    """Validate that a DataFrame has proper OHLCV structure.

    Args:
        df: DataFrame to validate.
        ticker: Ticker symbol for logging context.

    Returns:
        True if valid, False otherwise.
    """
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)

    if missing:
        logger.warning(f"[{ticker}] Missing OHLCV columns: {missing}")
        return False

    if df.empty:
        logger.warning(f"[{ticker}] DataFrame is empty")
        return False

    # Check for all-NaN columns
    for col in required_cols:
        if df[col].isna().all():
            logger.warning(f"[{ticker}] Column '{col}' is entirely NaN")
            return False

    # Check for negative prices
    price_cols = ["Open", "High", "Low", "Close"]
    for col in price_cols:
        if (df[col].dropna() < 0).any():
            logger.warning(f"[{ticker}] Negative values found in '{col}'")
            return False

    # Check High >= Low
    valid_hl = df["High"].dropna() >= df["Low"].dropna()
    if not valid_hl.all():
        bad_count = (~valid_hl).sum()
        logger.warning(f"[{ticker}] {bad_count} rows where High < Low")

    return True


def validate_series_length(series: pd.Series, min_length: int, context: str = "") -> bool:
    """Check that a time series has enough data points.

    Args:
        series: Time series to check.
        min_length: Minimum required length.
        context: Description for logging.

    Returns:
        True if series meets minimum length.
    """
    if len(series) < min_length:
        logger.warning(f"[{context}] Series has {len(series)} points, need at least {min_length}")
        return False
    return True


def check_data_freshness(df: pd.DataFrame, max_age_days: int = 3, ticker: str = "") -> bool:
    """Check if data is reasonably up to date.

    Args:
        df: DataFrame with a DatetimeIndex.
        max_age_days: Maximum acceptable age in calendar days.
        ticker: Ticker symbol for logging.

    Returns:
        True if data is fresh enough.
    """
    if df.empty:
        return False

    latest = pd.Timestamp(df.index.max())
    age = (pd.Timestamp.now() - latest).days

    if age > max_age_days:
        logger.info(f"[{ticker}] Data is {age} days old (threshold: {max_age_days})")
        return False

    return True


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate index entries, keeping the last occurrence."""
    if df.index.duplicated().any():
        n_dupes = df.index.duplicated().sum()
        logger.info(f"Removing {n_dupes} duplicate index entries")
        df = df[~df.index.duplicated(keep="last")]
    return df
