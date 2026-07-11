from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import get_settings
from src.scraper.storage import upsert_dataframe, load_dataframe
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Fetch and track short interest metrics (short ratio, % float, shares short) from yfinance per ticker.

Connections:
  - src/scraper/storage.py: upsert_dataframe(), load_dataframe() for parquet persistence
  - config/settings.py: short_interest_high threshold (5.0 days-to-cover), news_short_interest_dir
  - src/news/runner.py: calls fetch_short_interest(), save_short_interest(), get_short_interest_signal()
  - src/ranking/ranker.py: reads short interest signal as one ranking input

In:  ticker symbol string
Out: data/news/short_interest/{ticker}.parquet (short_ratio, short_pct_float, shares_short, high_short_interest flag)
"""

# yfinance info keys we care about
_FIELDS = {
    "shortRatio": "short_ratio",
    "shortPercentOfFloat": "short_pct_float",
    "sharesShort": "shares_short",
    "floatShares": "shares_float",
}


def fetch_short_interest(ticker: str) -> dict:
    """Fetch current short interest data for a single ticker.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dict with keys: ticker, date, short_ratio, short_pct_float,
        shares_short, shares_float, high_short_interest.
        Numeric fields are None when data is unavailable.
    """
    settings = get_settings()
    info: dict = {}

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        logger.warning(f"[{ticker}] yfinance info fetch failed: {exc}")

    row: dict = {"ticker": ticker, "date": date.today().isoformat()}

    for yf_key, col_name in _FIELDS.items():
        raw = info.get(yf_key)
        row[col_name] = float(raw) if raw is not None else None

    short_ratio = row.get("short_ratio")
    row["high_short_interest"] = (
        short_ratio is not None and short_ratio >= settings.short_interest_high
    )

    return row


def fetch_batch_short_interest(tickers: list[str]) -> list[dict]:
    """Fetch short interest for multiple tickers.

    Args:
        tickers: List of ticker symbols.

    Returns:
        List of short interest dicts (one per ticker).
    """
    results = []
    for i, ticker in enumerate(tickers):
        logger.info(f"[{ticker}] Fetching short interest ({i + 1}/{len(tickers)})")
        results.append(fetch_short_interest(ticker))
    return results


def save_short_interest(records: list[dict], short_dir: Path) -> None:
    """Persist a batch of short interest records to per-ticker Parquet files.

    Each record is appended to data/news/short_interest/{ticker}.parquet
    with the fetch date as the index, so we build a day-by-day history.

    Args:
        records: Output of :func:`fetch_batch_short_interest`.
        short_dir: Directory to store files (e.g. data/news/short_interest/).
    """
    short_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        ticker = record["ticker"]
        filepath = short_dir / f"{ticker}.parquet"

        row_df = pd.DataFrame([record]).set_index("date")
        row_df.index = pd.to_datetime(row_df.index)
        row_df = row_df.drop(columns=["ticker"], errors="ignore")

        upsert_dataframe(row_df, filepath)
        logger.debug(f"[{ticker}] Short interest saved")


def load_short_interest(ticker: str, short_dir: Path) -> pd.DataFrame:
    """Load historical short interest data for a ticker.

    Args:
        ticker: Stock ticker symbol.
        short_dir: Directory where short interest Parquet files are stored.

    Returns:
        DataFrame with date index and short interest columns.
        Empty DataFrame if no data is stored yet.
    """
    filepath = short_dir / f"{ticker}.parquet"
    return load_dataframe(filepath)


def get_short_interest_signal(ticker: str, short_dir: Path) -> dict:
    """Return the most recent short interest snapshot and a high/low flag.

    Args:
        ticker: Stock ticker symbol.
        short_dir: Directory containing short interest Parquet files.

    Returns:
        Dict with latest short_ratio, short_pct_float, and high_short_interest flag.
        Returns defaults (ratio=None, flag=False) if no data is stored.
    """
    df = load_short_interest(ticker, short_dir)
    if df.empty:
        return {"ticker": ticker, "short_ratio": None, "short_pct_float": None, "high_short_interest": False}

    latest = df.sort_index().iloc[-1]
    settings = get_settings()
    short_ratio = latest.get("short_ratio")

    return {
        "ticker": ticker,
        "short_ratio": short_ratio,
        "short_pct_float": latest.get("short_pct_float"),
        "high_short_interest": (
            short_ratio is not None and short_ratio >= settings.short_interest_high
        ),
    }
