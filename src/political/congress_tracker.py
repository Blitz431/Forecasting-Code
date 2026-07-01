from __future__ import annotations

import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from config.settings import get_settings
from src.scraper.storage import upsert_dataframe, load_dataframe
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Fetch and track U.S. congressional stock trades (STOCK Act) via Quiver Quant API.

Connections:
  - src/scraper/storage.py: upsert_dataframe(), load_dataframe() for parquet persistence
  - config/settings.py: quiver_base_url, congress_lookback_days, political_congress_dir
  - src/ranking/ranker.py: calls get_congress_signal() as one ranking input

In:  ticker symbol string
Out: data/political/congress/{ticker}.parquet (representative, transaction, amount_mid, party, chamber, report_date)
     get_congress_signal() → float [-1, +1] based on net buy/sell value over lookback window
"""

# ---------------------------------------------------------------------------#
# Amount parsing
# ---------------------------------------------------------------------------#

# Quiver Quant reports amounts as "$X - $Y" range strings.
# We map these to midpoint dollar values for signal computation.
_AMOUNT_RANGE_MIDPOINTS: dict[str, float] = {
    "$1,001 - $15,000":           8_000.0,
    "$15,001 - $50,000":          32_500.0,
    "$50,001 - $100,000":         75_000.0,
    "$100,001 - $250,000":       175_000.0,
    "$250,001 - $500,000":       375_000.0,
    "$500,001 - $1,000,000":     750_000.0,
    "$1,000,001 - $5,000,000": 3_000_000.0,
    "Above $5,000,000":        7_500_000.0,
}


def _parse_amount(raw: str | None) -> float | None:
    """Convert a Quiver Quant amount range string to an estimated midpoint.

    Falls back to a regex-based extraction when the string is not in the
    known lookup table (e.g. slightly different formatting from API).

    Args:
        raw: Amount string like "$1,001 - $15,000" or None.

    Returns:
        Estimated midpoint in USD, or None if unparseable.
    """
    if not raw:
        return None

    raw = raw.strip()

    # Exact match lookup (fast path)
    if raw in _AMOUNT_RANGE_MIDPOINTS:
        return _AMOUNT_RANGE_MIDPOINTS[raw]

    # "Above $X" pattern
    above_match = re.search(r"[Aa]bove\s*\$?([\d,]+)", raw)
    if above_match:
        val = float(above_match.group(1).replace(",", ""))
        return val * 1.5  # rough estimate: 50% above the stated floor

    # Generic "$X - $Y" pattern
    nums = re.findall(r"\$?([\d,]+)", raw)
    if len(nums) >= 2:
        lo = float(nums[0].replace(",", ""))
        hi = float(nums[1].replace(",", ""))
        return (lo + hi) / 2.0
    if len(nums) == 1:
        return float(nums[0].replace(",", ""))

    return None


# ---------------------------------------------------------------------------#
# API fetching
# ---------------------------------------------------------------------------#

_CONGRESS_ENDPOINT = "{base}/historical/congresstrading/{ticker}"

# Standard browser-like headers — Quiver Quant blocks bare Python UA strings
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def fetch_congress_trades(
    ticker: str,
    lookback_days: int | None = None,
    timeout: int = 15,
) -> list[dict]:
    """Fetch congressional trades for a single ticker from Quiver Quant.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        lookback_days: Only return trades within this many days.
                       Defaults to settings.congress_lookback_days.
        timeout: HTTP request timeout in seconds.

    Returns:
        List of trade dicts with keys:
          - date (datetime, UTC-aware): trade/transaction date
          - ticker (str)
          - representative (str)
          - transaction (str): "Purchase" or "Sale"
          - amount_mid (float | None): estimated midpoint dollar value
          - party (str): "D", "R", or ""
          - chamber (str): "House" or "Senate" or ""
          - report_date (str): ISO date the trade was reported (STOCK Act filing)
        Sorted oldest-first.
    """
    settings = get_settings()
    cutoff_days = lookback_days if lookback_days is not None else settings.congress_lookback_days
    cutoff_date = datetime.now(tz=timezone.utc) - timedelta(days=cutoff_days)

    url = _CONGRESS_ENDPOINT.format(base=settings.quiver_base_url, ticker=ticker)

    raw_data: list[dict] = []
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        if resp.status_code == 404:
            logger.debug(f"[{ticker}] No congressional trade data on Quiver Quant (404)")
            return []
        resp.raise_for_status()
        raw_data = resp.json()
    except requests.RequestException as exc:
        logger.warning(f"[{ticker}] Quiver Quant request failed: {exc}")
        return []
    except ValueError as exc:
        logger.warning(f"[{ticker}] Quiver Quant JSON parse error: {exc}")
        return []

    trades: list[dict] = []
    for item in raw_data:
        try:
            raw_date = item.get("Date") or item.get("TransactionDate") or ""
            if not raw_date:
                continue
            trade_date = datetime.fromisoformat(raw_date).replace(tzinfo=timezone.utc)

            if trade_date < cutoff_date:
                continue

            trades.append({
                "date": trade_date,
                "ticker": ticker.upper(),
                "representative": str(item.get("Representative", "") or "").strip(),
                "transaction": str(item.get("Transaction", "") or "").strip(),
                "amount_mid": _parse_amount(item.get("Amount")),
                "party": str(item.get("Party", "") or "").strip(),
                "chamber": str(item.get("Chamber", "") or "").strip(),
                "report_date": str(item.get("ReportDate", "") or "").strip(),
            })
        except Exception as exc:
            logger.debug(f"[{ticker}] Skipping malformed trade record: {exc}")
            continue

    trades.sort(key=lambda t: t["date"])
    logger.info(f"[{ticker}] Fetched {len(trades)} congressional trades (last {cutoff_days}d)")
    return trades


def fetch_batch_congress_trades(
    tickers: list[str],
    lookback_days: int | None = None,
    delay: float = 1.0,
) -> dict[str, list[dict]]:
    """Fetch congressional trades for multiple tickers.

    Args:
        tickers: List of ticker symbols.
        lookback_days: Days lookback; defaults to settings value.
        delay: Seconds to sleep between requests (rate limiting).

    Returns:
        Dict mapping ticker -> list of trade dicts.
    """
    results: dict[str, list[dict]] = {}
    for i, ticker in enumerate(tickers):
        results[ticker] = fetch_congress_trades(ticker, lookback_days=lookback_days)
        if delay > 0 and i < len(tickers) - 1:
            time.sleep(delay)
    return results


# ---------------------------------------------------------------------------#
# Persistence
# ---------------------------------------------------------------------------#

def save_congress_trades(records: list[dict], congress_dir: Path) -> None:
    """Persist congressional trade records to per-ticker Parquet files.

    Each record is upserted (no duplicates) into:
    data/political/congress/{ticker}.parquet

    Args:
        records: Output of :func:`fetch_congress_trades` (any single ticker).
        congress_dir: Directory for congress Parquet files.
    """
    if not records:
        return

    congress_dir.mkdir(parents=True, exist_ok=True)
    ticker = records[0]["ticker"]
    filepath = congress_dir / f"{ticker}.parquet"

    df = pd.DataFrame(records).set_index("date")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.drop(columns=["ticker"], errors="ignore")

    upsert_dataframe(df, filepath)
    logger.debug(f"[{ticker}] Saved {len(records)} congressional trades to {filepath.name}")


def save_batch_congress_trades(
    batch: dict[str, list[dict]],
    congress_dir: Path,
) -> None:
    """Save congress trades for all tickers in a batch result.

    Args:
        batch: Output of :func:`fetch_batch_congress_trades`.
        congress_dir: Storage directory.
    """
    for ticker, records in batch.items():
        if records:
            save_congress_trades(records, congress_dir)


def load_congress_trades(ticker: str, congress_dir: Path) -> pd.DataFrame:
    """Load all stored congressional trades for a ticker.

    Args:
        ticker: Stock ticker symbol.
        congress_dir: Directory containing congress Parquet files.

    Returns:
        DataFrame indexed by trade date (UTC DatetimeIndex) with columns:
        representative, transaction, amount_mid, party, chamber, report_date.
        Empty DataFrame if no data is stored.
    """
    filepath = congress_dir / f"{ticker}.parquet"
    return load_dataframe(filepath)


# ---------------------------------------------------------------------------#
# Signal
# ---------------------------------------------------------------------------#

def get_congress_signal(
    ticker: str,
    congress_dir: Path,
    window_days: int | None = None,
) -> dict:
    """Compute a congressional trading signal for the ranker.

    Aggregates trades over the most recent ``window_days`` and computes:
      - congress_net_buys / congress_net_sells: transaction counts
      - congress_buy_value / congress_sell_value: estimated dollar flow
      - congress_signal: normalised [-1, 1] float (positive = bullish)

    Formula for congress_signal:
      (buy_value - sell_value) / (buy_value + sell_value + 1)
    The +1 prevents division by zero when both are 0.

    Args:
        ticker: Stock ticker symbol.
        congress_dir: Storage directory.
        window_days: Rolling window. Defaults to settings.congress_lookback_days.

    Returns:
        Dict with keys: ticker, congress_net_buys, congress_net_sells,
        congress_buy_value, congress_sell_value, congress_signal.
    """
    settings = get_settings()
    days = window_days if window_days is not None else settings.congress_lookback_days

    base: dict = {
        "ticker": ticker,
        "congress_net_buys": 0,
        "congress_net_sells": 0,
        "congress_buy_value": 0.0,
        "congress_sell_value": 0.0,
        "congress_signal": 0.0,
    }

    df = load_congress_trades(ticker, congress_dir)
    if df.empty:
        return base

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    recent = df[df.index >= cutoff]
    if recent.empty:
        return base

    # Normalise transaction label — API may return "Purchase"/"Sale" or "Buy"/"Sell"
    is_buy = recent["transaction"].str.lower().str.startswith("p") | \
             recent["transaction"].str.lower().str.startswith("b")
    is_sell = recent["transaction"].str.lower().str.startswith("s")

    buy_rows = recent[is_buy]
    sell_rows = recent[is_sell]

    buy_value = buy_rows["amount_mid"].fillna(0).sum()
    sell_value = sell_rows["amount_mid"].fillna(0).sum()

    total = buy_value + sell_value
    signal = (buy_value - sell_value) / (total + 1.0)

    return {
        "ticker": ticker,
        "congress_net_buys": int(len(buy_rows)),
        "congress_net_sells": int(len(sell_rows)),
        "congress_buy_value": float(buy_value),
        "congress_sell_value": float(sell_value),
        "congress_signal": float(signal),
    }
