from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import get_settings
from src.scraper.storage import upsert_dataframe, load_dataframe
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Fetch options chain via yfinance; compute put/call ratio, unusual volume flag, and flow signal.

Connections:
  - src/scraper/storage.py: upsert_dataframe(), load_dataframe() to persist summary metrics
  - config/settings.py: options_unusual_volume_multiplier, options_dir
  - src/ranking/ranker.py: calls get_flow_signal() with weight 1.0

In:  ticker symbol string (live yfinance options chain — not persisted raw; chains change intraday)
Out: data/options/{ticker}.parquet (put_call_ratio, total_call_volume, total_put_volume, unusual_activity, flow_signal)
"""


# ---------------------------------------------------------------------------#
# Chain fetching & aggregation
# ---------------------------------------------------------------------------#

def _fetch_options_chain(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download all available options expiry chains for a ticker.

    Iterates over every expiration date yfinance returns and concatenates
    calls and puts into two aggregate DataFrames.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Tuple (calls_df, puts_df). Both may be empty on failure or thin markets.
        Each DataFrame has columns: strike, lastPrice, bid, ask, volume,
        openInterest, impliedVolatility (from yfinance).
    """
    t = yf.Ticker(ticker)
    try:
        expirations = t.options
    except Exception as exc:
        logger.warning(f"[{ticker}] Could not fetch options expirations: {exc}")
        return pd.DataFrame(), pd.DataFrame()

    if not expirations:
        logger.debug(f"[{ticker}] No options expirations available")
        return pd.DataFrame(), pd.DataFrame()

    all_calls: list[pd.DataFrame] = []
    all_puts: list[pd.DataFrame] = []

    for exp in expirations:
        try:
            chain = t.option_chain(exp)
            if chain.calls is not None and not chain.calls.empty:
                all_calls.append(chain.calls)
            if chain.puts is not None and not chain.puts.empty:
                all_puts.append(chain.puts)
        except Exception as exc:
            logger.debug(f"[{ticker}] Skipping expiry {exp}: {exc}")
            continue

    calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
    puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()
    return calls_df, puts_df


def _detect_unusual_activity(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    multiplier: float,
) -> bool:
    """Flag unusual options activity.

    A contract is considered "unusual" when its volume exceeds
    ``multiplier * openInterest``, indicating a large directional bet
    relative to existing positioning.

    Args:
        calls_df: Call contracts DataFrame from yfinance.
        puts_df: Put contracts DataFrame from yfinance.
        multiplier: Threshold multiplier vs open interest.

    Returns:
        True if any single contract qualifies as unusual.
    """
    for df in (calls_df, puts_df):
        if df.empty:
            continue
        vol = pd.to_numeric(df.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0)
        oi = pd.to_numeric(df.get("openInterest", pd.Series(dtype=float)), errors="coerce").fillna(0)
        # Only flag when there is meaningful open interest to compare against
        mask = (oi > 100) & (vol > multiplier * oi)
        if mask.any():
            return True
    return False


# ---------------------------------------------------------------------------#
# Main compute function
# ---------------------------------------------------------------------------#

def compute_options_metrics(ticker: str) -> dict:
    """Fetch options chain and compute Put/Call ratio + unusual activity.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dict with keys:
          - ticker (str)
          - fetch_time (datetime, UTC-aware)
          - put_call_ratio (float | None): put OI / call OI; None if no data
          - total_call_volume (int)
          - total_put_volume (int)
          - unusual_activity (bool)
          - flow_signal (float): [-1, 1], positive = bullish
    """
    settings = get_settings()
    calls_df, puts_df = _fetch_options_chain(ticker)

    result: dict = {
        "ticker": ticker,
        "fetch_time": datetime.now(tz=timezone.utc),
        "put_call_ratio": None,
        "total_call_volume": 0,
        "total_put_volume": 0,
        "unusual_activity": False,
        "flow_signal": 0.0,
    }

    if calls_df.empty and puts_df.empty:
        logger.info(f"[{ticker}] No options chain data available")
        return result

    # Volume and open interest
    call_vol = int(pd.to_numeric(calls_df.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not calls_df.empty else 0
    put_vol = int(pd.to_numeric(puts_df.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not puts_df.empty else 0

    call_oi = pd.to_numeric(calls_df.get("openInterest", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() if not calls_df.empty else 0.0
    put_oi = pd.to_numeric(puts_df.get("openInterest", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() if not puts_df.empty else 0.0

    pcr = (put_oi / call_oi) if call_oi > 0 else None

    unusual = _detect_unusual_activity(
        calls_df, puts_df,
        multiplier=settings.options_unusual_volume_multiplier,
    )

    # Flow signal: call dominance = bullish (+1), put dominance = bearish (-1)
    total_vol = call_vol + put_vol
    if total_vol > 0:
        flow_signal = (call_vol - put_vol) / total_vol
    else:
        flow_signal = 0.0

    result.update({
        "put_call_ratio": float(pcr) if pcr is not None else None,
        "total_call_volume": call_vol,
        "total_put_volume": put_vol,
        "unusual_activity": unusual,
        "flow_signal": float(flow_signal),
    })

    pcr_str = f"{pcr:.3f}" if pcr is not None else "N/A"
    logger.info(
        f"[{ticker}] PCR={pcr_str} "
        f"call_vol={call_vol} put_vol={put_vol} unusual={unusual}"
    )
    return result


def compute_batch_options_metrics(
    tickers: list[str],
    delay: float = 0.5,
) -> dict[str, dict]:
    """Compute options metrics for multiple tickers.

    Args:
        tickers: List of ticker symbols.
        delay: Seconds to sleep between tickers (rate limiting).

    Returns:
        Dict mapping ticker -> metrics dict.
    """
    results: dict[str, dict] = {}
    for i, ticker in enumerate(tickers):
        results[ticker] = compute_options_metrics(ticker)
        if delay > 0 and i < len(tickers) - 1:
            time.sleep(delay)
    return results


# ---------------------------------------------------------------------------#
# Persistence
# ---------------------------------------------------------------------------#

def save_options_metrics(metrics: dict, options_dir: Path) -> None:
    """Persist options metrics snapshot to data/options/{ticker}.parquet.

    Appends one row per call so we can track put/call ratio trends over time.

    Args:
        metrics: Output of :func:`compute_options_metrics`.
        options_dir: Root options storage directory.
    """
    options_dir.mkdir(parents=True, exist_ok=True)
    ticker = metrics["ticker"]
    filepath = options_dir / f"{ticker}.parquet"

    row = {
        "put_call_ratio": metrics["put_call_ratio"],
        "total_call_volume": metrics["total_call_volume"],
        "total_put_volume": metrics["total_put_volume"],
        "unusual_activity": metrics["unusual_activity"],
        "flow_signal": metrics["flow_signal"],
    }
    df = pd.DataFrame([row], index=pd.DatetimeIndex([metrics["fetch_time"]], name="date"))
    upsert_dataframe(df, filepath)
    logger.debug(f"[{ticker}] Options metrics saved")


def load_options_metrics(ticker: str, options_dir: Path) -> pd.DataFrame:
    """Load historical options metrics for a ticker.

    Args:
        ticker: Stock ticker symbol.
        options_dir: Root options storage directory.

    Returns:
        DataFrame with DatetimeIndex and columns: put_call_ratio,
        total_call_volume, total_put_volume, unusual_activity, flow_signal.
        Empty DataFrame if no data stored yet.
    """
    filepath = options_dir / f"{ticker}.parquet"
    return load_dataframe(filepath)


# ---------------------------------------------------------------------------#
# Signal
# ---------------------------------------------------------------------------#

def get_options_signal(ticker: str, options_dir: Path) -> dict:
    """Return the most recent options signal snapshot for the ranker.

    Args:
        ticker: Stock ticker symbol.
        options_dir: Root options storage directory.

    Returns:
        Dict with keys: ticker, put_call_ratio, total_call_volume,
        total_put_volume, unusual_activity, flow_signal.
        Returns neutral defaults when no data is stored.
    """
    base = {
        "ticker": ticker,
        "put_call_ratio": None,
        "total_call_volume": 0,
        "total_put_volume": 0,
        "unusual_activity": False,
        "flow_signal": 0.0,
    }
    df = load_options_metrics(ticker, options_dir)
    if df.empty:
        return base

    latest = df.sort_index().iloc[-1]
    return {
        "ticker": ticker,
        "put_call_ratio": latest.get("put_call_ratio"),
        "total_call_volume": int(latest.get("total_call_volume", 0)),
        "total_put_volume": int(latest.get("total_put_volume", 0)),
        "unusual_activity": bool(latest.get("unusual_activity", False)),
        "flow_signal": float(latest.get("flow_signal", 0.0)),
    }
