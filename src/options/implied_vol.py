from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import get_settings
from src.scraper.storage import upsert_dataframe, load_dataframe
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Compute IV vs HV spread for each ticker and produce a vol signal for the ranker.

Connections:
  - src/scraper/storage.py: load_dataframe() reads daily prices; upsert_dataframe() saves results
  - config/settings.py: options_iv_spike_threshold, options_lookback_days, options_dir
  - src/ranking/ranker.py: calls get_vol_signal() with weight 0.5

In:  data/raw/daily/{ticker}.parquet (for HV); yfinance options chain (for ATM IV)
Out: data/options/iv/{ticker}.parquet (implied_vol, historical_vol, iv_hv_spread, vol_signal)
"""

# ATM band: include contracts whose strike is within this fraction of spot
_ATM_BAND = 0.05  # 5%


# ---------------------------------------------------------------------------#
# Historical volatility
# ---------------------------------------------------------------------------#

def _compute_historical_vol(ticker: str, raw_daily_dir: Path, window: int) -> float | None:
    """Compute annualised historical volatility from stored daily OHLCV data.

    Uses the close-to-close log-return standard deviation, annualised by
    multiplying by sqrt(252).

    Args:
        ticker: Ticker symbol.
        raw_daily_dir: Path to data/raw/daily/.
        window: Number of trading days to use (default from settings: 30).

    Returns:
        Annualised historical vol as a decimal (e.g. 0.25 = 25%), or None
        if insufficient data is available.
    """
    filepath = raw_daily_dir / f"{ticker}.parquet"
    if not filepath.exists():
        logger.debug(f"[{ticker}] No daily price data for HV computation")
        return None

    try:
        df = pd.read_parquet(filepath)
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < window + 1:
            logger.debug(f"[{ticker}] Not enough price history for {window}-day HV")
            return None
        returns = np.log(close / close.shift(1)).dropna()
        hv = float(returns.iloc[-window:].std() * math.sqrt(252))
        return hv
    except Exception as exc:
        logger.warning(f"[{ticker}] HV computation failed: {exc}")
        return None


# ---------------------------------------------------------------------------#
# Implied volatility extraction
# ---------------------------------------------------------------------------#

def _compute_implied_vol(ticker: str) -> tuple[float | None, float | None]:
    """Extract the median ATM implied volatility and spot price via yfinance.

    Fetches all available options expiries, filters for near-ATM contracts
    (within ATM_BAND of current spot), and returns the median IV.

    Args:
        ticker: Ticker symbol.

    Returns:
        Tuple (implied_vol, spot_price). Both are None on failure.
        implied_vol is a decimal (e.g. 0.30 = 30% annualised IV).
        spot_price is the last trade price used for the ATM filter.
    """
    t = yf.Ticker(ticker)

    # Get spot price
    try:
        info = t.fast_info
        spot = float(info.last_price) if info.last_price else None
    except Exception:
        spot = None

    if spot is None or spot <= 0:
        logger.debug(f"[{ticker}] Could not determine spot price for IV extraction")
        return None, None

    try:
        expirations = t.options
    except Exception as exc:
        logger.warning(f"[{ticker}] Could not fetch options expirations for IV: {exc}")
        return None, spot

    if not expirations:
        return None, spot

    iv_values: list[float] = []
    lo = spot * (1 - _ATM_BAND)
    hi = spot * (1 + _ATM_BAND)

    for exp in expirations:
        try:
            chain = t.option_chain(exp)
            for df in (chain.calls, chain.puts):
                if df is None or df.empty:
                    continue
                atm = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
                iv_raw = pd.to_numeric(atm.get("impliedVolatility", pd.Series(dtype=float)), errors="coerce")
                iv_clean = iv_raw.dropna()
                iv_clean = iv_clean[iv_clean > 0.001]  # filter out near-zero noise
                iv_values.extend(iv_clean.tolist())
        except Exception as exc:
            logger.debug(f"[{ticker}] Skipping expiry {exp} for IV: {exc}")
            continue

    if not iv_values:
        logger.debug(f"[{ticker}] No ATM IV data found across all expiries")
        return None, spot

    median_iv = float(np.median(iv_values))
    return median_iv, spot


# ---------------------------------------------------------------------------#
# Main compute function
# ---------------------------------------------------------------------------#

def compute_iv_metrics(ticker: str, raw_daily_dir: Path | None = None) -> dict:
    """Compute IV vs HV comparison metrics for a single ticker.

    Args:
        ticker: Stock ticker symbol.
        raw_daily_dir: Override for data/raw/daily/ directory.
                       Defaults to settings.raw_daily_dir.

    Returns:
        Dict with keys:
          - ticker (str)
          - fetch_time (datetime, UTC-aware)
          - spot_price (float | None)
          - implied_vol (float | None): annualised IV decimal
          - historical_vol (float | None): annualised 30-day HV decimal
          - iv_hv_spread (float | None): IV - HV in pp (None if either missing)
          - iv_spike (bool): True when iv_hv_spread > settings.options_iv_spike_threshold
          - vol_signal (float): [-1, 1], positive = IV < HV (calm options market)
    """
    settings = get_settings()
    daily_dir = raw_daily_dir or settings.raw_daily_dir
    window = settings.options_lookback_days

    implied_vol, spot = _compute_implied_vol(ticker)
    hist_vol = _compute_historical_vol(ticker, daily_dir, window)

    iv_hv_spread: float | None = None
    iv_spike = False
    vol_signal = 0.0

    if implied_vol is not None and hist_vol is not None:
        iv_hv_spread = implied_vol - hist_vol
        iv_spike = iv_hv_spread > settings.options_iv_spike_threshold
        # tanh(-spread / 0.15): positive when IV < HV, negative when IV >> HV
        vol_signal = float(math.tanh(-iv_hv_spread / 0.15))
    elif implied_vol is not None and hist_vol is None:
        # No HV available — use absolute IV as a rough proxy
        # High absolute IV (>50%) → slightly negative signal
        vol_signal = float(math.tanh(-(implied_vol - 0.25) / 0.15))

    result = {
        "ticker": ticker,
        "fetch_time": datetime.now(tz=timezone.utc),
        "spot_price": spot,
        "implied_vol": implied_vol,
        "historical_vol": hist_vol,
        "iv_hv_spread": iv_hv_spread,
        "iv_spike": iv_spike,
        "vol_signal": vol_signal,
    }

    logger.info(
        f"[{ticker}] IV={implied_vol:.1%} HV={hist_vol:.1%} spread={iv_hv_spread:+.1%} signal={vol_signal:.3f}"
        if implied_vol is not None and hist_vol is not None
        else f"[{ticker}] IV={implied_vol} HV={hist_vol} (incomplete data)"
    )
    return result


def compute_batch_iv_metrics(
    tickers: list[str],
    raw_daily_dir: Path | None = None,
    delay: float = 0.5,
) -> dict[str, dict]:
    """Compute IV metrics for multiple tickers.

    Args:
        tickers: List of ticker symbols.
        raw_daily_dir: Override for daily price data directory.
        delay: Seconds to sleep between tickers.

    Returns:
        Dict mapping ticker -> metrics dict.
    """
    results: dict[str, dict] = {}
    for i, ticker in enumerate(tickers):
        results[ticker] = compute_iv_metrics(ticker, raw_daily_dir=raw_daily_dir)
        if delay > 0 and i < len(tickers) - 1:
            time.sleep(delay)
    return results


# ---------------------------------------------------------------------------#
# Persistence
# ---------------------------------------------------------------------------#

def save_iv_metrics(metrics: dict, options_dir: Path) -> None:
    """Persist IV metrics snapshot to data/options/iv/{ticker}.parquet.

    Args:
        metrics: Output of :func:`compute_iv_metrics`.
        options_dir: Root options directory (IV saved in options_dir/iv/).
    """
    iv_dir = options_dir / "iv"
    iv_dir.mkdir(parents=True, exist_ok=True)
    ticker = metrics["ticker"]
    filepath = iv_dir / f"{ticker}.parquet"

    row = {
        "spot_price": metrics["spot_price"],
        "implied_vol": metrics["implied_vol"],
        "historical_vol": metrics["historical_vol"],
        "iv_hv_spread": metrics["iv_hv_spread"],
        "iv_spike": metrics["iv_spike"],
        "vol_signal": metrics["vol_signal"],
    }
    df = pd.DataFrame([row], index=pd.DatetimeIndex([metrics["fetch_time"]], name="date"))
    upsert_dataframe(df, filepath)
    logger.debug(f"[{ticker}] IV metrics saved")


def load_iv_metrics(ticker: str, options_dir: Path) -> pd.DataFrame:
    """Load historical IV metrics for a ticker.

    Args:
        ticker: Stock ticker symbol.
        options_dir: Root options directory.

    Returns:
        DataFrame with DatetimeIndex and columns: spot_price, implied_vol,
        historical_vol, iv_hv_spread, iv_spike, vol_signal.
        Empty DataFrame if no data stored.
    """
    filepath = options_dir / "iv" / f"{ticker}.parquet"
    return load_dataframe(filepath)


# ---------------------------------------------------------------------------#
# Signal
# ---------------------------------------------------------------------------#

def get_iv_signal(ticker: str, options_dir: Path) -> dict:
    """Return the most recent IV signal for the ranker.

    Args:
        ticker: Stock ticker symbol.
        options_dir: Root options directory.

    Returns:
        Dict with keys: ticker, implied_vol, historical_vol, iv_hv_spread,
        iv_spike, vol_signal.
        Returns neutral defaults when no data is stored.
    """
    base = {
        "ticker": ticker,
        "implied_vol": None,
        "historical_vol": None,
        "iv_hv_spread": None,
        "iv_spike": False,
        "vol_signal": 0.0,
    }
    df = load_iv_metrics(ticker, options_dir)
    if df.empty:
        return base

    latest = df.sort_index().iloc[-1]
    return {
        "ticker": ticker,
        "implied_vol": latest.get("implied_vol"),
        "historical_vol": latest.get("historical_vol"),
        "iv_hv_spread": latest.get("iv_hv_spread"),
        "iv_spike": bool(latest.get("iv_spike", False)),
        "vol_signal": float(latest.get("vol_signal", 0.0)),
    }
