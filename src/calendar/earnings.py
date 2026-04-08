"""Phase 7: Earnings calendar — upcoming dates and historical beat/miss record.

Data source: yfinance (Ticker.calendar for next earnings date,
             Ticker.earnings_history for historical EPS actuals vs estimates).

For each ticker this module provides:
  1. next_earnings_date      — when the next report is due (datetime UTC or None)
  2. days_to_earnings        — calendar days until that date (negative = past)
  3. earnings_approaching    — True when within settings.earnings_upcoming_days
  4. beat_rate               — fraction of last N quarters where EPS beat estimate
  5. avg_eps_surprise_pct    — average % surprise (positive = consistent beats)
  6. earnings_signal         — [-1, 1] composite: upcoming date risk + beat history

Signal logic:
  - A stock approaching earnings (< 14 days) with a strong beat history
    gets a mild positive boost (market often drifts up into earnings for
    consistent beaters).
  - Approaching earnings with a miss history is a mild negative flag.
  - Far from earnings: signal is driven purely by beat rate relative to 50%.

Formula:
  beat_component  = (beat_rate - 0.5) * 2  → [-1, 1]
  approach_factor = 1.0 if days_to_earnings <= 14 else 0.5
  earnings_signal = beat_component * approach_factor

Storage: data/calendar/earnings/{ticker}.parquet
Columns: next_earnings_date, days_to_earnings, earnings_approaching,
         beat_rate, avg_eps_surprise_pct, earnings_signal
Index:   DatetimeIndex (fetch date, UTC)
"""

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


# ---------------------------------------------------------------------------#
# Data extraction helpers
# ---------------------------------------------------------------------------#

def _get_next_earnings_date(ticker_obj: yf.Ticker) -> datetime | None:
    """Extract the next earnings date from yfinance Ticker.calendar.

    yfinance returns a dict with an "Earnings Date" key whose value is a
    list of Timestamps. We take the first future-or-present one.

    Args:
        ticker_obj: An instantiated yfinance Ticker object.

    Returns:
        UTC-aware datetime of the next earnings date, or None if unavailable.
    """
    try:
        cal = ticker_obj.calendar
        if cal is None:
            return None

        # cal may be a dict or a DataFrame depending on yfinance version
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
        elif isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"]
                dates = val.tolist() if hasattr(val, "tolist") else [val]
            else:
                return None
        else:
            return None

        now = datetime.now(tz=timezone.utc)
        for d in dates:
            if d is None:
                continue
            try:
                dt = pd.Timestamp(d).tz_localize("UTC") if pd.Timestamp(d).tzinfo is None else pd.Timestamp(d).tz_convert("UTC")
                dt_native = dt.to_pydatetime()
                if dt_native >= now:
                    return dt_native
            except Exception:
                continue
        return None
    except Exception as exc:
        logger.debug(f"Could not parse earnings calendar: {exc}")
        return None


def _get_beat_miss_history(
    ticker_obj: yf.Ticker,
    lookback_days: int,
) -> tuple[float | None, float | None]:
    """Compute beat rate and average EPS surprise from earnings history.

    Uses yfinance earnings_history (quarterly EPS actuals vs estimates).

    Args:
        ticker_obj: An instantiated yfinance Ticker object.
        lookback_days: Only count quarters within this many days.

    Returns:
        Tuple (beat_rate, avg_surprise_pct).
          beat_rate: fraction of beats [0, 1], or None if no data.
          avg_surprise_pct: mean % surprise, or None if no data.
    """
    try:
        hist = ticker_obj.earnings_history
        if hist is None or (isinstance(hist, pd.DataFrame) and hist.empty):
            return None, None

        if not isinstance(hist, pd.DataFrame):
            return None, None

        # Normalize column names (yfinance varies across versions)
        hist.columns = [str(c).strip() for c in hist.columns]

        # Find EPS actual and estimate columns
        actual_col = next((c for c in hist.columns if "actual" in c.lower()), None)
        est_col = next((c for c in hist.columns if "estimate" in c.lower()), None)
        if actual_col is None or est_col is None:
            return None, None

        # Filter by lookback window
        if hist.index.tz is None:
            hist.index = hist.index.tz_localize("UTC")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
        recent = hist[hist.index >= cutoff].copy()
        if recent.empty:
            return None, None

        actual = pd.to_numeric(recent[actual_col], errors="coerce")
        estimate = pd.to_numeric(recent[est_col], errors="coerce")

        valid = pd.DataFrame({"actual": actual, "estimate": estimate}).dropna()
        if valid.empty:
            return None, None

        beats = (valid["actual"] >= valid["estimate"]).sum()
        beat_rate = float(beats / len(valid))

        # Surprise %: (actual - estimate) / abs(estimate), capped at ±100%
        surprise_pct = ((valid["actual"] - valid["estimate"]) / valid["estimate"].abs().replace(0, float("nan"))) * 100.0
        avg_surprise = float(surprise_pct.dropna().mean()) if not surprise_pct.dropna().empty else None

        return beat_rate, avg_surprise

    except Exception as exc:
        logger.debug(f"Could not parse earnings history: {exc}")
        return None, None


# ---------------------------------------------------------------------------#
# Main compute function
# ---------------------------------------------------------------------------#

def compute_earnings_metrics(ticker: str) -> dict:
    """Fetch earnings calendar data and compute signal for a single ticker.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Dict with keys:
          - ticker (str)
          - fetch_time (datetime, UTC-aware)
          - next_earnings_date (datetime | None)
          - days_to_earnings (int | None)
          - earnings_approaching (bool)
          - beat_rate (float | None): fraction of beats in lookback window
          - avg_eps_surprise_pct (float | None)
          - earnings_signal (float): [-1, 1]
    """
    settings = get_settings()
    t = yf.Ticker(ticker)
    now = datetime.now(tz=timezone.utc)

    next_date = _get_next_earnings_date(t)
    beat_rate, avg_surprise = _get_beat_miss_history(t, settings.earnings_lookback_days)

    days_to_earnings: int | None = None
    earnings_approaching = False

    if next_date is not None:
        delta = (next_date - now).days
        days_to_earnings = delta
        earnings_approaching = 0 <= delta <= settings.earnings_upcoming_days

    # Signal computation
    beat_component = (beat_rate - 0.5) * 2.0 if beat_rate is not None else 0.0
    if days_to_earnings is not None and 0 <= days_to_earnings <= 14:
        approach_factor = 1.0
    else:
        approach_factor = 0.5

    earnings_signal = float(beat_component * approach_factor)

    result = {
        "ticker": ticker,
        "fetch_time": now,
        "next_earnings_date": next_date,
        "days_to_earnings": days_to_earnings,
        "earnings_approaching": earnings_approaching,
        "beat_rate": beat_rate,
        "avg_eps_surprise_pct": avg_surprise,
        "earnings_signal": earnings_signal,
    }

    logger.info(
        f"[{ticker}] next_earnings={next_date.date() if next_date else 'N/A'} "
        f"days={days_to_earnings} beat_rate={beat_rate:.0%} signal={earnings_signal:.3f}"
        if beat_rate is not None
        else f"[{ticker}] next_earnings={next_date.date() if next_date else 'N/A'} (no beat history)"
    )
    return result


def compute_batch_earnings_metrics(
    tickers: list[str],
    delay: float = 0.5,
) -> dict[str, dict]:
    """Compute earnings metrics for multiple tickers.

    Args:
        tickers: List of ticker symbols.
        delay: Seconds to sleep between requests.

    Returns:
        Dict mapping ticker -> metrics dict.
    """
    results: dict[str, dict] = {}
    for i, ticker in enumerate(tickers):
        try:
            results[ticker] = compute_earnings_metrics(ticker)
        except Exception as exc:
            logger.error(f"[{ticker}] Earnings metrics failed: {exc}")
            results[ticker] = {"ticker": ticker, "error": str(exc)}
        if delay > 0 and i < len(tickers) - 1:
            time.sleep(delay)
    return results


# ---------------------------------------------------------------------------#
# Persistence
# ---------------------------------------------------------------------------#

def save_earnings_metrics(metrics: dict, calendar_dir: Path) -> None:
    """Persist earnings metrics to data/calendar/earnings/{ticker}.parquet.

    Args:
        metrics: Output of :func:`compute_earnings_metrics`.
        calendar_dir: Root calendar storage directory.
    """
    if "error" in metrics:
        return

    earnings_dir = calendar_dir / "earnings"
    earnings_dir.mkdir(parents=True, exist_ok=True)
    ticker = metrics["ticker"]
    filepath = earnings_dir / f"{ticker}.parquet"

    next_date_str = metrics["next_earnings_date"].isoformat() if metrics["next_earnings_date"] else None

    row = {
        "next_earnings_date": next_date_str,
        "days_to_earnings": metrics["days_to_earnings"],
        "earnings_approaching": metrics["earnings_approaching"],
        "beat_rate": metrics["beat_rate"],
        "avg_eps_surprise_pct": metrics["avg_eps_surprise_pct"],
        "earnings_signal": metrics["earnings_signal"],
    }
    df = pd.DataFrame([row], index=pd.DatetimeIndex([metrics["fetch_time"]], name="date"))
    upsert_dataframe(df, filepath)
    logger.debug(f"[{ticker}] Earnings metrics saved")


def load_earnings_metrics(ticker: str, calendar_dir: Path) -> pd.DataFrame:
    """Load historical earnings metrics for a ticker.

    Args:
        ticker: Stock ticker symbol.
        calendar_dir: Root calendar storage directory.

    Returns:
        DataFrame with DatetimeIndex (fetch dates) and columns:
        next_earnings_date, days_to_earnings, earnings_approaching,
        beat_rate, avg_eps_surprise_pct, earnings_signal.
        Empty DataFrame if no data stored.
    """
    filepath = calendar_dir / "earnings" / f"{ticker}.parquet"
    return load_dataframe(filepath)


# ---------------------------------------------------------------------------#
# Signal
# ---------------------------------------------------------------------------#

def get_earnings_signal(ticker: str, calendar_dir: Path) -> dict:
    """Return the most recent earnings signal for the ranker.

    Args:
        ticker: Stock ticker symbol.
        calendar_dir: Root calendar storage directory.

    Returns:
        Dict with keys: ticker, next_earnings_date, days_to_earnings,
        earnings_approaching, beat_rate, avg_eps_surprise_pct, earnings_signal.
    """
    base = {
        "ticker": ticker,
        "next_earnings_date": None,
        "days_to_earnings": None,
        "earnings_approaching": False,
        "beat_rate": None,
        "avg_eps_surprise_pct": None,
        "earnings_signal": 0.0,
    }
    df = load_earnings_metrics(ticker, calendar_dir)
    if df.empty:
        return base

    latest = df.sort_index().iloc[-1]
    return {
        "ticker": ticker,
        "next_earnings_date": latest.get("next_earnings_date"),
        "days_to_earnings": latest.get("days_to_earnings"),
        "earnings_approaching": bool(latest.get("earnings_approaching", False)),
        "beat_rate": latest.get("beat_rate"),
        "avg_eps_surprise_pct": latest.get("avg_eps_surprise_pct"),
        "earnings_signal": float(latest.get("earnings_signal", 0.0)),
    }
