"""Composite ranking engine — pulls from all upstream signal sources.

Aggregates signals from every Phase (2-7) module into a single normalised
composite score per ticker, then returns the top-N picks.

Signal sources and weights
--------------------------
| Signal            | Source                        | Weight |
|-------------------|-------------------------------|--------|
| forecast_signal   | data/forecasts/ (best method) |  2.0   |
| indicator_score   | indicators/signal_aggregator  |  2.0   |
| ml_signal         | ml/runner.predict_latest()    |  2.0   |
| news_sentiment    | news/aggregator.get_summary() |  1.0   |
| short_interest    | news/short_interest           |  0.75  |
| congress_signal   | political/congress_tracker    |  1.0   |
| insider_signal    | political/insider_tracker     |  1.0   |
| options_flow      | options/options_data          |  1.0   |
| iv_signal         | options/implied_vol           |  0.5   |
| earnings_signal   | calendar/earnings             |  0.5   |

All signals are normalised to [-1, +1] before weighting.
Missing signals (no data yet) are excluded from the weighted average.

Public API
----------
rank_tickers(tickers, settings, include_ml, include_indicators)
    -> list[RankEntry]          (sorted desc by composite_score)

top_picks(n, settings, include_ml, include_indicators)
    -> list[RankEntry]          (top-N from the full universe)

to_dataframe(entries)
    -> pd.DataFrame             (display-ready table)
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import get_settings
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------#
# Weights
# ---------------------------------------------------------------------------#

_WEIGHTS: dict[str, float] = {
    "forecast_signal":      2.0,
    "indicator_score":      2.0,
    "ml_signal":            2.0,
    "news_sentiment":       1.0,
    "short_interest":       0.75,
    "congress_signal":      1.0,
    "insider_signal":       1.0,
    "options_flow":         1.0,
    "iv_signal":            0.5,
    "earnings_signal":      0.5,
}


# ---------------------------------------------------------------------------#
# Data class
# ---------------------------------------------------------------------------#

@dataclass
class RankEntry:
    """Composite ranking result for a single ticker.

    Attributes:
        ticker:            Stock symbol.
        composite_score:   Weighted average of all available signals, in [-1, +1].
        rank:              Position in the sorted list (1 = best).
        signals:           Dict of signal_name -> float value (only present signals).
        signals_available: How many signals contributed to the composite score.
        details:           Extra per-signal context dicts (for dashboard drill-down).
    """

    ticker: str
    composite_score: float
    rank: int = 0
    signals: dict[str, float] = field(default_factory=dict)
    signals_available: int = 0
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        row = {
            "Rank":             self.rank,
            "Ticker":           self.ticker,
            "Score":            round(self.composite_score, 4),
            "Signals":          self.signals_available,
        }
        for name in _WEIGHTS:
            row[_signal_label(name)] = round(self.signals.get(name, float("nan")), 3)
        return row


def _signal_label(name: str) -> str:
    """Convert snake_case signal name to a short display label."""
    mapping = {
        "forecast_signal":   "Forecast",
        "indicator_score":   "Indicators",
        "ml_signal":         "ML",
        "news_sentiment":    "News",
        "short_interest":    "Short Int.",
        "congress_signal":   "Congress",
        "insider_signal":    "Insider",
        "options_flow":      "Options",
        "iv_signal":         "IV",
        "earnings_signal":   "Earnings",
    }
    return mapping.get(name, name)


# ---------------------------------------------------------------------------#
# Individual signal extractors
# ---------------------------------------------------------------------------#

def _safe(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), return None on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.debug(f"Signal extractor {fn.__name__} failed: {exc}")
        return None


def _get_forecast_signal(ticker: str, settings) -> float | None:
    """Compute forecast upside % vs current price → [-1, +1]."""
    forecast_path = settings.forecasts_dir / f"{ticker}_forecasts.parquet"
    if not forecast_path.exists():
        return None

    try:
        df = pd.read_parquet(forecast_path)
    except Exception:
        return None

    if df.empty or "Forecast_Price" not in df.columns:
        return None

    # Pick the best method row (lowest RMSE) for each forecast date
    if "RMSE" in df.columns:
        df = df.sort_values("RMSE")
        df = df.drop_duplicates(subset=["Forecast_Date"], keep="first")

    # Use the nearest future forecast date
    now = pd.Timestamp.now()
    if "Forecast_Date" in df.columns:
        df["Forecast_Date"] = pd.to_datetime(df["Forecast_Date"])
        future = df[df["Forecast_Date"] > now].sort_values("Forecast_Date")
    else:
        future = df.tail(1)

    if future.empty:
        # Fall back to the last forecast in the file
        future = df.sort_values("Forecast_Date").tail(1)

    forecast_price = float(future.iloc[0]["Forecast_Price"])

    # Current price from daily parquet
    daily_path = settings.raw_daily_dir / f"{ticker}.parquet"
    if not daily_path.exists():
        return None

    from src.scraper.storage import load_dataframe
    df_daily = load_dataframe(daily_path)
    if df_daily.empty or "Close" not in df_daily.columns:
        return None

    current_price = float(df_daily["Close"].dropna().iloc[-1])
    if current_price <= 0:
        return None

    upside = (forecast_price - current_price) / current_price
    return float(np.clip(upside / 0.30, -1.0, 1.0))


def _get_indicator_score(ticker: str, settings) -> float | None:
    """Run signal aggregator on latest daily data → score_normalized in [-1, +1]."""
    from src.indicators.signal_aggregator import run_and_aggregate
    from src.scraper.storage import get_ticker_filepath, load_dataframe

    filepath = get_ticker_filepath(ticker, settings.raw_daily_dir)
    df = load_dataframe(filepath)
    if df.empty:
        return None

    result = run_and_aggregate(df, ticker)
    if result.succeeded == 0:
        return None
    return float(result.score_normalized)


def _get_ml_signal(ticker: str, settings) -> float | None:
    """Call predict_latest() with XGBoost → normalized predicted 1-day return."""
    from src.ml.runner import predict_latest

    predicted_return = predict_latest(
        ticker, model_name="XGBoost", settings=settings, target_days=1
    )
    if predicted_return is None:
        return None
    # Clip at ±5% daily return = signal ±1.0
    return float(np.clip(predicted_return / 0.05, -1.0, 1.0))


def _get_news_sentiment(ticker: str, settings) -> float | None:
    """7-day rolling sentiment score from FinBERT-scored articles."""
    from src.news.aggregator import get_sentiment_summary

    summary = get_sentiment_summary(
        ticker,
        settings.news_articles_dir,
        days=settings.news_sentiment_window_days,
    )
    if summary.get("article_count", 0) == 0:
        return None
    return float(np.clip(summary.get("avg_score", 0.0), -1.0, 1.0))


def _get_short_interest_signal(ticker: str, settings) -> float | None:
    """Short days-to-cover → negative signal (high short interest = bearish)."""
    from src.news.short_interest import get_short_interest_signal

    result = get_short_interest_signal(ticker, settings.news_short_interest_dir)
    ratio = result.get("short_ratio")
    if ratio is None:
        return None
    # Days-to-cover > 5 = high. Scale: 0 days = 0, 10 days = -1.0
    return float(np.clip(-ratio / 10.0, -1.0, 0.0))


def _get_congress_signal(ticker: str, settings) -> float | None:
    """Congressional trading net flow → [-1, +1]."""
    from src.political.congress_tracker import get_congress_signal

    result = get_congress_signal(ticker, settings.political_congress_dir)
    sig = result.get("congress_signal", 0.0)
    # Only return a signal if there was actual activity
    if result.get("congress_net_buys", 0) == 0 and result.get("congress_net_sells", 0) == 0:
        return None
    return float(np.clip(sig, -1.0, 1.0))


def _get_insider_signal(ticker: str, settings) -> float | None:
    """Insider filing net flow → [-1, +1]."""
    from src.political.insider_tracker import get_insider_signal

    result = get_insider_signal(ticker, settings.political_insider_dir)
    sig = result.get("insider_signal", 0.0)
    if result.get("insider_net_buys", 0) == 0 and result.get("insider_net_sells", 0) == 0:
        return None
    return float(np.clip(sig, -1.0, 1.0))


def _get_options_flow(ticker: str, settings) -> float | None:
    """Options put/call flow signal → [-1, +1]."""
    from src.options.options_data import get_options_signal

    result = get_options_signal(ticker, settings.options_dir)
    sig = result.get("flow_signal", 0.0)
    if result.get("total_call_volume", 0) == 0 and result.get("total_put_volume", 0) == 0:
        return None
    return float(np.clip(sig, -1.0, 1.0))


def _get_iv_signal(ticker: str, settings) -> float | None:
    """Implied vol vs historical vol → [-1, +1] (spike = risk-off = negative)."""
    from src.options.implied_vol import get_iv_signal

    result = get_iv_signal(ticker, settings.options_dir)
    if result.get("implied_vol") is None:
        return None
    return float(np.clip(result.get("vol_signal", 0.0), -1.0, 1.0))


def _get_earnings_signal(ticker: str, settings) -> float | None:
    """Earnings beat rate + upcoming catalyst → [-1, +1]."""
    from src.calendar.earnings import get_earnings_signal

    result = get_earnings_signal(ticker, settings.calendar_dir)
    if result.get("beat_rate") is None and not result.get("earnings_approaching", False):
        return None
    return float(np.clip(result.get("earnings_signal", 0.0), -1.0, 1.0))


# ---------------------------------------------------------------------------#
# Composite score
# ---------------------------------------------------------------------------#

def _compute_composite(signals: dict[str, float | None]) -> tuple[float, int]:
    """Weighted average of non-None signals → (composite_score, n_signals)."""
    weighted_sum = 0.0
    total_weight = 0.0
    n = 0

    for name, value in signals.items():
        if value is None:
            continue
        weight = _WEIGHTS.get(name, 1.0)
        weighted_sum += float(value) * weight
        total_weight += weight
        n += 1

    if total_weight == 0.0:
        return 0.0, 0

    composite = weighted_sum / total_weight
    return float(np.clip(composite, -1.0, 1.0)), n


# ---------------------------------------------------------------------------#
# Per-ticker scoring
# ---------------------------------------------------------------------------#

def _score_ticker(
    ticker: str,
    settings,
    include_ml: bool = False,
    include_indicators: bool = True,
) -> RankEntry:
    """Collect all signals and compute a composite score for *ticker*."""

    # Gather all signals (safe-wrapped)
    forecast   = _safe(_get_forecast_signal,       ticker, settings)
    indicators = _safe(_get_indicator_score,        ticker, settings) if include_indicators else None
    ml         = _safe(_get_ml_signal,              ticker, settings) if include_ml else None
    news       = _safe(_get_news_sentiment,         ticker, settings)
    short      = _safe(_get_short_interest_signal,  ticker, settings)
    congress   = _safe(_get_congress_signal,        ticker, settings)
    insider    = _safe(_get_insider_signal,         ticker, settings)
    options    = _safe(_get_options_flow,           ticker, settings)
    iv         = _safe(_get_iv_signal,              ticker, settings)
    earnings   = _safe(_get_earnings_signal,        ticker, settings)

    raw_signals: dict[str, float | None] = {
        "forecast_signal":  forecast,
        "indicator_score":  indicators,
        "ml_signal":        ml,
        "news_sentiment":   news,
        "short_interest":   short,
        "congress_signal":  congress,
        "insider_signal":   insider,
        "options_flow":     options,
        "iv_signal":        iv,
        "earnings_signal":  earnings,
    }

    composite, n = _compute_composite(raw_signals)

    present_signals = {k: v for k, v in raw_signals.items() if v is not None}

    return RankEntry(
        ticker=ticker,
        composite_score=composite,
        signals=present_signals,
        signals_available=n,
        details={
            "forecast_raw": forecast,
            "ml_raw":       ml,
        },
    )


# ---------------------------------------------------------------------------#
# Public API
# ---------------------------------------------------------------------------#

def rank_tickers(
    tickers: list[str],
    settings=None,
    include_ml: bool = False,
    include_indicators: bool = True,
) -> list[RankEntry]:
    """Score and rank a list of tickers.

    Args:
        tickers:            Ticker symbols to rank.
        settings:           Optional settings override.
        include_ml:         If True, call ``predict_latest()`` per ticker
                            (much slower — only use for small lists or overnight runs).
        include_indicators: If True, run the full indicator pipeline per ticker.

    Returns:
        List of :class:`RankEntry` sorted descending by composite score.
        ``rank`` attribute is set to 1-based position.
    """
    if settings is None:
        settings = get_settings()

    entries: list[RankEntry] = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"[{i:>3}/{total}] Scoring {ticker} …")
        try:
            entry = _score_ticker(
                ticker, settings,
                include_ml=include_ml,
                include_indicators=include_indicators,
            )
            entries.append(entry)
        except Exception as exc:
            logger.warning(f"[{ticker}] score failed: {exc}\n{traceback.format_exc()}")
            entries.append(RankEntry(ticker=ticker, composite_score=float("-inf")))

    # Sort descending by composite score
    entries.sort(key=lambda e: e.composite_score, reverse=True)

    # Assign ranks (skip −inf placeholder entries)
    rank = 1
    for e in entries:
        if e.composite_score != float("-inf"):
            e.rank = rank
            rank += 1

    return entries


def top_picks(
    n: int | None = None,
    settings=None,
    include_ml: bool = False,
    include_indicators: bool = True,
    tickers: list[str] | None = None,
) -> list[RankEntry]:
    """Rank the full ticker universe and return the top-N picks.

    Args:
        n:                  Number of top picks to return.
                            Defaults to ``settings.top_n_picks`` (20).
        settings:           Optional settings override.
        include_ml:         Enable ML signal (slow).
        include_indicators: Enable indicator pipeline.
        tickers:            Custom list. Defaults to full S&P 500 universe.

    Returns:
        Top-N :class:`RankEntry` objects sorted by composite score (best first).
    """
    if settings is None:
        settings = get_settings()
    if n is None:
        n = settings.top_n_picks

    if tickers is None:
        from src.utils.tickers import get_tickers
        tickers = get_tickers(settings.ticker_source)

    # Only score tickers that have daily data (skip unscraped ones)
    available = [
        t for t in tickers
        if (settings.raw_daily_dir / f"{t}.parquet").exists()
    ]

    if not available:
        logger.warning(
            "No ticker data found in %s. "
            "Run `cli/scrape.py` first.",
            settings.raw_daily_dir,
        )
        return []

    logger.info(
        f"Ranking {len(available)} tickers "
        f"(ML={'on' if include_ml else 'off'}, "
        f"indicators={'on' if include_indicators else 'off'}) …"
    )

    all_entries = rank_tickers(
        available,
        settings=settings,
        include_ml=include_ml,
        include_indicators=include_indicators,
    )

    return [e for e in all_entries if e.composite_score > float("-inf")][:n]


def to_dataframe(entries: list[RankEntry]) -> pd.DataFrame:
    """Convert a list of RankEntry objects to a display-ready DataFrame."""
    if not entries:
        return pd.DataFrame()
    return pd.DataFrame([e.to_dict() for e in entries])


# ---------------------------------------------------------------------------#
# Ranking cache (parquet)
# ---------------------------------------------------------------------------#

_CACHE_FILENAME = "ranking_cache.parquet"


def save_ranking_cache(
    entries: list[RankEntry],
    settings=None,
    current_prices: dict | None = None,
    ml_targets: dict | None = None,
    forecast_targets: dict | None = None,
) -> Path:
    """Persist ranking results to data/ranking_cache.parquet.

    Each row is one ticker. Columns include composite score, all individual
    signal values, and optional price targets from the pipeline.

    Returns the path the file was written to.
    """
    if settings is None:
        settings = get_settings()

    rows = []
    now = pd.Timestamp.now(tz="UTC")
    for e in entries:
        row: dict = {
            "ticker":            e.ticker,
            "rank":              e.rank,
            "composite_score":   round(e.composite_score, 6),
            "signals_available": e.signals_available,
            "updated_at":        now,
        }
        for sig in _WEIGHTS:
            val = e.signals.get(sig)
            row[sig] = float(val) if val is not None else float("nan")

        if current_prices:
            row["current_price"] = current_prices.get(e.ticker)
        if ml_targets:
            ml_p, ml_d = ml_targets.get(e.ticker, (None, None))
            row["ml_target"]      = ml_p
            row["ml_target_date"] = ml_d
        if forecast_targets:
            fc_p, fc_d = forecast_targets.get(e.ticker, (None, None))
            row["fcst_target"] = fc_p
            row["fcst_date"]   = fc_d

        rows.append(row)

    df = pd.DataFrame(rows).set_index("ticker")
    path = settings.data_dir / _CACHE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info(f"Ranking cache saved → {path} ({len(rows)} tickers)")
    return path


def load_ranking_cache(settings=None) -> list[RankEntry] | None:
    """Load cached ranking results from data/ranking_cache.parquet.

    Returns a list of RankEntry objects (preserving rank order), or None
    if no cache file exists.
    """
    if settings is None:
        settings = get_settings()

    path = settings.data_dir / _CACHE_FILENAME
    if not path.exists():
        return None

    try:
        df = pd.read_parquet(path).reset_index()
    except Exception as exc:
        logger.warning(f"Could not read ranking cache: {exc}")
        return None

    entries: list[RankEntry] = []
    for _, row in df.iterrows():
        sigs = {}
        for sig in _WEIGHTS:
            val = row.get(sig)
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                sigs[sig] = float(val)

        entries.append(RankEntry(
            ticker=str(row["ticker"]),
            composite_score=float(row["composite_score"]),
            rank=int(row["rank"]),
            signals=sigs,
            signals_available=int(row.get("signals_available", len(sigs))),
        ))

    entries.sort(key=lambda e: e.rank)
    logger.info(f"Ranking cache loaded from {path} ({len(entries)} tickers)")
    return entries
