from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.indicators.base import Indicator, IndicatorResult, Signal
from src.indicators.bollinger import BollingerBandsIndicator
from src.indicators.correlation import CorrelationIndicator
from src.indicators.fundamentals import FundamentalsIndicator
from src.indicators.macd import MACDIndicator
from src.indicators.momentum import MomentumIndicator
from src.indicators.moving_averages import MovingAveragesIndicator
from src.indicators.rsi import RSIIndicator
from src.indicators.stochastic import StochasticIndicator
from src.indicators.volume import VolumeIndicator
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Combine all 9 indicator results into a weighted composite signal score for a ticker.

Connections:
  - src/indicators/rsi.py, macd.py, moving_averages.py, bollinger.py, stochastic.py,
    volume.py, momentum.py, fundamentals.py, correlation.py: instantiates and runs each
  - src/indicators/base.py: Indicator, IndicatorResult, Signal types
  - cli/indicators.py: calls run_and_aggregate() for each ticker
  - src/ranking/ranker.py: reads indicator parquets (score, signal) as one ranking signal

In:  daily OHLCV pd.DataFrame + ticker string
Out: AggregateResult (composite signal -2..+2, normalized score -1..+1, per-indicator breakdown)
"""

# Weights mirror indicator.weight properties but are listed here for clarity
_DEFAULT_WEIGHTS = {
    "RSI (14)": 1.5,
    "MACD (12,26,9)": 2.0,
    "Moving Averages (50/200)": 2.0,
    "Bollinger Bands (20, 2std)": 1.0,
    "Stochastic (14,3)": 1.0,
    "Volume Analysis": 1.0,
    "Momentum (20d/60d ROC)": 1.5,
    "Fundamentals (P/S + R/R)": 0.75,
    "Correlation (price/vol, 20d)": 0.75,
}


@dataclass
class AggregateResult:
    """Composite result across all indicators for one ticker.

    Attributes:
        ticker: Stock symbol.
        signal: Final composite :class:`Signal`.
        score: Raw weighted average in [−2, +2].
        score_normalized: Score normalised to [−1, +1] for display.
        results: Individual :class:`IndicatorResult` for each indicator.
        succeeded: Number of indicators that computed successfully.
        failed: Number that raised errors.
    """

    ticker: str
    signal: Signal
    score: float
    score_normalized: float
    results: list[IndicatorResult] = field(default_factory=list)
    succeeded: int = 0
    failed: int = 0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "signal": self.signal.label(),
            "signal_value": int(self.signal),
            "score": round(self.score, 3),
            "score_normalized": round(self.score_normalized, 3),
            "succeeded": self.succeeded,
            "failed": self.failed,
        }


def _build_indicators(ticker: str) -> list[Indicator]:
    """Instantiate all indicator objects for *ticker*."""
    return [
        RSIIndicator(),
        MACDIndicator(),
        MovingAveragesIndicator(),
        BollingerBandsIndicator(),
        StochasticIndicator(),
        VolumeIndicator(),
        MomentumIndicator(),
        FundamentalsIndicator(ticker),
        CorrelationIndicator(),
    ]


def run_all_indicators(
    df: pd.DataFrame,
    ticker: str,
) -> list[IndicatorResult]:
    """Run every indicator against *df* for *ticker*.

    Args:
        df: Daily OHLCV DataFrame with DatetimeIndex, sorted ascending.
        ticker: Stock symbol (needed by FundamentalsIndicator).

    Returns:
        List of :class:`IndicatorResult`, one per indicator (9 total).
    """
    indicators = _build_indicators(ticker)
    results: list[IndicatorResult] = []

    for ind in indicators:
        result = ind.safe_compute(df)
        if result.error:
            logger.warning(f"[{ticker}] {ind.name}: {result.error}")
        else:
            logger.debug(
                f"[{ticker}] {ind.name:40s}  "
                f"{result.signal.label():12s}  {result.interpretation[:60]}"
            )
        results.append(result)

    return results


def aggregate(
    results: list[IndicatorResult],
    ticker: str = "",
) -> AggregateResult:
    """Compute weighted composite signal from a list of indicator results.

    Failed indicators (``error`` set) are excluded from the weighted average.

    Args:
        results: Output of :func:`run_all_indicators`.
        ticker: Ticker symbol for the result object.

    Returns:
        :class:`AggregateResult` with composite signal and score.
    """
    weighted_sum = 0.0
    total_weight = 0.0
    succeeded = 0
    failed = 0

    for r in results:
        if r.error is not None:
            failed += 1
            continue
        w = _DEFAULT_WEIGHTS.get(r.indicator_name, 1.0)
        weighted_sum += int(r.signal) * w
        total_weight += w
        succeeded += 1

    if total_weight == 0:
        return AggregateResult(
            ticker=ticker,
            signal=Signal.NEUTRAL,
            score=0.0,
            score_normalized=0.0,
            results=results,
            succeeded=0,
            failed=failed,
        )

    score = weighted_sum / total_weight                 # in [−2, +2]
    score_normalized = score / 2.0                     # in [−1, +1]

    # Map score to Signal enum
    if score >= 1.5:
        final_signal = Signal.STRONG_BUY
    elif score >= 0.5:
        final_signal = Signal.BUY
    elif score <= -1.5:
        final_signal = Signal.STRONG_SELL
    elif score <= -0.5:
        final_signal = Signal.SELL
    else:
        final_signal = Signal.NEUTRAL

    return AggregateResult(
        ticker=ticker,
        signal=final_signal,
        score=round(score, 3),
        score_normalized=round(score_normalized, 3),
        results=results,
        succeeded=succeeded,
        failed=failed,
    )


def run_and_aggregate(
    df: pd.DataFrame,
    ticker: str,
) -> AggregateResult:
    """Convenience wrapper: run all indicators + aggregate in one call."""
    results = run_all_indicators(df, ticker)
    return aggregate(results, ticker)
