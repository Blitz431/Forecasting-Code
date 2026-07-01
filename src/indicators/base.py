from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum

"""
Purpose: Abstract base classes and data structures for all 9 technical indicators (Signal enum, IndicatorResult, Indicator ABC).

Connections:
  - src/indicators/rsi.py, macd.py, moving_averages.py, bollinger.py, stochastic.py,
    volume.py, momentum.py, fundamentals.py, correlation.py: all subclass Indicator
  - src/indicators/signal_aggregator.py: consumes IndicatorResult list to build composite score

In:  daily OHLCV pd.DataFrame
Out: IndicatorResult dataclass (signal: Signal enum -2..+2, value, details, interpretation)
"""


class Signal(IntEnum):
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2

    def label(self) -> str:
        return {
            2: "STRONG BUY",
            1: "BUY",
            0: "NEUTRAL",
            -1: "SELL",
            -2: "STRONG SELL",
        }[self.value]

    def emoji(self) -> str:
        return {
            2: "++",
            1: "+",
            0: "~",
            -1: "-",
            -2: "--",
        }[self.value]


@dataclass
class IndicatorResult:
    """Output from a single indicator computation.

    Attributes:
        indicator_name: Display name (e.g. "RSI (14)").
        signal: Typed :class:`Signal` enum value.
        value: Primary numeric value for this indicator (e.g. RSI = 45.2).
        details: Dict of additional numeric outputs (e.g. for MACD: line, signal, hist).
        interpretation: One-line human-readable explanation of the signal.
        error: Set if computation raised an exception; None otherwise.
    """

    indicator_name: str
    signal: Signal
    value: float | None = None
    details: dict = field(default_factory=dict)
    interpretation: str = ""
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "indicator": self.indicator_name,
            "signal": self.signal.label(),
            "signal_value": int(self.signal),
            "value": self.value,
            "interpretation": self.interpretation,
            "error": self.error,
        }


class Indicator(ABC):
    """Abstract base for all technical indicators.

    Subclasses implement :meth:`compute`.  The shared :meth:`safe_compute`
    wrapper catches exceptions and returns an error result so the aggregator
    never crashes on a single bad indicator.

    Expected input DataFrame columns: Open, High, Low, Close, Volume
    (daily OHLCV, DatetimeIndex).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable indicator name shown in reports."""
        ...

    @property
    def weight(self) -> float:
        """Relative weight used by the signal aggregator (default 1.0)."""
        return 1.0

    @abstractmethod
    def compute(self, df) -> IndicatorResult:
        """Compute the indicator on *df* and return an :class:`IndicatorResult`.

        Args:
            df: Daily OHLCV DataFrame with DatetimeIndex, sorted ascending.
                Must contain at least: Close, High, Low, Volume.

        Returns:
            :class:`IndicatorResult` with signal and values set.
        """
        ...

    def safe_compute(self, df) -> IndicatorResult:
        """Compute with exception handling — never raises."""
        try:
            return self.compute(df)
        except Exception as exc:
            return IndicatorResult(
                indicator_name=self.name,
                signal=Signal.NEUTRAL,
                interpretation=f"Error: {exc}",
                error=str(exc),
            )
