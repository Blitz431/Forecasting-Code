"""Momentum indicator — fast (4-week) vs slow (12-week) Rate of Change.

Uses ROC (Rate of Change) to measure short-term vs medium-term price momentum.

Signal logic
------------
Both fast and slow ROC positive, fast > slow  → STRONG_BUY  (accelerating uptrend)
Both positive                                  → BUY
Mixed signals                                  → NEUTRAL
Both negative                                  → SELL
Both negative, fast < slow (accelerating down) → STRONG_SELL
"""

from __future__ import annotations

import pandas as pd

from src.indicators._calc import roc
from src.indicators.base import Indicator, IndicatorResult, Signal

_FAST_WINDOW = 20   # ~4 trading weeks
_SLOW_WINDOW = 60   # ~12 trading weeks (one quarter)


class MomentumIndicator(Indicator):
    """Fast vs slow momentum using Rate of Change."""

    def __init__(
        self,
        fast_window: int = _FAST_WINDOW,
        slow_window: int = _SLOW_WINDOW,
    ) -> None:
        self.fast_window = fast_window
        self.slow_window = slow_window

    @property
    def name(self) -> str:
        return f"Momentum ({self.fast_window}d/{self.slow_window}d ROC)"

    @property
    def weight(self) -> float:
        return 1.5

    def compute(self, df: pd.DataFrame) -> IndicatorResult:
        fast = roc(df["Close"], self.fast_window)
        slow = roc(df["Close"], self.slow_window)

        cur_fast = float(fast.dropna().iloc[-1])
        cur_slow = float(slow.dropna().iloc[-1])

        accelerating_up = cur_fast > 0 and cur_slow > 0 and cur_fast > cur_slow
        decelerating_down = cur_fast < 0 and cur_slow < 0 and cur_fast < cur_slow

        if accelerating_up:
            signal_val = Signal.STRONG_BUY
            interp = (
                f"Accelerating uptrend — fast ROC {cur_fast:+.1%}, "
                f"slow ROC {cur_slow:+.1%}"
            )
        elif cur_fast > 0 and cur_slow > 0:
            signal_val = Signal.BUY
            interp = (
                f"Positive momentum — fast ROC {cur_fast:+.1%}, "
                f"slow ROC {cur_slow:+.1%}"
            )
        elif decelerating_down:
            signal_val = Signal.STRONG_SELL
            interp = (
                f"Accelerating downtrend — fast ROC {cur_fast:+.1%}, "
                f"slow ROC {cur_slow:+.1%}"
            )
        elif cur_fast < 0 and cur_slow < 0:
            signal_val = Signal.SELL
            interp = (
                f"Negative momentum — fast ROC {cur_fast:+.1%}, "
                f"slow ROC {cur_slow:+.1%}"
            )
        else:
            signal_val = Signal.NEUTRAL
            interp = (
                f"Mixed momentum — fast ROC {cur_fast:+.1%}, "
                f"slow ROC {cur_slow:+.1%}"
            )

        return IndicatorResult(
            indicator_name=self.name,
            signal=signal_val,
            value=round(cur_fast * 100, 2),
            details={
                "fast_roc_pct": round(cur_fast * 100, 2),
                "slow_roc_pct": round(cur_slow * 100, 2),
                "fast_window": self.fast_window,
                "slow_window": self.slow_window,
            },
            interpretation=interp,
        )
