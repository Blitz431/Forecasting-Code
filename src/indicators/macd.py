from __future__ import annotations

import pandas as pd

from src.indicators._calc import macd as _macd
from src.indicators.base import Indicator, IndicatorResult, Signal

"""
Purpose: MACD indicator (12-26-9) — momentum signal from EMA crossover and histogram direction.

Connections:
  - src/indicators/_calc.py: macd() math function
  - src/indicators/base.py: Indicator ABC, IndicatorResult, Signal types
  - src/indicators/signal_aggregator.py: called with weight 2.0 (highest weight)

In:  daily OHLCV pd.DataFrame (needs Close, min ~35 rows)
Out: IndicatorResult with Signal enum and MACD/signal/histogram values
"""


class MACDIndicator(Indicator):
    """Moving Average Convergence Divergence (12, 26, 9)."""

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    @property
    def name(self) -> str:
        return f"MACD ({self.fast},{self.slow},{self.signal_period})"

    @property
    def weight(self) -> float:
        return 2.0

    def compute(self, df: pd.DataFrame) -> IndicatorResult:
        macd_line, signal_line, histogram = _macd(
            df["Close"], self.fast, self.slow, self.signal_period
        )

        # Current values
        cur_macd = float(macd_line.dropna().iloc[-1])
        cur_sig = float(signal_line.dropna().iloc[-1])
        cur_hist = float(histogram.dropna().iloc[-1])

        # Previous histogram value to detect expansion/contraction
        hist_clean = histogram.dropna()
        prev_hist = float(hist_clean.iloc[-2]) if len(hist_clean) >= 2 else cur_hist
        hist_expanding = abs(cur_hist) > abs(prev_hist)

        # Crossover detection (last 2 bars)
        macd_clean = macd_line.dropna()
        sig_clean = signal_line.dropna()
        if len(macd_clean) >= 2 and len(sig_clean) >= 2:
            bullish_cross = (
                macd_clean.iloc[-2] < sig_clean.iloc[-2]
                and macd_clean.iloc[-1] >= sig_clean.iloc[-1]
            )
            bearish_cross = (
                macd_clean.iloc[-2] > sig_clean.iloc[-2]
                and macd_clean.iloc[-1] <= sig_clean.iloc[-1]
            )
        else:
            bullish_cross = bearish_cross = False

        # Signal logic
        if bullish_cross and hist_expanding:
            signal_val = Signal.STRONG_BUY
            interp = f"Bullish crossover + histogram expanding (MACD {cur_macd:+.2f})"
        elif cur_macd > cur_sig and cur_hist > 0:
            signal_val = Signal.BUY
            interp = f"MACD above signal line, positive momentum (MACD {cur_macd:+.2f})"
        elif bearish_cross and hist_expanding:
            signal_val = Signal.STRONG_SELL
            interp = f"Bearish crossover + histogram expanding (MACD {cur_macd:+.2f})"
        elif cur_macd < cur_sig and cur_hist < 0:
            signal_val = Signal.SELL
            interp = f"MACD below signal line, negative momentum (MACD {cur_macd:+.2f})"
        else:
            signal_val = Signal.NEUTRAL
            interp = f"MACD near signal line, no clear direction (MACD {cur_macd:+.2f})"

        return IndicatorResult(
            indicator_name=self.name,
            signal=signal_val,
            value=round(cur_macd, 4),
            details={
                "macd": round(cur_macd, 4),
                "signal": round(cur_sig, 4),
                "histogram": round(cur_hist, 4),
                "bullish_cross": bullish_cross,
                "bearish_cross": bearish_cross,
            },
            interpretation=interp,
        )
