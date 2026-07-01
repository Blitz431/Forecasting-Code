from __future__ import annotations

import pandas as pd

from config.settings import get_settings
from src.indicators._calc import stochastic as _stoch
from src.indicators.base import Indicator, IndicatorResult, Signal

"""
Purpose: Stochastic Oscillator indicator (14, 3) — overbought/oversold with %K/%D cross confirmation.

Connections:
  - src/indicators/_calc.py: stochastic() math function
  - src/indicators/base.py: Indicator ABC, IndicatorResult, Signal types
  - config/settings.py: stochastic_overbought (80), stochastic_oversold (20) thresholds
  - src/indicators/signal_aggregator.py: called with weight 1.0

In:  daily OHLCV pd.DataFrame (needs High/Low/Close, min ~20 rows)
Out: IndicatorResult with Signal enum and %K/%D values
"""


class StochasticIndicator(Indicator):
    """Stochastic Oscillator (%K / %D)."""

    def __init__(self, k_window: int = 14, d_window: int = 3) -> None:
        self.k_window = k_window
        self.d_window = d_window

    @property
    def name(self) -> str:
        return f"Stochastic ({self.k_window},{self.d_window})"

    @property
    def weight(self) -> float:
        return 1.0

    def compute(self, df: pd.DataFrame) -> IndicatorResult:
        settings = get_settings()
        ob = settings.stochastic_overbought  # 80
        os_ = settings.stochastic_oversold   # 20

        k, d = _stoch(df["High"], df["Low"], df["Close"], self.k_window, self.d_window)

        k_clean = k.dropna()
        d_clean = d.dropna()

        cur_k = float(k_clean.iloc[-1])
        cur_d = float(d_clean.iloc[-1]) if len(d_clean) > 0 else cur_k

        # %K / %D crossover detection
        if len(k_clean) >= 2 and len(d_clean) >= 2:
            bullish_cross = (
                k_clean.iloc[-2] < d_clean.iloc[-2]
                and k_clean.iloc[-1] >= d_clean.iloc[-1]
            )
            bearish_cross = (
                k_clean.iloc[-2] > d_clean.iloc[-2]
                and k_clean.iloc[-1] <= d_clean.iloc[-1]
            )
        else:
            bullish_cross = bearish_cross = False

        if cur_k < os_ and bullish_cross:
            signal_val = Signal.STRONG_BUY
            interp = f"%K {cur_k:.1f} crosses above %D in oversold zone — strong reversal signal"
        elif cur_k < os_:
            signal_val = Signal.BUY
            interp = f"%K {cur_k:.1f} oversold (< {os_})"
        elif cur_k > ob and bearish_cross:
            signal_val = Signal.STRONG_SELL
            interp = f"%K {cur_k:.1f} crosses below %D in overbought zone — strong reversal signal"
        elif cur_k > ob:
            signal_val = Signal.SELL
            interp = f"%K {cur_k:.1f} overbought (> {ob})"
        else:
            signal_val = Signal.NEUTRAL
            interp = f"%K {cur_k:.1f} — neutral zone ({os_}-{ob})"

        return IndicatorResult(
            indicator_name=self.name,
            signal=signal_val,
            value=round(cur_k, 2),
            details={
                "k": round(cur_k, 2),
                "d": round(cur_d, 2),
                "overbought": ob,
                "oversold": os_,
                "bullish_cross": bullish_cross,
                "bearish_cross": bearish_cross,
            },
            interpretation=interp,
        )
