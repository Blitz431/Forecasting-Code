from __future__ import annotations

import pandas as pd

from config.settings import get_settings
from src.indicators._calc import rsi as _rsi
from src.indicators.base import Indicator, IndicatorResult, Signal

"""
Purpose: RSI indicator — Wilder's 14-period Relative Strength Index, signals overbought/oversold.

Connections:
  - src/indicators/_calc.py: rsi() math function
  - src/indicators/base.py: Indicator ABC, IndicatorResult, Signal types
  - config/settings.py: rsi_overbought (70), rsi_oversold (30) thresholds
  - src/indicators/signal_aggregator.py: called with weight 1.5

In:  daily OHLCV pd.DataFrame (needs Close column, min ~30 rows)
Out: IndicatorResult with Signal enum and RSI value
"""


class RSIIndicator(Indicator):
    """Relative Strength Index (14-period, Wilder smoothing).

    Signal logic
    ------------
    RSI < oversold         → STRONG_BUY  (deeply oversold)
    oversold ≤ RSI < 40    → BUY
    40 ≤ RSI ≤ 60          → NEUTRAL
    60 < RSI ≤ overbought  → SELL
    RSI > overbought       → STRONG_SELL (deeply overbought)
    """

    def __init__(self, window: int = 14) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"RSI ({self.window})"

    @property
    def weight(self) -> float:
        return 1.5

    def compute(self, df: pd.DataFrame) -> IndicatorResult:
        settings = get_settings()
        ob = settings.rsi_overbought   # 70
        os_ = settings.rsi_oversold    # 30

        series = _rsi(df["Close"], self.window)
        current = float(series.dropna().iloc[-1])

        if current < os_:
            signal = Signal.STRONG_BUY
            interp = f"RSI {current:.1f} — deeply oversold (< {os_})"
        elif current < 40:
            signal = Signal.BUY
            interp = f"RSI {current:.1f} — approaching oversold territory"
        elif current <= 60:
            signal = Signal.NEUTRAL
            interp = f"RSI {current:.1f} — neutral zone (40-60)"
        elif current <= ob:
            signal = Signal.SELL
            interp = f"RSI {current:.1f} — approaching overbought territory"
        else:
            signal = Signal.STRONG_SELL
            interp = f"RSI {current:.1f} — deeply overbought (> {ob})"

        return IndicatorResult(
            indicator_name=self.name,
            signal=signal,
            value=round(current, 2),
            details={"rsi": round(current, 2), "overbought": ob, "oversold": os_},
            interpretation=interp,
        )
