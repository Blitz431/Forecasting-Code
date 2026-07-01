from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators._calc import obv as _obv, volume_ratio as _vol_ratio
from src.indicators.base import Indicator, IndicatorResult, Signal

"""
Purpose: Volume analysis indicator — volume ratio vs 20-day MA combined with OBV trend direction.

Connections:
  - src/indicators/_calc.py: obv(), volume_ratio() math functions
  - src/indicators/base.py: Indicator ABC, IndicatorResult, Signal types
  - src/indicators/signal_aggregator.py: called with weight 1.0

In:  daily OHLCV pd.DataFrame (needs Close/Volume, min ~35 rows)
Out: IndicatorResult with Signal enum and volume ratio / OBV slope values
"""

_HIGH_VOL_THRESHOLD = 1.5   # volume 50% above 20-day MA
_LOW_VOL_THRESHOLD = 0.7    # volume 30% below 20-day MA
_OBV_SLOPE_WINDOW = 10      # bars for short OBV slope
_OBV_BASELINE_WINDOW = 30   # bars for baseline OBV slope


def _slope(series: pd.Series, window: int) -> float:
    """Linear regression slope of the last *window* values."""
    y = series.dropna().iloc[-window:]
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y.values, 1)[0])


class VolumeIndicator(Indicator):
    """Volume analysis: volume ratio + OBV trend."""

    @property
    def name(self) -> str:
        return "Volume Analysis"

    @property
    def weight(self) -> float:
        return 1.0

    def compute(self, df: pd.DataFrame) -> IndicatorResult:
        vol_ratio = _vol_ratio(df["Volume"], window=20)
        obv_series = _obv(df["Close"], df["Volume"])

        cur_ratio = float(vol_ratio.dropna().iloc[-1])
        cur_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else cur_close
        price_up = cur_close > prev_close

        obv_slope_short = _slope(obv_series, _OBV_SLOPE_WINDOW)
        obv_slope_base = _slope(obv_series, _OBV_BASELINE_WINDOW)
        obv_rising = obv_slope_short > 0 and obv_slope_short > obv_slope_base * 0.5
        obv_falling = obv_slope_short < 0 and obv_slope_short < obv_slope_base * 0.5

        high_vol = cur_ratio >= _HIGH_VOL_THRESHOLD
        low_vol = cur_ratio <= _LOW_VOL_THRESHOLD

        if high_vol and price_up and obv_rising:
            signal_val = Signal.STRONG_BUY
            interp = (
                f"High volume ({cur_ratio:.1f}x) + price up + OBV rising "
                "— strong accumulation"
            )
        elif high_vol and price_up:
            signal_val = Signal.BUY
            interp = f"High volume ({cur_ratio:.1f}x) confirming price increase"
        elif high_vol and not price_up and obv_falling:
            signal_val = Signal.STRONG_SELL
            interp = (
                f"High volume ({cur_ratio:.1f}x) + price down + OBV falling "
                "— strong distribution"
            )
        elif high_vol and not price_up:
            signal_val = Signal.SELL
            interp = f"High volume ({cur_ratio:.1f}x) on a down day — bearish pressure"
        elif low_vol:
            signal_val = Signal.NEUTRAL
            interp = f"Below-average volume ({cur_ratio:.1f}x) — move lacks conviction"
        else:
            signal_val = Signal.NEUTRAL
            interp = f"Average volume ({cur_ratio:.1f}x) — no strong confirmation"

        return IndicatorResult(
            indicator_name=self.name,
            signal=signal_val,
            value=round(cur_ratio, 2),
            details={
                "volume_ratio": round(cur_ratio, 2),
                "price_direction": "up" if price_up else "down",
                "obv_slope_short": round(obv_slope_short, 0),
                "obv_rising": obv_rising,
                "obv_falling": obv_falling,
            },
            interpretation=interp,
        )
