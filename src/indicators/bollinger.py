"""Bollinger Bands indicator (20-period, 2 std dev).

Signal logic (based on %B — normalised position within bands)
--------------------------------------------------------------
%B < 0      (below lower band)  → STRONG_BUY  (price extended below bands)
%B < 0.2                        → BUY
0.2 ≤ %B ≤ 0.8                 → NEUTRAL
%B > 0.8                        → SELL
%B > 1.0    (above upper band)  → STRONG_SELL (price extended above bands)

Bandwidth squeeze (low volatility preceding a breakout) is noted in details.
"""

from __future__ import annotations

import pandas as pd

from src.indicators._calc import bollinger_bands, bb_position
from src.indicators.base import Indicator, IndicatorResult, Signal


class BollingerBandsIndicator(Indicator):
    """Bollinger Bands with %B signal."""

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self.window = window
        self.num_std = num_std

    @property
    def name(self) -> str:
        return f"Bollinger Bands ({self.window}, {self.num_std:.0f}std)"

    @property
    def weight(self) -> float:
        return 1.0

    def compute(self, df: pd.DataFrame) -> IndicatorResult:
        upper, mid, lower = bollinger_bands(df["Close"], self.window, self.num_std)
        bb_pos = bb_position(df["Close"], self.window, self.num_std)

        cur_close = float(df["Close"].iloc[-1])
        cur_upper = float(upper.dropna().iloc[-1])
        cur_mid = float(mid.dropna().iloc[-1])
        cur_lower = float(lower.dropna().iloc[-1])
        cur_pos = float(bb_pos.dropna().iloc[-1])

        # Bandwidth (relative width of bands) — detect squeeze
        bandwidth = (cur_upper - cur_lower) / cur_mid if cur_mid != 0 else 0
        # Squeeze: bandwidth in bottom 20th percentile of recent history
        recent_bw = ((upper - lower) / mid.replace(0, float("nan"))).dropna().tail(50)
        squeeze = bandwidth < float(recent_bw.quantile(0.20)) if len(recent_bw) >= 20 else False

        if cur_pos < 0:
            signal_val = Signal.STRONG_BUY
            interp = f"Price below lower band (%B={cur_pos:.2f}) — oversold extension"
        elif cur_pos < 0.2:
            signal_val = Signal.BUY
            interp = f"Price near lower band (%B={cur_pos:.2f})"
        elif cur_pos > 1.0:
            signal_val = Signal.STRONG_SELL
            interp = f"Price above upper band (%B={cur_pos:.2f}) — overbought extension"
        elif cur_pos > 0.8:
            signal_val = Signal.SELL
            interp = f"Price near upper band (%B={cur_pos:.2f})"
        else:
            signal_val = Signal.NEUTRAL
            interp = f"Price in middle of bands (%B={cur_pos:.2f})"

        if squeeze:
            interp += " | Bandwidth squeeze — watch for breakout"

        return IndicatorResult(
            indicator_name=self.name,
            signal=signal_val,
            value=round(cur_pos, 3),
            details={
                "percent_b": round(cur_pos, 3),
                "upper": round(cur_upper, 2),
                "middle": round(cur_mid, 2),
                "lower": round(cur_lower, 2),
                "close": round(cur_close, 2),
                "bandwidth": round(bandwidth, 4),
                "squeeze": squeeze,
            },
            interpretation=interp,
        )
