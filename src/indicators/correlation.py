"""Correlation indicator — rolling price vs volume and price vs volatility.

Two rolling correlations over a 20-day window:

1. **Price-Volume Correlation**: correlation between daily % price change and
   daily volume change.
   High positive → volume confirms price moves (trend health).
   High negative → volume diverging from price (potential reversal).

2. **Price-Volatility Correlation**: correlation between price level and ATR.
   Rising price with rising ATR = buying climax risk.
   Rising price with falling ATR = calm trend (healthier).

Combined signal
---------------
Both correlations bullish   → BUY
Both correlations bearish   → SELL
Price-vol +, price-vola -   → STRONG_BUY (confirmation + calm trend)
Price-vol -, price-vola +   → STRONG_SELL (divergence + climax)
Mixed                       → NEUTRAL
"""

from __future__ import annotations

import pandas as pd

from src.indicators._calc import atr
from src.indicators.base import Indicator, IndicatorResult, Signal

_WINDOW = 20
_STRONG_THRESHOLD = 0.5
_WEAK_THRESHOLD = 0.1


class CorrelationIndicator(Indicator):
    """Rolling price-volume and price-volatility correlation."""

    def __init__(self, window: int = _WINDOW) -> None:
        self.window = window

    @property
    def name(self) -> str:
        return f"Correlation (price/vol, {self.window}d)"

    @property
    def weight(self) -> float:
        return 0.75

    def compute(self, df: pd.DataFrame) -> IndicatorResult:
        price_chg = df["Close"].pct_change()
        vol_chg = df["Volume"].pct_change()
        atr_series = atr(df["High"], df["Low"], df["Close"], 14)

        # Price-volume correlation
        pv_corr = price_chg.rolling(self.window).corr(vol_chg)
        cur_pv = float(pv_corr.dropna().iloc[-1]) if pv_corr.dropna().shape[0] > 0 else 0.0

        # Price-volatility correlation
        pa_corr = df["Close"].rolling(self.window).corr(atr_series)
        cur_pa = float(pa_corr.dropna().iloc[-1]) if pa_corr.dropna().shape[0] > 0 else 0.0

        # Sub-signals
        pv_bullish = cur_pv > _WEAK_THRESHOLD    # volume confirms price moves
        pv_bearish = cur_pv < -_WEAK_THRESHOLD   # volume diverging
        pa_bullish = cur_pa < -_WEAK_THRESHOLD   # price rising, vol falling = calm
        pa_bearish = cur_pa > _STRONG_THRESHOLD  # price rising, vol rising = climax risk

        if pv_bullish and pa_bullish:
            signal_val = Signal.STRONG_BUY
            interp = (
                f"Volume confirms price ({cur_pv:+.2f}) + calm trend (ATR corr {cur_pa:+.2f})"
            )
        elif pv_bearish and pa_bearish:
            signal_val = Signal.STRONG_SELL
            interp = (
                f"Volume diverging ({cur_pv:+.2f}) + volatility rising with price ({cur_pa:+.2f})"
            )
        elif pv_bullish:
            signal_val = Signal.BUY
            interp = f"Volume confirming price moves (corr {cur_pv:+.2f})"
        elif pv_bearish:
            signal_val = Signal.SELL
            interp = f"Volume diverging from price (corr {cur_pv:+.2f}) — watch for reversal"
        else:
            signal_val = Signal.NEUTRAL
            interp = f"No significant price-volume correlation ({cur_pv:+.2f})"

        return IndicatorResult(
            indicator_name=self.name,
            signal=signal_val,
            value=round(cur_pv, 3),
            details={
                "price_volume_corr": round(cur_pv, 3),
                "price_atr_corr": round(cur_pa, 3),
                "window": self.window,
            },
            interpretation=interp,
        )
