"""Moving averages indicator — SMA/EMA 50/200 + Golden/Death Cross.

Signal logic
------------
Golden Cross: SMA50 > SMA200 AND recently crossed above   → STRONG_BUY
SMA50 > SMA200 (uptrend confirmed)                        → BUY
Price above SMA50 but SMA50 < SMA200                      → NEUTRAL
SMA50 < SMA200 (downtrend)                                → SELL
Death Cross: SMA50 < SMA200 AND recently crossed below    → STRONG_SELL
"""

from __future__ import annotations

import pandas as pd

from src.indicators._calc import ema, sma
from src.indicators.base import Indicator, IndicatorResult, Signal

_CROSS_LOOKBACK = 5  # bars to look back for a recent cross


def _recent_cross(fast: pd.Series, slow: pd.Series, lookback: int) -> tuple[bool, bool]:
    """Detect a golden/death cross in the last *lookback* bars."""
    fast_c = fast.dropna()
    slow_c = slow.dropna()
    n = min(len(fast_c), len(slow_c), lookback + 1)
    if n < 2:
        return False, False

    diff = fast_c.iloc[-n:].values - slow_c.iloc[-n:].values
    golden = any(diff[i - 1] < 0 and diff[i] >= 0 for i in range(1, len(diff)))
    death = any(diff[i - 1] > 0 and diff[i] <= 0 for i in range(1, len(diff)))
    return golden, death


class MovingAveragesIndicator(Indicator):
    """SMA 50/200 + EMA 50/200 with Golden/Death Cross detection."""

    @property
    def name(self) -> str:
        return "Moving Averages (50/200)"

    @property
    def weight(self) -> float:
        return 2.0

    def compute(self, df: pd.DataFrame) -> IndicatorResult:
        close = df["Close"]
        sma50 = sma(close, 50)
        sma200 = sma(close, 200)
        ema50 = ema(close, 50)

        cur_close = float(close.iloc[-1])
        cur_sma50 = float(sma50.dropna().iloc[-1]) if sma50.dropna().shape[0] else None
        cur_sma200 = float(sma200.dropna().iloc[-1]) if sma200.dropna().shape[0] else None
        cur_ema50 = float(ema50.dropna().iloc[-1]) if ema50.dropna().shape[0] else None

        if cur_sma50 is None or cur_sma200 is None:
            return IndicatorResult(
                indicator_name=self.name,
                signal=Signal.NEUTRAL,
                interpretation="Not enough data for SMA50/200 (need 200+ daily bars)",
            )

        golden_cross, death_cross = _recent_cross(sma50, sma200, _CROSS_LOOKBACK)

        if golden_cross:
            signal_val = Signal.STRONG_BUY
            interp = f"Golden Cross — SMA50 crossed above SMA200 (SMA50={cur_sma50:.2f}, SMA200={cur_sma200:.2f})"
        elif cur_sma50 > cur_sma200:
            signal_val = Signal.BUY
            interp = f"Uptrend — SMA50 {cur_sma50:.2f} > SMA200 {cur_sma200:.2f}"
        elif death_cross:
            signal_val = Signal.STRONG_SELL
            interp = f"Death Cross — SMA50 crossed below SMA200 (SMA50={cur_sma50:.2f}, SMA200={cur_sma200:.2f})"
        elif cur_sma50 < cur_sma200:
            signal_val = Signal.SELL
            interp = f"Downtrend — SMA50 {cur_sma50:.2f} < SMA200 {cur_sma200:.2f}"
        else:
            signal_val = Signal.NEUTRAL
            interp = f"SMA50 ≈ SMA200 — no clear trend ({cur_sma50:.2f} vs {cur_sma200:.2f})"

        # Annotate price position relative to SMA50
        price_vs_sma50 = (cur_close - cur_sma50) / cur_sma50
        if abs(price_vs_sma50) > 0.01:
            direction = "above" if price_vs_sma50 > 0 else "below"
            interp += f" | Price {abs(price_vs_sma50):.1%} {direction} SMA50"

        return IndicatorResult(
            indicator_name=self.name,
            signal=signal_val,
            value=round(cur_sma50, 2),
            details={
                "sma50": round(cur_sma50, 2),
                "sma200": round(cur_sma200, 2),
                "ema50": round(cur_ema50, 2) if cur_ema50 else None,
                "close": round(cur_close, 2),
                "golden_cross": golden_cross,
                "death_cross": death_cross,
                "price_vs_sma50_pct": round(price_vs_sma50 * 100, 2),
            },
            interpretation=interp,
        )
