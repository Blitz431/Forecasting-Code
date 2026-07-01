from __future__ import annotations

import pandas as pd

from src.indicators._calc import atr, bollinger_bands
from src.indicators.base import Indicator, IndicatorResult, Signal

"""
Purpose: Fundamentals indicator — P/S ratio (yfinance) and ATR-based Risk/Reward as lightweight tiebreakers.

Connections:
  - src/indicators/_calc.py: atr(), bollinger_bands() math functions
  - src/indicators/base.py: Indicator ABC, IndicatorResult, Signal types
  - src/indicators/signal_aggregator.py: called with weight 0.75 (lowest weight)

In:  daily OHLCV pd.DataFrame + yfinance API call for P/S ratio
Out: IndicatorResult with Signal enum and P/S value / R/R ratio
"""


def _get_ps_ratio(ticker: str) -> float | None:
    """Fetch P/S ratio from yfinance. Returns None on any failure."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        ps = info.get("priceToSalesTrailing12Months")
        if ps and ps > 0:
            return float(ps)
        return None
    except Exception:
        return None


def _ps_signal(ps: float) -> tuple[Signal, str]:
    if ps < 2:
        return Signal.STRONG_BUY, f"P/S {ps:.1f} — potentially undervalued"
    if ps < 4:
        return Signal.BUY, f"P/S {ps:.1f} — reasonably valued"
    if ps <= 8:
        return Signal.NEUTRAL, f"P/S {ps:.1f} — fairly valued"
    if ps <= 15:
        return Signal.SELL, f"P/S {ps:.1f} — elevated valuation"
    return Signal.STRONG_SELL, f"P/S {ps:.1f} — high valuation risk"


def _rr_signal(df: pd.DataFrame) -> tuple[Signal, str, float]:
    """ATR-based risk/reward signal. Returns (signal, interp, rr_ratio)."""
    cur_close = float(df["Close"].iloc[-1])
    atr_val = float(atr(df["High"], df["Low"], df["Close"], 14).dropna().iloc[-1])
    upper, _, _ = bollinger_bands(df["Close"], 20, 2.0)
    cur_upper = float(upper.dropna().iloc[-1])

    upside = max(cur_upper - cur_close, 0)
    downside = atr_val * 2  # 2× ATR as stop-loss proxy

    rr = upside / downside if downside > 0 else 0.0

    if rr >= 3.0:
        sig, interp = Signal.STRONG_BUY, f"R/R {rr:.1f} — very favourable"
    elif rr >= 2.0:
        sig, interp = Signal.BUY, f"R/R {rr:.1f} — favourable"
    elif rr >= 1.0:
        sig, interp = Signal.NEUTRAL, f"R/R {rr:.1f} — balanced"
    elif rr >= 0.5:
        sig, interp = Signal.SELL, f"R/R {rr:.1f} — unfavourable"
    else:
        sig, interp = Signal.STRONG_SELL, f"R/R {rr:.1f} — poor risk/reward"

    return sig, interp, round(rr, 2)


class FundamentalsIndicator(Indicator):
    """P/S ratio + ATR risk/reward fundamentals indicator.

    Args:
        ticker: Required to fetch P/S from yfinance.
    """

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    @property
    def name(self) -> str:
        return "Fundamentals (P/S + R/R)"

    @property
    def weight(self) -> float:
        return 0.75  # lower weight — rough signal only

    def compute(self, df: pd.DataFrame) -> IndicatorResult:
        ps = _get_ps_ratio(self.ticker)
        rr_sig, rr_interp, rr_val = _rr_signal(df)

        if ps is not None:
            ps_sig, ps_interp = _ps_signal(ps)
            # Average the two signals (round toward zero for ties)
            combined_val = (int(ps_sig) + int(rr_sig)) / 2
            if combined_val >= 1.5:
                final_signal = Signal.STRONG_BUY
            elif combined_val >= 0.5:
                final_signal = Signal.BUY
            elif combined_val <= -1.5:
                final_signal = Signal.STRONG_SELL
            elif combined_val <= -0.5:
                final_signal = Signal.SELL
            else:
                final_signal = Signal.NEUTRAL

            interp = f"{ps_interp} | {rr_interp}"
            details = {"ps_ratio": round(ps, 2), "risk_reward": rr_val}
            value = round(ps, 2)
        else:
            # Fall back to R/R only
            final_signal = rr_sig
            interp = f"P/S unavailable | {rr_interp}"
            details = {"ps_ratio": None, "risk_reward": rr_val}
            value = rr_val

        return IndicatorResult(
            indicator_name=self.name,
            signal=final_signal,
            value=value,
            details=details,
            interpretation=interp,
        )
