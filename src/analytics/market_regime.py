"""Market regime detection: Bull / Bear / Sideways / High-Volatility.

Uses a multi-factor scoring approach:
  1. VIX level (fear gauge)
  2. 10Y-2Y yield curve spread (recession signal)
  3. SPY 50-day and 200-day momentum
  4. Market breadth (% of S&P 500 stocks above their 200-day MA)

Usage
-----
    from src.analytics.market_regime import MarketRegimeAnalyzer, RegimeLabel

    analyzer = MarketRegimeAnalyzer(settings)
    snapshot = analyzer.current_snapshot()
    print(snapshot.regime)    # "Bull" / "Bear" / "Sideways" / "High-Volatility"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd


class RegimeLabel(str, Enum):
    BULL        = "Bull"
    BEAR        = "Bear"
    SIDEWAYS    = "Sideways"
    HIGH_VOL    = "High-Volatility"
    RECOVERY    = "Recovery"
    UNKNOWN     = "Unknown"

    def color(self) -> str:
        return {
            "Bull":             "#26a69a",
            "Bear":             "#ef5350",
            "Sideways":         "#ffa726",
            "High-Volatility":  "#ab47bc",
            "Recovery":         "#66bb6a",
            "Unknown":          "#9e9e9e",
        }.get(self.value, "#9e9e9e")

    def strategy_hint(self) -> str:
        return {
            "Bull":            "Aggressive — favour momentum, growth, higher position sizes",
            "Bear":            "Defensive — reduce exposure, favour cash, utilities, bonds",
            "Sideways":        "Neutral — mean-reversion strategies, tighter stops",
            "High-Volatility": "Caution — reduce size by 50%, widen stops, avoid earnings plays",
            "Recovery":        "Opportunistic — add exposure gradually, favour quality",
            "Unknown":         "Insufficient data — default to neutral sizing",
        }.get(self.value, "Neutral sizing")


@dataclass
class RegimeSnapshot:
    regime: RegimeLabel = RegimeLabel.UNKNOWN
    vix: float | None = None
    yield_spread: float | None = None      # 10Y - 2Y
    spy_50d_return: float | None = None
    spy_200d_return: float | None = None
    breadth_pct: float | None = None       # % stocks above 200-day MA (0-100)
    score: float = 0.0                     # aggregate score: >0 bullish, <0 bearish
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["regime"] = self.regime.value
        return d


# ---------------------------------------------------------------------------#
# Scoring weights
# ---------------------------------------------------------------------------#

def _classify(
    vix: float | None,
    spread: float | None,
    spy_50d: float | None,
    spy_200d: float | None,
    breadth: float | None,
) -> tuple[RegimeLabel, float, dict]:
    """Multi-factor regime classification.  Returns (label, score, details)."""

    score = 0.0
    details: dict = {}

    # --- VIX ---
    if vix is not None:
        if vix > 35:
            score -= 2.0
            details["vix"] = f"Crisis ({vix:.1f})"
        elif vix > 25:
            score -= 1.0
            details["vix"] = f"Elevated ({vix:.1f})"
        elif vix < 15:
            score += 1.0
            details["vix"] = f"Low/Complacent ({vix:.1f})"
        else:
            details["vix"] = f"Normal ({vix:.1f})"

    # --- Yield curve ---
    if spread is not None:
        if spread < -0.5:
            score -= 1.5
            details["yield_curve"] = f"Deeply inverted ({spread:.2f}%)"
        elif spread < 0:
            score -= 0.5
            details["yield_curve"] = f"Inverted ({spread:.2f}%)"
        elif spread > 1.5:
            score += 0.5
            details["yield_curve"] = f"Steep ({spread:.2f}%)"
        else:
            details["yield_curve"] = f"Flat/Normal ({spread:.2f}%)"

    # --- SPY momentum ---
    if spy_50d is not None:
        if spy_50d > 0.08:
            score += 1.5
            details["spy_50d"] = f"Strong uptrend ({spy_50d*100:.1f}%)"
        elif spy_50d > 0.02:
            score += 0.5
            details["spy_50d"] = f"Mild uptrend ({spy_50d*100:.1f}%)"
        elif spy_50d < -0.10:
            score -= 2.0
            details["spy_50d"] = f"Strong downtrend ({spy_50d*100:.1f}%)"
        elif spy_50d < -0.03:
            score -= 1.0
            details["spy_50d"] = f"Mild downtrend ({spy_50d*100:.1f}%)"
        else:
            details["spy_50d"] = f"Flat ({spy_50d*100:.1f}%)"

    if spy_200d is not None:
        if spy_200d > 0.15:
            score += 1.0
            details["spy_200d"] = f"Long-term bull ({spy_200d*100:.1f}%)"
        elif spy_200d < -0.15:
            score -= 1.0
            details["spy_200d"] = f"Long-term bear ({spy_200d*100:.1f}%)"
        else:
            details["spy_200d"] = f"Neutral ({spy_200d*100:.1f}%)"

    # --- Breadth ---
    if breadth is not None:
        if breadth > 70:
            score += 1.0
            details["breadth"] = f"Broad participation ({breadth:.0f}%)"
        elif breadth < 30:
            score -= 1.0
            details["breadth"] = f"Narrow/Weak ({breadth:.0f}%)"
        else:
            details["breadth"] = f"Mixed ({breadth:.0f}%)"

    # --- Label ---
    if vix is not None and vix > 35:
        label = RegimeLabel.HIGH_VOL
    elif score >= 2.5:
        label = RegimeLabel.BULL
    elif score <= -2.5:
        label = RegimeLabel.BEAR
    elif -2.5 < score < -0.5:
        # Recovering if recent 50d positive but still net negative
        if spy_50d is not None and spy_50d > 0 and score > -1.5:
            label = RegimeLabel.RECOVERY
        else:
            label = RegimeLabel.BEAR
    elif -0.5 <= score <= 0.5:
        label = RegimeLabel.SIDEWAYS
    else:
        label = RegimeLabel.BULL

    return label, score, details


class MarketRegimeAnalyzer:
    """Compute and cache the current market regime from FRED + price data."""

    def __init__(self, settings=None):
        self._settings = settings

    def _load_fred(self, series_id: str) -> pd.DataFrame:
        if self._settings is None:
            return pd.DataFrame()
        fp = self._settings.raw_macro_dir / f"{series_id}.parquet"
        if not fp.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(fp)
        except Exception:
            return pd.DataFrame()

    def _load_spy(self) -> pd.DataFrame:
        if self._settings is None:
            return pd.DataFrame()
        fp = self._settings.raw_daily_dir / "SPY.parquet"
        if not fp.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(fp)
        except Exception:
            return pd.DataFrame()

    def compute_breadth(self, tickers: list[str] | None = None) -> float | None:
        """Compute % of tickers with price above their 200-day MA.

        Parameters
        ----------
        tickers:
            List of tickers to check.  Defaults to all files in raw_daily_dir.

        Returns
        -------
        Float 0-100, or None if insufficient data.
        """
        if self._settings is None:
            return None

        data_dir = self._settings.raw_daily_dir
        if not data_dir.exists():
            return None

        if tickers is None:
            files = list(data_dir.glob("*.parquet"))[:200]   # cap at 200 for speed
            tickers = [f.stem for f in files]

        above = 0
        total = 0
        for t in tickers:
            fp = data_dir / f"{t}.parquet"
            if not fp.exists():
                continue
            try:
                df = pd.read_parquet(fp)
                if "Close" not in df.columns or len(df) < 200:
                    continue
                close = df["Close"].dropna()
                ma200 = float(close.iloc[-200:].mean())
                current = float(close.iloc[-1])
                total += 1
                if current > ma200:
                    above += 1
            except Exception:
                continue

        return (above / total * 100) if total > 0 else None

    def current_snapshot(self, compute_breadth: bool = False) -> RegimeSnapshot:
        """Build the current regime snapshot from available data."""
        vix_df    = self._load_fred("VIXCLS")
        t10y2y_df = self._load_fred("T10Y2Y")
        spy_df    = self._load_spy()

        # Latest values
        vix = float(vix_df.iloc[-1].iloc[0]) if not vix_df.empty else None
        spread = float(t10y2y_df.iloc[-1].iloc[0]) if not t10y2y_df.empty else None

        spy_50d = spy_200d = None
        if not spy_df.empty and "Close" in spy_df.columns:
            closes = spy_df["Close"].dropna()
            if len(closes) >= 200:
                cur = float(closes.iloc[-1])
                spy_50d  = cur / float(closes.iloc[-50])  - 1
                spy_200d = cur / float(closes.iloc[-200]) - 1
            elif len(closes) >= 50:
                cur = float(closes.iloc[-1])
                spy_50d = cur / float(closes.iloc[-50]) - 1

        breadth = self.compute_breadth() if compute_breadth else None

        label, score, details = _classify(vix, spread, spy_50d, spy_200d, breadth)

        return RegimeSnapshot(
            regime=label,
            vix=vix,
            yield_spread=spread,
            spy_50d_return=spy_50d,
            spy_200d_return=spy_200d,
            breadth_pct=breadth,
            score=score,
            details=details,
        )

    def historical_regimes(self, window: int = 63) -> pd.DataFrame:
        """Compute regime label for each day in SPY history using a rolling window.

        Returns DataFrame: date, regime, score, vix_approx.
        Useful for overlaying regimes on the backtest equity curve.
        """
        spy_df = self._load_spy()
        vix_df = self._load_fred("VIXCLS")
        t10y2y_df = self._load_fred("T10Y2Y")

        if spy_df.empty or "Close" not in spy_df.columns:
            return pd.DataFrame()

        closes = spy_df["Close"].dropna()
        vix_col = vix_df.columns[0] if not vix_df.empty else None
        spread_col = t10y2y_df.columns[0] if not t10y2y_df.empty else None

        rows = []
        for i in range(window, len(closes)):
            date = closes.index[i]
            window_close = closes.iloc[i - window: i + 1]
            spy_ret = float(window_close.iloc[-1] / window_close.iloc[0] - 1)

            vix_val = None
            if vix_df is not None and vix_col:
                try:
                    past_vix = vix_df[vix_col][:date]
                    if not past_vix.empty:
                        vix_val = float(past_vix.iloc[-1])
                except Exception:
                    pass

            spread_val = None
            if t10y2y_df is not None and spread_col:
                try:
                    past_spread = t10y2y_df[spread_col][:date]
                    if not past_spread.empty:
                        spread_val = float(past_spread.iloc[-1])
                except Exception:
                    pass

            label, score, _ = _classify(vix_val, spread_val, spy_ret, None, None)
            rows.append({"date": date, "regime": label.value, "score": score,
                         "vix": vix_val, "spy_ret": spy_ret})

        return pd.DataFrame(rows).set_index("date")
