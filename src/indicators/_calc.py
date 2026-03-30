"""Low-level indicator math — pure pandas/numpy, no external TA library.

These functions mirror the standard definitions used by Bloomberg, TradingView,
and most quant textbooks.  They are intentionally dependency-free so that
indicator modules can be imported even if optional packages are unavailable.

All inputs are pandas Series with a DatetimeIndex.
All outputs are pandas Series aligned to the same index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# RSI (Wilder / exponential smoothing variant)
# ------------------------------------------------------------------ #

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's exponential smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    # Wilder smoothing = EWM with alpha = 1/window
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_out = 100 - 100 / (1 + rs)
    # avg_loss == 0 and avg_gain > 0  →  RSI = 100 (no down days at all)
    # avg_loss == 0 and avg_gain == 0 →  RSI = 50  (completely flat)
    rsi_out = rsi_out.where(avg_loss != 0, np.where(avg_gain > 0, 100.0, 50.0))
    return rsi_out.rename("RSI")


# ------------------------------------------------------------------ #
# MACD
# ------------------------------------------------------------------ #

def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram.

    Returns:
        (macd_line, signal_line, histogram)
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow).rename("MACD")
    signal_line = macd_line.ewm(span=signal, adjust=False).mean().rename("MACD_Signal")
    histogram = (macd_line - signal_line).rename("MACD_Hist")
    return macd_line, signal_line, histogram


# ------------------------------------------------------------------ #
# Bollinger Bands
# ------------------------------------------------------------------ #

def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Upper, middle (SMA), lower Bollinger Bands.

    Returns:
        (upper, middle, lower)
    """
    mid = close.rolling(window).mean().rename("BB_Mid")
    std = close.rolling(window).std()
    upper = (mid + num_std * std).rename("BB_Upper")
    lower = (mid - num_std * std).rename("BB_Lower")
    return upper, mid, lower


def bb_position(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Normalised position within Bollinger Bands: 0 = lower, 1 = upper."""
    upper, _, lower = bollinger_bands(close, window, num_std)
    width = (upper - lower).replace(0, np.nan)
    return ((close - lower) / width).rename("BB_Pos")


# ------------------------------------------------------------------ #
# Stochastic Oscillator
# ------------------------------------------------------------------ #

def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K and %D.

    Returns:
        (%K, %D)
    """
    low_min = low.rolling(k_window).min()
    high_max = high.rolling(k_window).max()
    k = (100 * (close - low_min) / (high_max - low_min).replace(0, np.nan)).rename("Stoch_K")
    d = k.rolling(d_window).mean().rename("Stoch_D")
    return k, d


# ------------------------------------------------------------------ #
# Moving Averages
# ------------------------------------------------------------------ #

def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean().rename(f"SMA{window}")


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean().rename(f"EMA{span}")


# ------------------------------------------------------------------ #
# ATR (Average True Range)
# ------------------------------------------------------------------ #

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Average True Range."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean().rename("ATR")


# ------------------------------------------------------------------ #
# OBV (On Balance Volume)
# ------------------------------------------------------------------ #

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum().rename("OBV")


# ------------------------------------------------------------------ #
# Rate of Change / Momentum
# ------------------------------------------------------------------ #

def roc(close: pd.Series, window: int) -> pd.Series:
    """Rate of Change: (price_t / price_{t-n}) - 1."""
    return ((close / close.shift(window)) - 1).rename(f"ROC{window}")


# ------------------------------------------------------------------ #
# Volume ratio
# ------------------------------------------------------------------ #

def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Current volume divided by rolling mean volume."""
    return (volume / volume.rolling(window).mean().replace(0, np.nan)).rename("VolumeRatio")
