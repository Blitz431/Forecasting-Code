"""Feature engineering for the ML Forecasting Engine (Phase 4).

Builds a feature matrix (X) and target vector (y) from:
  - Daily OHLCV price data
  - FRED macro series
  - Technical indicators (ta library)
  - Dividend yield
  - Seasonality features

Target variable: next-N-day percentage return (default N=1).

Public API
----------
build_features(ticker, daily_df, macro_df, dividends_df, target_days, drop_na)
    -> tuple[pd.DataFrame, pd.Series, list[str]]

load_and_build(ticker, settings, target_days)
    -> tuple[pd.DataFrame, pd.Series, list[str]]
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# Rolling window sizes for price/volume features
_WINDOWS = [5, 10, 20, 60]

# Lag sizes for lagged return features
_RETURN_LAGS = [1, 2, 3, 5, 10, 20, 60]


# ------------------------------------------------------------------ #
# Price + volume features
# ------------------------------------------------------------------ #


def _price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price-derived features from OHLCV data."""
    feats = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Log price level (helps with non-stationarity)
    feats["log_close"] = np.log(close.clip(lower=1e-6))

    # Lagged returns
    for lag in _RETURN_LAGS:
        feats[f"return_{lag}d"] = close.pct_change(lag)

    # Rolling volatility (std of daily returns)
    daily_ret = close.pct_change()
    for w in _WINDOWS:
        feats[f"vol_{w}d"] = daily_ret.rolling(w).std()

    # Rolling mean / close ratio (price relative to its moving average)
    for w in _WINDOWS:
        feats[f"close_sma_ratio_{w}d"] = close / close.rolling(w).mean()

    # High-low channel position: (close - low_N) / (high_N - low_N)
    for w in _WINDOWS:
        roll_low = low.rolling(w).min()
        roll_high = high.rolling(w).max()
        denom = (roll_high - roll_low).replace(0, np.nan)
        feats[f"channel_pos_{w}d"] = (close - roll_low) / denom

    # Daily gap: open vs previous close
    if "Open" in df.columns:
        feats["gap_pct"] = (df["Open"] - close.shift(1)) / close.shift(1)

    # Intraday range as % of close
    feats["intraday_range_pct"] = (high - low) / close

    # Volume features
    feats["volume_change_1d"] = volume.pct_change()
    for w in _WINDOWS:
        vol_ma = volume.rolling(w).mean()
        feats[f"volume_ratio_{w}d"] = volume / vol_ma.replace(0, np.nan)

    return feats


# ------------------------------------------------------------------ #
# Technical indicator features (using `ta` library)
# ------------------------------------------------------------------ #


def _ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical analysis features using the `ta` library."""
    try:
        import ta
    except ImportError:
        logger.warning("ta library not installed — skipping TA features")
        return pd.DataFrame(index=df.index)

    feats = pd.DataFrame(index=df.index)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # RSI
        try:
            feats["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            feats["rsi_7"] = ta.momentum.RSIIndicator(close=close, window=7).rsi()
        except Exception:
            pass

        # MACD
        try:
            macd = ta.trend.MACD(close=close)
            feats["macd_line"] = macd.macd()
            feats["macd_signal"] = macd.macd_signal()
            feats["macd_hist"] = macd.macd_diff()
            feats["macd_cross"] = (feats["macd_line"] > feats["macd_signal"]).astype(float)
        except Exception:
            pass

        # Bollinger Bands
        try:
            bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            feats["bb_pband"] = bb.bollinger_pband()   # %B: 0=lower, 1=upper
            feats["bb_wband"] = bb.bollinger_wband()   # bandwidth
        except Exception:
            pass

        # Stochastic Oscillator
        try:
            stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close)
            feats["stoch_k"] = stoch.stoch()
            feats["stoch_d"] = stoch.stoch_signal()
        except Exception:
            pass

        # ATR (Average True Range) — normalized by close
        try:
            atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close).average_true_range()
            feats["atr_pct"] = atr / close
        except Exception:
            pass

        # ADX (trend strength)
        try:
            adx_ind = ta.trend.ADXIndicator(high=high, low=low, close=close)
            feats["adx"] = adx_ind.adx()
            feats["adx_pos"] = adx_ind.adx_pos()
            feats["adx_neg"] = adx_ind.adx_neg()
        except Exception:
            pass

        # CCI (Commodity Channel Index)
        try:
            feats["cci_20"] = ta.trend.CCIIndicator(high=high, low=low, close=close, window=20).cci()
        except Exception:
            pass

        # ROC (Rate of Change / Momentum)
        try:
            feats["roc_10"] = ta.momentum.ROCIndicator(close=close, window=10).roc()
            feats["roc_20"] = ta.momentum.ROCIndicator(close=close, window=20).roc()
        except Exception:
            pass

        # OBV (On-Balance Volume) normalized as % change
        try:
            obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
            feats["obv_change_10d"] = obv.pct_change(10)
        except Exception:
            pass

        # Williams %R
        try:
            feats["williams_r"] = ta.momentum.WilliamsRIndicator(
                high=high, low=low, close=close, lbp=14
            ).williams_r()
        except Exception:
            pass

        # EMA ratios (close / EMA)
        try:
            for span in [9, 21, 50, 200]:
                ema = ta.trend.EMAIndicator(close=close, window=span).ema_indicator()
                feats[f"close_ema_ratio_{span}"] = close / ema.replace(0, np.nan)
        except Exception:
            pass

    return feats


# ------------------------------------------------------------------ #
# Macro features
# ------------------------------------------------------------------ #


def _macro_features(macro_df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Align FRED macro series to the daily price index via forward-fill."""
    if macro_df is None or macro_df.empty:
        return pd.DataFrame(index=index)

    # Forward-fill within macro_df first so quarterly/monthly series propagate
    # across the daily-frequency rows already present in the combined index
    # (method="ffill" on reindex only fills *new* dates, not pre-existing NaNs).
    macro_filled = macro_df.sort_index().ffill()
    macro_aligned = macro_filled.reindex(index, method="ffill")

    # Rename columns to avoid clashes
    macro_aligned.columns = [f"macro_{c}" for c in macro_aligned.columns]

    # Add yield curve slope (10Y - 2Y) if both present
    if "macro_DGS10" in macro_aligned.columns and "macro_DGS2" in macro_aligned.columns:
        macro_aligned["macro_yield_slope"] = (
            macro_aligned["macro_DGS10"] - macro_aligned["macro_DGS2"]
        )

    # Percentage changes for flow variables (make stationary)
    for col in ["macro_GDP", "macro_CPIAUCSL", "macro_INDPRO", "macro_UMCSENT"]:
        if col in macro_aligned.columns:
            macro_aligned[f"{col}_chg"] = macro_aligned[col].pct_change(periods=21)

    return macro_aligned


# ------------------------------------------------------------------ #
# Dividend features
# ------------------------------------------------------------------ #


def _dividend_features(dividends_df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill dividend yield into daily price index."""
    if dividends_df is None or dividends_df.empty:
        return pd.DataFrame(index=index)

    feats = pd.DataFrame(index=index)

    # Look for dividend yield column
    for col in ["Dividend_Yield", "dividend_yield", "yield"]:
        if col in dividends_df.columns:
            aligned = dividends_df[col].reindex(index, method="ffill")
            feats["dividend_yield"] = aligned
            break

    return feats


# ------------------------------------------------------------------ #
# Seasonality features
# ------------------------------------------------------------------ #


def _seasonality_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Encode calendar-based seasonality features."""
    feats = pd.DataFrame(index=index)

    feats["day_of_week"] = index.dayofweek               # 0=Mon … 4=Fri
    feats["month"] = index.month                          # 1-12
    feats["quarter"] = index.quarter                      # 1-4
    feats["week_of_year"] = index.isocalendar().week.astype(int)
    feats["is_month_end"] = index.is_month_end.astype(float)
    feats["is_quarter_end"] = index.is_quarter_end.astype(float)

    # Cyclical encoding for day-of-week and month (avoids ordinal distance issues)
    feats["dow_sin"] = np.sin(2 * np.pi * feats["day_of_week"] / 5)
    feats["dow_cos"] = np.cos(2 * np.pi * feats["day_of_week"] / 5)
    feats["month_sin"] = np.sin(2 * np.pi * (feats["month"] - 1) / 12)
    feats["month_cos"] = np.cos(2 * np.pi * (feats["month"] - 1) / 12)

    return feats


# ------------------------------------------------------------------ #
# Main builder
# ------------------------------------------------------------------ #


def build_features(
    ticker: str,
    daily_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    dividends_df: pd.DataFrame | None = None,
    target_days: int = 1,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build the ML feature matrix and target vector for *ticker*.

    Args:
        ticker: Stock symbol (used only for logging).
        daily_df: Daily OHLCV DataFrame with DatetimeIndex, sorted ascending.
                  Required columns: Open, High, Low, Close, Volume.
        macro_df: Optional FRED macro DataFrame with DatetimeIndex.
        dividends_df: Optional dividend history DataFrame with DatetimeIndex.
        target_days: Number of trading days ahead for the return target.
                     1 = next-day close return, 5 = next-week, etc.
        drop_na: If True, drop rows with any NaN features or NaN target.

    Returns:
        X: Feature DataFrame (DatetimeIndex).
        y: Target Series — forward return over *target_days* (DatetimeIndex).
        feature_names: List of column names in X (same as X.columns).
    """
    df = daily_df.copy().sort_index()
    if df.empty:
        raise ValueError(f"[{ticker}] daily_df is empty")

    # Ensure required columns are present
    for col in ("Close", "High", "Low", "Volume"):
        if col not in df.columns:
            raise ValueError(f"[{ticker}] daily_df missing required column '{col}'")
    if "Open" not in df.columns:
        df["Open"] = df["Close"]

    # ---- Target: next-N-day forward return ----
    y = df["Close"].pct_change(target_days).shift(-target_days)
    y.name = f"fwd_return_{target_days}d"

    # ---- Feature groups ----
    groups = [
        _price_features(df),
        _ta_features(df),
        _macro_features(macro_df, df.index),
        _dividend_features(dividends_df, df.index),
        _seasonality_features(df.index),
    ]

    X = pd.concat([g for g in groups if not g.empty], axis=1)

    # ---- Align y with X ----
    X, y = X.align(y, join="left", axis=0)

    # Ensure all columns are float64 so pd.NA (pandas 3.x integer columns) is
    # converted to np.nan before the notna() check runs.
    X = X.astype(np.float64)

    # Replace inf values produced by pct_change / division on zeros.
    X = X.replace([np.inf, -np.inf], np.nan)

    if drop_na:
        valid = X.notna().all(axis=1) & y.notna()
        X = X.loc[valid]
        y = y.loc[valid]

    feature_names = list(X.columns)
    logger.info(
        f"[{ticker}] Feature matrix: {len(X)} samples × {len(feature_names)} features "
        f"(target=fwd_return_{target_days}d)"
    )

    return X, y, feature_names


# ------------------------------------------------------------------ #
# Convenience loader
# ------------------------------------------------------------------ #


def load_and_build(
    ticker: str,
    settings=None,
    target_days: int = 1,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load stored Parquet data for *ticker* and call :func:`build_features`.

    Args:
        ticker: Stock symbol.
        settings: Optional settings override.
        target_days: Forward return horizon in trading days.

    Returns:
        (X, y, feature_names) from :func:`build_features`.
    """
    from config.settings import get_settings
    from src.scraper.storage import get_ticker_filepath, load_dataframe

    if settings is None:
        settings = get_settings()

    # Load daily OHLCV
    daily_path = get_ticker_filepath(ticker, settings.raw_daily_dir)
    daily_df = load_dataframe(daily_path)
    if daily_df.empty:
        raise FileNotFoundError(
            f"No daily data for {ticker}. Run `python cli/scrape.py` first."
        )

    # Load macro data (all FRED series combined into one DataFrame)
    macro_df = _load_macro(settings)

    # Load dividend data
    dividends_df = _load_dividends(ticker, settings)

    return build_features(
        ticker=ticker,
        daily_df=daily_df,
        macro_df=macro_df,
        dividends_df=dividends_df,
        target_days=target_days,
    )


def _load_macro(settings) -> pd.DataFrame | None:
    """Load all FRED series and combine into a single wide DataFrame."""
    from src.scraper.storage import load_dataframe

    frames = []
    for series_id in settings.fred_series:
        path = settings.raw_macro_dir / f"{series_id}.parquet"
        df = load_dataframe(path)
        if df.empty:
            continue
        # Macro files have the value column named after the series
        if "value" in df.columns:
            s = df["value"].rename(series_id)
        elif series_id in df.columns:
            s = df[series_id]
        else:
            s = df.iloc[:, 0].rename(series_id)
        frames.append(s)

    if not frames:
        return None

    combined = pd.concat(frames, axis=1)
    combined.index = pd.to_datetime(combined.index)
    return combined.sort_index()


def _load_dividends(ticker: str, settings) -> pd.DataFrame | None:
    """Load dividend history for *ticker*."""
    from src.scraper.storage import load_dataframe

    path = settings.raw_dividends_dir / f"{ticker}.parquet"
    df = load_dataframe(path)
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    return df.sort_index()
