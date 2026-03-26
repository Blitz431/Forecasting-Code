"""Moving-average forecast methods (Methods 6-7).

Methods
-------
6. Simple Moving Average (SMA) — unweighted average of the last *window*
   quarters, projected flat as a constant forecast + seasonal adjustment.
7. Weighted Moving Average (WMA) — linearly weighted average so the most
   recent quarter contributes most, then same flat projection.

Both methods handle seasonality by computing a simple seasonal index from
the trailing history and reapplying it to the flat-forecast baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.base import ForecastMethod

_SEASONAL_PERIOD = 4  # quarterly


def _project_dates(last_date: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start=last_date, periods=n + 1, freq="QE")[1:]


def _seasonal_indices(series: pd.Series, period: int = _SEASONAL_PERIOD) -> np.ndarray:
    """Compute additive seasonal indices from the full series.

    Returns an array of length ``period`` where index 0 corresponds to
    the same quarter-of-year as the first observation in ``series``.
    """
    values = series.values.astype(float)
    n = len(values)

    # Build a matrix of shape (n_complete_cycles, period) and average
    n_full = (n // period) * period
    if n_full < period:
        return np.zeros(period)

    mat = values[-n_full:].reshape(-1, period)
    row_means = mat.mean(axis=1, keepdims=True)
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(row_means != 0, mat - row_means, 0.0)

    seasonal_idx = ratios.mean(axis=0)
    return seasonal_idx


def _future_seasonal(series: pd.Series, n: int) -> np.ndarray:
    """Get the seasonal adjustments for the next n quarters."""
    idx = _seasonal_indices(series)
    # Phase: the next quarter after series end
    last_quarter_pos = (len(series) - 1) % _SEASONAL_PERIOD
    future = [(last_quarter_pos + k + 1) % _SEASONAL_PERIOD for k in range(n)]
    return np.array([idx[q] for q in future])


class SimpleMovingAverage(ForecastMethod):
    """Method 6 — Simple Moving Average.

    Computes the unweighted mean of the last *window* observed quarters.
    Projects that mean flat into the future, then adds a seasonal adjustment
    derived from the historical average seasonal pattern.

    Args:
        window: Number of trailing quarters to average (default 4 = 1 year).
    """

    def __init__(self, window: int = 4) -> None:
        self.window = window
        self._series: pd.Series | None = None
        self._level: float = 0.0

    @property
    def name(self) -> str:
        return "Simple Moving Average"

    @property
    def number(self) -> int:
        return 6

    def fit(self, series: pd.Series) -> None:
        if len(series) < self.window:
            raise ValueError(
                f"[{self.name}] Need ≥{self.window} quarters, got {len(series)}"
            )
        self._series = series
        self._level = float(series.iloc[-self.window:].mean())

    def predict(self, horizons: int) -> pd.Series:
        seasonal_adj = _future_seasonal(self._series, horizons)
        future_dates = _project_dates(self._series.index[-1], horizons)
        forecasts = self._level + seasonal_adj
        return pd.Series(forecasts, index=future_dates, name="forecast")

    def fitted_values(self) -> pd.Series:
        rolling = self._series.rolling(window=self.window).mean()
        # Shift by 1 so the rolling mean predicts the next value
        fitted = rolling.shift(1).dropna()
        return fitted.rename("fitted")


class WeightedMovingAverage(ForecastMethod):
    """Method 7 — Weighted Moving Average.

    Applies linearly increasing weights to the last *window* quarters so the
    most recent observation has weight *window* and the oldest has weight 1.
    Same seasonal adjustment as Method 6.

    Args:
        window: Number of trailing quarters to include (default 8 = 2 years).
    """

    def __init__(self, window: int = 8) -> None:
        self.window = window
        self._series: pd.Series | None = None
        self._level: float = 0.0

    @property
    def name(self) -> str:
        return "Weighted Moving Average"

    @property
    def number(self) -> int:
        return 7

    def fit(self, series: pd.Series) -> None:
        w = min(self.window, len(series))
        self._series = series
        values = series.iloc[-w:].values.astype(float)
        weights = np.arange(1, w + 1, dtype=float)
        self._level = float(np.dot(weights, values) / weights.sum())

    def predict(self, horizons: int) -> pd.Series:
        seasonal_adj = _future_seasonal(self._series, horizons)
        future_dates = _project_dates(self._series.index[-1], horizons)
        forecasts = self._level + seasonal_adj
        return pd.Series(forecasts, index=future_dates, name="forecast")

    def fitted_values(self) -> pd.Series:
        """Rolling WMA as fitted values (1-step-ahead)."""
        w = min(self.window, len(self._series))
        weights = np.arange(1, w + 1, dtype=float)
        weights /= weights.sum()

        values = self._series.values.astype(float)
        fitted = np.full(len(values), np.nan)
        for i in range(w, len(values)):
            fitted[i] = np.dot(weights, values[i - w : i])

        return pd.Series(fitted, index=self._series.index, name="fitted").dropna()
