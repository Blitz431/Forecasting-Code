from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from src.forecasting.base import ForecastMethod

"""
Purpose: Decomposition-based forecast methods 1–5 (Additive, Multiplicative, Flat-Trend, Linear-Trend, Ensemble).

Connections:
  - src/forecasting/base.py: subclasses ForecastMethod, returns ForecastResult
  - src/forecasting/runner.py: instantiated and called via ALL_METHODS list

In:  quarterly price pd.Series (min 12 quarters — 3 full seasonal cycles)
Out: ForecastResult with forecasts + holdout RMSE/MAE/MAPE per method
"""

_MIN_PERIODS = 12  # 3 full annual cycles at quarterly frequency
_SEASONAL_PERIOD = 4  # quarterly data


def _project_dates(last_date: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """Generate n quarterly period-end dates after last_date."""
    return pd.date_range(start=last_date, periods=n + 1, freq="QE")[1:]


def _avg_trend_increment(trend: pd.Series) -> float:
    """Mean period-over-period increment of the trend component."""
    diffs = trend.dropna().diff().dropna()
    return float(diffs.mean()) if len(diffs) > 0 else 0.0


def _linear_trend_slope(trend: pd.Series) -> float:
    """OLS slope of the trend component vs. integer time index."""
    t = trend.dropna()
    if len(t) < 2:
        return 0.0
    x = np.arange(len(t))
    slope = float(np.polyfit(x, t.values, 1)[0])
    return slope


def _repeat_seasonal(seasonal: pd.Series, n: int) -> np.ndarray:
    """Tile the last seasonal cycle forward for n future steps."""
    cycle = seasonal.values[-_SEASONAL_PERIOD:]
    reps = (n // _SEASONAL_PERIOD) + 2
    tiled = np.tile(cycle, reps)
    return tiled[:n]


class AdditiveDecomposition(ForecastMethod):
    """Method 1 — Additive seasonal decomposition, average-trend extrapolation.

    Y = Trend + Seasonal + Residual.
    Forecast = (last_trend + k * avg_increment) + seasonal[k % 4]
    """

    _series: pd.Series | None = None
    _trend: pd.Series | None = None
    _seasonal: pd.Series | None = None

    @property
    def name(self) -> str:
        return "Additive Decomposition"

    @property
    def number(self) -> int:
        return 1

    def fit(self, series: pd.Series) -> None:
        if len(series) < _MIN_PERIODS:
            raise ValueError(
                f"[{self.name}] Need ≥{_MIN_PERIODS} quarters, got {len(series)}"
            )
        self._series = series
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = seasonal_decompose(
                series, model="additive", period=_SEASONAL_PERIOD, extrapolate_trend="freq"
            )
        self._trend = result.trend
        self._seasonal = result.seasonal

    def predict(self, horizons: int) -> pd.Series:
        last_trend = float(self._trend.dropna().iloc[-1])
        increment = _avg_trend_increment(self._trend)
        future_seasonal = _repeat_seasonal(self._seasonal, horizons)
        future_dates = _project_dates(self._series.index[-1], horizons)

        forecasts = [
            last_trend + (k + 1) * increment + future_seasonal[k]
            for k in range(horizons)
        ]
        return pd.Series(forecasts, index=future_dates, name="forecast")

    def fitted_values(self) -> pd.Series:
        return (self._trend + self._seasonal).rename("fitted").dropna()


class MultiplicativeDecomposition(ForecastMethod):
    """Method 2 — Multiplicative seasonal decomposition, average-trend extrapolation.

    Y = Trend × Seasonal × Residual.
    Forecast = (last_trend + k * avg_increment) × seasonal[k % 4]
    """

    _series: pd.Series | None = None
    _trend: pd.Series | None = None
    _seasonal: pd.Series | None = None

    @property
    def name(self) -> str:
        return "Multiplicative Decomposition"

    @property
    def number(self) -> int:
        return 2

    def fit(self, series: pd.Series) -> None:
        if len(series) < _MIN_PERIODS:
            raise ValueError(
                f"[{self.name}] Need ≥{_MIN_PERIODS} quarters, got {len(series)}"
            )
        if (series <= 0).any():
            raise ValueError(f"[{self.name}] Multiplicative model requires positive values")
        self._series = series
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = seasonal_decompose(
                series, model="multiplicative", period=_SEASONAL_PERIOD, extrapolate_trend="freq"
            )
        self._trend = result.trend
        self._seasonal = result.seasonal

    def predict(self, horizons: int) -> pd.Series:
        last_trend = float(self._trend.dropna().iloc[-1])
        increment = _avg_trend_increment(self._trend)
        future_seasonal = _repeat_seasonal(self._seasonal, horizons)
        future_dates = _project_dates(self._series.index[-1], horizons)

        forecasts = [
            (last_trend + (k + 1) * increment) * future_seasonal[k]
            for k in range(horizons)
        ]
        return pd.Series(forecasts, index=future_dates, name="forecast")

    def fitted_values(self) -> pd.Series:
        return (self._trend * self._seasonal).rename("fitted").dropna()


class FlatTrendDecomposition(ForecastMethod):
    """Method 3 — Additive decomposition, flat-trend (conservative).

    Assumes the trend stays at its most recent level — no further growth.
    Good lower-bound estimate for range forecasting.

    Forecast = last_trend + seasonal[k % 4]
    """

    _series: pd.Series | None = None
    _trend: pd.Series | None = None
    _seasonal: pd.Series | None = None

    @property
    def name(self) -> str:
        return "Flat-Trend Decomposition"

    @property
    def number(self) -> int:
        return 3

    def fit(self, series: pd.Series) -> None:
        if len(series) < _MIN_PERIODS:
            raise ValueError(
                f"[{self.name}] Need ≥{_MIN_PERIODS} quarters, got {len(series)}"
            )
        self._series = series
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = seasonal_decompose(
                series, model="additive", period=_SEASONAL_PERIOD, extrapolate_trend="freq"
            )
        self._trend = result.trend
        self._seasonal = result.seasonal

    def predict(self, horizons: int) -> pd.Series:
        last_trend = float(self._trend.dropna().iloc[-1])
        future_seasonal = _repeat_seasonal(self._seasonal, horizons)
        future_dates = _project_dates(self._series.index[-1], horizons)

        forecasts = [last_trend + future_seasonal[k] for k in range(horizons)]
        return pd.Series(forecasts, index=future_dates, name="forecast")

    def fitted_values(self) -> pd.Series:
        return (self._trend + self._seasonal).rename("fitted").dropna()


class LinearTrendDecomposition(ForecastMethod):
    """Method 4 — Additive decomposition, linear-regression trend (aggressive).

    Fits a least-squares line through the trend component and extrapolates it
    forward. Generally produces steeper forecasts than the average-increment
    method when there is strong recent momentum.

    Forecast = (β0 + β1*(T+k)) + seasonal[k % 4]
    """

    _series: pd.Series | None = None
    _trend: pd.Series | None = None
    _seasonal: pd.Series | None = None
    _intercept: float = 0.0
    _slope: float = 0.0
    _last_t: int = 0

    @property
    def name(self) -> str:
        return "Linear-Trend Decomposition"

    @property
    def number(self) -> int:
        return 4

    def fit(self, series: pd.Series) -> None:
        if len(series) < _MIN_PERIODS:
            raise ValueError(
                f"[{self.name}] Need ≥{_MIN_PERIODS} quarters, got {len(series)}"
            )
        self._series = series
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = seasonal_decompose(
                series, model="additive", period=_SEASONAL_PERIOD, extrapolate_trend="freq"
            )
        self._trend = result.trend
        self._seasonal = result.seasonal

        t = self._trend.dropna()
        x = np.arange(len(t))
        self._slope, self._intercept = np.polyfit(x, t.values, 1)
        self._last_t = len(t) - 1

    def predict(self, horizons: int) -> pd.Series:
        future_seasonal = _repeat_seasonal(self._seasonal, horizons)
        future_dates = _project_dates(self._series.index[-1], horizons)

        forecasts = [
            self._intercept + self._slope * (self._last_t + k + 1) + future_seasonal[k]
            for k in range(horizons)
        ]
        return pd.Series(forecasts, index=future_dates, name="forecast")

    def fitted_values(self) -> pd.Series:
        t = self._trend.dropna()
        x = np.arange(len(t))
        trend_fitted = self._intercept + self._slope * x
        fitted = pd.Series(trend_fitted, index=t.index) + self._seasonal.reindex(t.index)
        return fitted.rename("fitted")


class EnsembleDecomposition(ForecastMethod):
    """Method 5 — Ensemble average of Methods 1-4.

    Averages the forecasts from Additive, Multiplicative, Flat-Trend, and
    Linear-Trend decompositions. Reduces the variance of any single method's
    idiosyncratic errors.
    """

    _series: pd.Series | None = None
    _sub_methods: list[ForecastMethod]

    def __init__(self) -> None:
        self._sub_methods = [
            AdditiveDecomposition(),
            MultiplicativeDecomposition(),
            FlatTrendDecomposition(),
            LinearTrendDecomposition(),
        ]

    @property
    def name(self) -> str:
        return "Ensemble Decomposition"

    @property
    def number(self) -> int:
        return 5

    def fit(self, series: pd.Series) -> None:
        self._series = series
        for m in self._sub_methods:
            try:
                m.fit(series)
            except Exception:
                pass  # tolerate individual failures; predict() handles them

    def predict(self, horizons: int) -> pd.Series:
        all_preds = []
        for m in self._sub_methods:
            try:
                preds = m.predict(horizons)
                all_preds.append(preds.values)
            except Exception:
                pass

        if not all_preds:
            raise RuntimeError(f"[{self.name}] All sub-methods failed")

        avg = np.mean(all_preds, axis=0)
        future_dates = _project_dates(self._series.index[-1], horizons)
        return pd.Series(avg, index=future_dates, name="forecast")

    def fitted_values(self) -> pd.Series:
        all_fv = []
        for m in self._sub_methods:
            try:
                fv = m.fitted_values()
                all_fv.append(fv)
            except Exception:
                pass

        if not all_fv:
            return pd.Series(dtype=float)

        combined = pd.concat(all_fv, axis=1).mean(axis=1)
        return combined.rename("fitted")
