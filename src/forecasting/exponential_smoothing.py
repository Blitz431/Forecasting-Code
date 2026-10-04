from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.forecasting.base import ForecastMethod

"""
Purpose: Exponential smoothing forecast methods 8–10 (SES, Holt Linear Trend, Holt-Winters).

Connections:
  - src/forecasting/base.py: subclasses ForecastMethod, returns ForecastResult
  - src/forecasting/runner.py: instantiated and called via ALL_METHODS list

In:  quarterly price pd.Series
Out: ForecastResult with forward forecasts and holdout error metrics
"""

_SEASONAL_PERIOD = 4


def _project_dates(last_date: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start=last_date, periods=n + 1, freq="QE")[1:]


class SimpleExpSmoothing(ForecastMethod):
    """Method 8 — Simple Exponential Smoothing (SES).

    Produces a **flat** forecast equal to the smoothed level at the end of
    the series. Alpha is optimised by maximum likelihood.

    Best suited for series with no clear trend or seasonality.
    """

    _model_fit = None
    _series: pd.Series | None = None

    @property
    def name(self) -> str:
        return "Simple Exponential Smoothing"

    @property
    def number(self) -> int:
        return 8

    def fit(self, series: pd.Series) -> None:
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing as _SES

        if len(series) < 3:
            raise ValueError(f"[{self.name}] Need ≥3 observations, got {len(series)}")
        self._series = series
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model_fit = _SES(series.values.astype(float), initialization_method="estimated").fit(optimized=True)

    def predict(self, horizons: int) -> pd.Series:
        preds = self._model_fit.forecast(horizons)
        future_dates = _project_dates(self._series.index[-1], horizons)
        return pd.Series(preds, index=future_dates, name="forecast")

    def fitted_values(self) -> pd.Series:
        return pd.Series(
            self._model_fit.fittedvalues,
            index=self._series.index,
            name="fitted",
        )


class HoltLinearTrend(ForecastMethod):
    """Method 9 — Holt's Linear Trend (double exponential smoothing).

    Captures both a level and a linear trend. Alpha and beta are optimised
    by maximum likelihood. Produces a linearly trending forecast.

    Best suited for series with a clear trend but no pronounced seasonality.
    """

    _model_fit = None
    _series: pd.Series | None = None

    @property
    def name(self) -> str:
        return "Holt Linear Trend"

    @property
    def number(self) -> int:
        return 9

    def fit(self, series: pd.Series) -> None:
        from statsmodels.tsa.holtwinters import Holt

        if len(series) < 4:
            raise ValueError(f"[{self.name}] Need ≥4 observations, got {len(series)}")
        self._series = series
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model_fit = Holt(series.values.astype(float), initialization_method="estimated").fit(optimized=True)

    def predict(self, horizons: int) -> pd.Series:
        preds = self._model_fit.forecast(horizons)
        future_dates = _project_dates(self._series.index[-1], horizons)
        return pd.Series(preds, index=future_dates, name="forecast")

    def fitted_values(self) -> pd.Series:
        return pd.Series(
            self._model_fit.fittedvalues,
            index=self._series.index,
            name="fitted",
        )


class HoltWinters(ForecastMethod):
    """Method 10 — Holt-Winters Triple Exponential Smoothing.

    Captures level, trend, AND quarterly seasonality (period=4).
    Attempts additive seasonality first; falls back to multiplicative if
    the series contains non-positive values or additive fitting fails.

    Alpha (level), beta (trend), and gamma (seasonal) are all optimised
    by maximum likelihood.

    Best suited for quarterly series with both trend and seasonal patterns,
    which describes the majority of S&P 500 price histories.
    """

    _model_fit = None
    _series: pd.Series | None = None

    @property
    def name(self) -> str:
        return "Holt-Winters"

    @property
    def number(self) -> int:
        return 10

    def fit(self, series: pd.Series) -> None:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        min_len = 2 * _SEASONAL_PERIOD + 1  # need at least 2 full cycles
        if len(series) < min_len:
            raise ValueError(
                f"[{self.name}] Need ≥{min_len} quarters, got {len(series)}"
            )
        self._series = series
        values = series.values.astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Try additive first (works even when values are 0; safer)
            try:
                model = ExponentialSmoothing(
                    values,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=_SEASONAL_PERIOD,
                )
                self._model_fit = model.fit(optimized=True, use_brute=True)
            except Exception:
                # Fall back to multiplicative (requires positive values)
                if (values <= 0).any():
                    raise ValueError(
                        f"[{self.name}] Cannot fit: series contains non-positive values"
                    )
                model = ExponentialSmoothing(
                    values,
                    trend="add",
                    seasonal="mul",
                    seasonal_periods=_SEASONAL_PERIOD,
                )
                self._model_fit = model.fit(optimized=True, use_brute=True)

    def predict(self, horizons: int) -> pd.Series:
        preds = self._model_fit.forecast(horizons)
        future_dates = _project_dates(self._series.index[-1], horizons)
        return pd.Series(preds, index=future_dates, name="forecast")

    def fitted_values(self) -> pd.Series:
        return pd.Series(
            self._model_fit.fittedvalues,
            index=self._series.index,
            name="fitted",
        )
