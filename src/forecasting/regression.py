"""OLS regression forecast method (Method 11).

Models quarterly prices as a linear function of time plus quarterly dummy
variables that capture systematic seasonal patterns:

    Price_t = β0 + β1·t + β2·Q1 + β3·Q2 + β4·Q3 + ε_t

where Q4 is the implicit baseline.  Coefficients are estimated by ordinary
least squares (statsmodels OLS).  The forecast simply evaluates the fitted
equation at future time indices with the appropriate quarterly dummies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.base import ForecastMethod

_SEASONAL_PERIOD = 4


def _project_dates(last_date: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start=last_date, periods=n + 1, freq="QE")[1:]


def _build_design_matrix(n: int, start_quarter_idx: int = 0) -> np.ndarray:
    """Build the [1, t, Q1, Q2, Q3] design matrix for *n* time steps.

    Args:
        n: Number of rows.
        start_quarter_idx: The quarter-of-year index (0-3) at position 0.

    Returns:
        (n, 5) float array: intercept, time, Q1 dummy, Q2 dummy, Q3 dummy.
    """
    intercept = np.ones(n)
    t = np.arange(n, dtype=float)

    quarter_of_year = [(start_quarter_idx + i) % _SEASONAL_PERIOD for i in range(n)]
    q1 = np.array([1.0 if q == 0 else 0.0 for q in quarter_of_year])
    q2 = np.array([1.0 if q == 1 else 0.0 for q in quarter_of_year])
    q3 = np.array([1.0 if q == 2 else 0.0 for q in quarter_of_year])

    return np.column_stack([intercept, t, q1, q2, q3])


class OLSRegression(ForecastMethod):
    """Method 11 — OLS with linear time trend and quarterly dummy variables.

    The model captures:
    - A global linear trend (β1·t).
    - Quarter-of-year fixed effects (β2·Q1, β3·Q2, β4·Q3).

    Fitted via statsmodels OLS (gives standard errors, R², etc. for free).
    Forecasting is done by evaluating the equation at future t values.

    Attributes:
        _params: Fitted coefficient vector [β0, β1, β2, β3, β4].
        _start_q: Quarter-of-year index (0-3) of the first observation.
    """

    _series: pd.Series | None = None
    _params: np.ndarray | None = None
    _start_q: int = 0
    _n_train: int = 0

    @property
    def name(self) -> str:
        return "OLS + Quarterly Dummies"

    @property
    def number(self) -> int:
        return 11

    def fit(self, series: pd.Series) -> None:
        import statsmodels.api as sm

        if len(series) < 6:
            raise ValueError(f"[{self.name}] Need ≥6 quarters, got {len(series)}")

        self._series = series
        self._n_train = len(series)

        # Determine the calendar quarter of the first observation (0=Q1 … 3=Q4)
        first_date = series.index[0]
        self._start_q = (first_date.month - 1) // 3  # 0-indexed

        X = _build_design_matrix(len(series), self._start_q)
        y = series.values.astype(float)

        ols = sm.OLS(y, X).fit()
        self._params = ols.params

    def predict(self, horizons: int) -> pd.Series:
        # Time index continues from where training ended
        t_start = self._n_train
        quarter_start = (self._start_q + self._n_train) % _SEASONAL_PERIOD

        X_future = _build_design_matrix(horizons, quarter_start)
        # Adjust the time column to continue from t_start
        X_future[:, 1] += t_start

        preds = X_future @ self._params
        future_dates = _project_dates(self._series.index[-1], horizons)
        return pd.Series(preds, index=future_dates, name="forecast")

    def fitted_values(self) -> pd.Series:
        X = _build_design_matrix(self._n_train, self._start_q)
        fitted = X @ self._params
        return pd.Series(fitted, index=self._series.index, name="fitted")
