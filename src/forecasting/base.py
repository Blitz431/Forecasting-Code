from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

"""
Purpose: Abstract base classes and shared data structures for all 12 forecasting methods.

Connections:
  - src/forecasting/decomposition.py, exponential_smoothing.py, moving_average.py, regression.py, auto_best.py:
    all subclass ForecastMethod and return ForecastResult
  - src/forecasting/runner.py: calls method.evaluate() on each registered method
  - src/forecasting/metrics.py: called from ForecastMethod.evaluate() to compute RMSE/MAE/MAPE

In:  quarterly price pd.Series (DatetimeIndex, no NaNs)
Out: ForecastResult dataclass (forecasts, fitted_values, rmse, mae, mape, error)
"""


@dataclass
class ForecastResult:
    """Outcome produced by one forecast method for one ticker.

    Attributes:
        method_name: Human-readable name (e.g. "Holt-Winters").
        method_number: Method index 1-12.
        forecasts: Predicted quarterly values for the next ``horizons`` periods,
            indexed by projected quarter-end dates.
        fitted_values: In-sample fitted values over the training series.
        rmse: Holdout root-mean-squared error.
        mae: Holdout mean-absolute error.
        mape: Holdout mean-absolute-percentage error.
        ticker: Ticker symbol (set by the runner).
        error: Error message if the method failed; None on success.
    """

    method_name: str
    method_number: int
    forecasts: pd.Series
    fitted_values: pd.Series
    rmse: float
    mae: float
    mape: float
    ticker: str = ""
    error: str | None = None

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "method_number": self.method_number,
            "method_name": self.method_name,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "ticker": self.ticker,
            "error": self.error,
        }


@dataclass
class _FailedResult:
    """Internal sentinel returned when a method raises an exception."""

    method_name: str
    method_number: int
    error: str

    def to_forecast_result(self) -> ForecastResult:
        return ForecastResult(
            method_name=self.method_name,
            method_number=self.method_number,
            forecasts=pd.Series(dtype=float),
            fitted_values=pd.Series(dtype=float),
            rmse=float("inf"),
            mae=float("inf"),
            mape=float("inf"),
            error=self.error,
        )


class ForecastMethod(ABC):
    """Strategy-pattern base class for all 12 forecasting methods.

    Subclasses implement :meth:`fit` and :meth:`predict`. The shared
    :meth:`evaluate` method handles holdout splitting, metric calculation, and
    refitting on the full series — so subclasses do not need to worry about it.

    All series passed to these methods are assumed to be **quarterly**
    close prices (positive floats), indexed by ``pd.DatetimeIndex``.
    """

    # ------------------------------------------------------------------ #
    # Abstract interface
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable method name."""
        ...

    @property
    @abstractmethod
    def number(self) -> int:
        """Method index, 1-12."""
        ...

    @abstractmethod
    def fit(self, series: pd.Series) -> None:
        """Fit the model on *series* (in-place).

        After this call :meth:`predict` and :meth:`fitted_values` are valid.

        Args:
            series: Quarterly price series, DatetimeIndex, no NaNs.
        """
        ...

    @abstractmethod
    def predict(self, horizons: int) -> pd.Series:
        """Return forecasts for the next *horizons* quarters.

        Args:
            horizons: Number of future quarters to forecast.

        Returns:
            Series of length *horizons* with projected quarter-end DatetimeIndex.
        """
        ...

    @abstractmethod
    def fitted_values(self) -> pd.Series:
        """Return in-sample fitted values over the last-fit series."""
        ...

    # ------------------------------------------------------------------ #
    # Shared evaluate logic
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        series: pd.Series,
        holdout: int = 8,
        horizons: int = 4,
    ) -> ForecastResult:
        """Full evaluation: holdout metrics + final forecast on full series.

        Steps:
        1. Split series into train / holdout.
        2. Fit on train, predict *holdout* steps, compute RMSE/MAE/MAPE.
        3. Refit on full series, predict *horizons* steps (the real forecast).

        Args:
            series: Full quarterly price series.
            holdout: Number of periods held out for error measurement.
            horizons: Number of future periods to forecast after refitting.

        Returns:
            :class:`ForecastResult` with metrics and final forecasts.
        """
        from src.forecasting.metrics import compute_mae, compute_mape, compute_rmse

        if len(series) <= holdout:
            raise ValueError(
                f"[{self.name}] Series too short ({len(series)} pts) "
                f"for holdout={holdout}"
            )

        train = series.iloc[:-holdout]
        actual = series.iloc[-holdout:]

        # --- Holdout evaluation ---
        self.fit(train)
        holdout_preds = self.predict(holdout)

        rmse = compute_rmse(actual.values, holdout_preds.values)
        mae = compute_mae(actual.values, holdout_preds.values)
        mape = compute_mape(actual.values, holdout_preds.values)

        # --- Refit on full data for final forecast ---
        self.fit(series)
        final_forecasts = self.predict(horizons)
        fv = self.fitted_values()

        return ForecastResult(
            method_name=self.name,
            method_number=self.number,
            forecasts=final_forecasts,
            fitted_values=fv,
            rmse=rmse,
            mae=mae,
            mape=mape,
        )
