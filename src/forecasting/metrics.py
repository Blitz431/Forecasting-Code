"""Forecast accuracy metrics: RMSE, MAE, MAPE.

Used by every ForecastMethod via the shared evaluate() call, and also by
AutoBest to compare methods during the bi-directional holdout evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root-mean-squared error."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean-absolute error."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean-absolute-percentage error.

    Excludes zero-valued actuals to avoid division-by-zero.
    Returns ``float('inf')`` if no valid pairs remain.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = actual != 0
    if not nonzero.any():
        return float("inf")
    return float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])))


def holdout_metrics(
    series: pd.Series,
    fitted_or_preds: pd.Series,
    holdout: int,
) -> dict[str, float]:
    """Compute metrics for the last *holdout* periods of *series*.

    Args:
        series: Full actual series.
        fitted_or_preds: Predicted values aligned to *series* index.
        holdout: Number of tail periods to evaluate.

    Returns:
        Dict with keys ``rmse``, ``mae``, ``mape``.
    """
    actual = series.iloc[-holdout:].values
    predicted = fitted_or_preds.iloc[-holdout:].values
    return {
        "rmse": compute_rmse(actual, predicted),
        "mae": compute_mae(actual, predicted),
        "mape": compute_mape(actual, predicted),
    }


def metrics_summary(results: list) -> pd.DataFrame:
    """Build a sorted comparison DataFrame from a list of ForecastResult.

    Args:
        results: List of :class:`~src.forecasting.base.ForecastResult`.

    Returns:
        DataFrame with columns method_number, method_name, rmse, mae, mape,
        sorted ascending by rmse.
    """
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("rmse").reset_index(drop=True)
