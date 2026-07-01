from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.forecasting.base import ForecastMethod, ForecastResult
from src.forecasting.decomposition import (
    AdditiveDecomposition,
    EnsembleDecomposition,
    FlatTrendDecomposition,
    LinearTrendDecomposition,
    MultiplicativeDecomposition,
)
from src.forecasting.exponential_smoothing import (
    HoltLinearTrend,
    HoltWinters,
    SimpleExpSmoothing,
)
from src.forecasting.metrics import compute_rmse
from src.forecasting.moving_average import SimpleMovingAverage, WeightedMovingAverage
from src.forecasting.regression import OLSRegression
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: AutoBest (Method 12) — picks the best of Methods 1–11 via bi-directional holdout evaluation.

Connections:
  - src/forecasting/decomposition.py, exponential_smoothing.py, moving_average.py, regression.py:
    instantiates all 11 methods as candidates
  - src/forecasting/base.py: ForecastMethod, ForecastResult types
  - src/forecasting/metrics.py: compute_rmse() to score each candidate
  - src/forecasting/runner.py: instantiated and called as Method 12 in ALL_METHODS

In:  quarterly price pd.Series
Out: ForecastResult from the winning method, with the winner's name and combined RMSE
"""


def _candidate_methods() -> list[ForecastMethod]:
    """Return fresh instances of Methods 1-11."""
    return [
        AdditiveDecomposition(),
        MultiplicativeDecomposition(),
        FlatTrendDecomposition(),
        LinearTrendDecomposition(),
        EnsembleDecomposition(),
        SimpleMovingAverage(),
        WeightedMovingAverage(),
        SimpleExpSmoothing(),
        HoltLinearTrend(),
        HoltWinters(),
        OLSRegression(),
    ]


def _holdout_rmse(
    method: ForecastMethod,
    series: pd.Series,
    holdout: int,
) -> float:
    """Fit *method* on series[:-holdout], predict *holdout* steps, return RMSE.

    Returns ``float('inf')`` if the method raises any exception.
    """
    if len(series) <= holdout:
        return float("inf")

    train = series.iloc[:-holdout]
    actual = series.iloc[-holdout:].values.astype(float)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            method.fit(train)
            preds = method.predict(holdout).values.astype(float)
        if len(preds) != holdout:
            return float("inf")
        rmse = compute_rmse(actual, preds)
        return float("inf") if not np.isfinite(rmse) else rmse
    except Exception as exc:
        logger.debug(f"[AutoBest] {method.name} holdout failed: {exc}")
        return float("inf")


class AutoBest(ForecastMethod):
    """Method 12 — AutoBest: bi-directional holdout model selection.

    After selection the winning method is exposed via
    ``best_method_name`` and ``best_rmse`` attributes.

    Args:
        holdout: Number of periods held out for forward and backward
            evaluation (default 8, as specified in the design plan).
    """

    def __init__(self, holdout: int = 8) -> None:
        self.holdout = holdout
        self.best_method_name: str = ""
        self.best_combined_rmse: float = float("inf")
        self._best_method: ForecastMethod | None = None
        self._series: pd.Series | None = None
        self._scores: dict[str, dict] = {}

    @property
    def name(self) -> str:
        return f"AutoBest (best: {self.best_method_name or '?'})"

    @property
    def number(self) -> int:
        return 12

    # ------------------------------------------------------------------ #
    # Core interface
    # ------------------------------------------------------------------ #

    def fit(self, series: pd.Series) -> None:
        """Select best method via bi-directional holdout, then fit on *series*."""
        self._series = series
        candidates = _candidate_methods()
        reversed_series = series.iloc[::-1].reset_index(drop=True)
        reversed_series.index = series.index[::-1]

        best_rmse = float("inf")
        best_method: ForecastMethod | None = None
        scores: dict[str, dict] = {}

        for method in candidates:
            fwd_rmse = _holdout_rmse(method, series, self.holdout)
            bwd_rmse = _holdout_rmse(method, reversed_series, self.holdout)

            combined = (
                np.mean([fwd_rmse, bwd_rmse])
                if np.isfinite(fwd_rmse) and np.isfinite(bwd_rmse)
                else float("inf")
            )

            scores[method.name] = {
                "forward_rmse": fwd_rmse,
                "backward_rmse": bwd_rmse,
                "combined_rmse": combined,
            }

            logger.debug(
                f"[AutoBest] {method.name:40s}  "
                f"fwd={fwd_rmse:8.2f}  bwd={bwd_rmse:8.2f}  combined={combined:8.2f}"
            )

            if combined < best_rmse:
                best_rmse = combined
                best_method = method

        self._scores = scores

        if best_method is None:
            raise RuntimeError("[AutoBest] All candidate methods failed — cannot select best.")

        # Refit the winner on the full series
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_method.fit(series)

        self._best_method = best_method
        self.best_method_name = best_method.name
        self.best_combined_rmse = best_rmse

        logger.info(
            f"[AutoBest] Winner: {self.best_method_name}  "
            f"combined RMSE={self.best_combined_rmse:.2f}"
        )

    def predict(self, horizons: int) -> pd.Series:
        if self._best_method is None:
            raise RuntimeError("[AutoBest] Not fitted — call fit() first.")
        return self._best_method.predict(horizons)

    def fitted_values(self) -> pd.Series:
        if self._best_method is None:
            raise RuntimeError("[AutoBest] Not fitted — call fit() first.")
        return self._best_method.fitted_values()

    # ------------------------------------------------------------------ #
    # Extras
    # ------------------------------------------------------------------ #

    def scores_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame of all methods' forward/backward/combined RMSEs."""
        if not self._scores:
            return pd.DataFrame()
        rows = [
            {"method": name, **vals}
            for name, vals in self._scores.items()
        ]
        df = pd.DataFrame(rows).sort_values("combined_rmse").reset_index(drop=True)
        return df
