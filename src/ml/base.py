"""Abstract base classes and data structures for the ML Forecasting Engine (Phase 4).

All ML models implement MLModel so the runner can treat them uniformly.
MLResult is the single return type from every model evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class MLResult:
    """Outcome produced by one ML model for one ticker.

    Attributes:
        model_name: Human-readable model name (e.g. "XGBoost").
        ticker: Stock ticker symbol.
        predictions: Array of predicted target values on the test set.
        actual: Array of actual target values on the test set.
        rmse: Root-mean-squared error on the test set.
        mae: Mean-absolute error on the test set.
        mape: Mean-absolute-percentage error on the test set.
        feature_importance: Dict mapping feature name -> importance score (None if unsupported).
        train_start: ISO date string marking train period start.
        train_end: ISO date string marking train period end.
        test_start: ISO date string marking test period start.
        test_end: ISO date string marking test period end.
        error: Error message if the model failed; None on success.
    """

    model_name: str
    ticker: str
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    actual: np.ndarray = field(default_factory=lambda: np.array([]))
    rmse: float = float("inf")
    mae: float = float("inf")
    mape: float = float("inf")
    feature_importance: dict | None = None
    train_start: str | None = None
    train_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "ticker": self.ticker,
            "rmse": round(self.rmse, 6) if self.rmse != float("inf") else None,
            "mae": round(self.mae, 6) if self.mae != float("inf") else None,
            "mape": round(self.mape, 6) if self.mape != float("inf") else None,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "error": self.error,
        }


class MLModel(ABC):
    """Strategy-pattern base class for all ML forecasting models.

    Subclasses implement :meth:`train` and :meth:`predict`. The shared
    :meth:`evaluate` method computes RMSE/MAE/MAPE so subclasses don't need to.

    All methods receive 2D numpy arrays:
        X: (n_samples, n_features)
        y: (n_samples,)  — next-N-day percentage return by default
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name shown in reports."""
        ...

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model on training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input features.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).
        """
        ...

    def get_feature_importance(self, feature_names: list[str] | None = None) -> dict | None:
        """Return feature importance scores, or None if unsupported.

        Override in tree-based models that expose feature importances.
        """
        return None

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        ticker: str = "",
        feature_names: list[str] | None = None,
        train_start: str | None = None,
        train_end: str | None = None,
        test_start: str | None = None,
        test_end: str | None = None,
    ) -> MLResult:
        """Run predictions on test data and compute evaluation metrics.

        Args:
            X_test: Test feature matrix.
            y_test: Test target vector.
            ticker: Stock symbol for the result.
            feature_names: Optional feature names for importance mapping.
            train_start/end, test_start/end: Period labels for the result.

        Returns:
            :class:`MLResult` populated with metrics and predictions.
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        preds = self.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))

        nonzero = np.abs(y_test) > 1e-8
        if nonzero.any():
            mape = float(np.mean(np.abs((y_test[nonzero] - preds[nonzero]) / y_test[nonzero])))
        else:
            mape = float("inf")

        importance = self.get_feature_importance(feature_names)

        return MLResult(
            model_name=self.name,
            ticker=ticker,
            predictions=preds,
            actual=y_test,
            rmse=rmse,
            mae=mae,
            mape=mape,
            feature_importance=importance,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

    def save(self, path: Path) -> None:
        """Persist model artifact to disk using pickle (override for .pt/.pkl)."""
        import pickle

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_disk(cls, path: Path) -> "MLModel":
        """Load a previously saved model from disk."""
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
