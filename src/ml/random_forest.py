"""Random Forest regressor model for Phase 4 ML Engine.

Uses scikit-learn's RandomForestRegressor with parallelism across all CPU cores.
Feature importances are exposed via get_feature_importance() for SHAP compatibility.
"""

from __future__ import annotations

import numpy as np

from src.ml.base import MLModel
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class RandomForestModel(MLModel):
    """Scikit-learn Random Forest regressor.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth per tree (None = expand until leaves are pure).
        min_samples_split: Minimum samples to split an internal node.
        min_samples_leaf: Minimum samples in a leaf node.
        max_features: Number of features to consider at each split.
                      "sqrt" is the standard for regression; float = fraction.
        max_samples: Fraction of samples to draw for each tree (bootstrap size).
        n_jobs: Number of parallel jobs (-1 = all cores).
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 3,
        max_features: str | float = 0.5,
        max_samples: float = 0.8,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self._model = None

    @property
    def name(self) -> str:
        return "Random Forest"

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.ensemble import RandomForestRegressor

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            max_samples=self.max_samples,
            n_jobs=self.n_jobs,
            random_state=42,
        )
        self._model.fit(X, y)
        logger.debug(f"Random Forest trained: {self.n_estimators} trees")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("RandomForestModel.train() must be called before predict()")
        return self._model.predict(X).astype(np.float64)

    def get_feature_importance(self, feature_names: list[str] | None = None) -> dict | None:
        if self._model is None:
            return None
        scores = self._model.feature_importances_
        if feature_names and len(feature_names) == len(scores):
            return dict(zip(feature_names, scores.tolist()))
        return {f"f{i}": float(s) for i, s in enumerate(scores)}
