from __future__ import annotations

import numpy as np

from src.ml.base import MLModel
from src.utils.logging import setup_logger

"""
Purpose: Ridge, Lasso, and ElasticNet linear regression models with StandardScaler — interpretable baselines for Phase 4.

Connections:
  - src/ml/base.py: subclasses MLModel, returns MLResult
  - src/ml/runner.py: instantiated and trained via run_ticker()

In:  (X_train, y_train) numpy arrays (scaled internally by StandardScaler)
Out: MLResult with predictions, RMSE/MAE/MAPE, feature_importance (coefficients); .pkl saved to disk
"""

logger = setup_logger(__name__)


# ------------------------------------------------------------------ #
# Ridge
# ------------------------------------------------------------------ #


class RidgeModel(MLModel):
    """Ridge (L2) linear regression with standard scaling.

    Args:
        alpha: Regularisation strength. Larger = more shrinkage.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._pipeline = None

    @property
    def name(self) -> str:
        return "Ridge"

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=self.alpha, random_state=42)),
        ])
        self._pipeline.fit(X, y)
        logger.debug(f"Ridge trained: alpha={self.alpha}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("RidgeModel.train() must be called before predict()")
        return self._pipeline.predict(X).astype(np.float64)

    def get_feature_importance(self, feature_names: list[str] | None = None) -> dict | None:
        if self._pipeline is None:
            return None
        coefs = np.abs(self._pipeline.named_steps["model"].coef_)
        if feature_names and len(feature_names) == len(coefs):
            return dict(zip(feature_names, coefs.tolist()))
        return {f"f{i}": float(c) for i, c in enumerate(coefs)}


# ------------------------------------------------------------------ #
# Lasso
# ------------------------------------------------------------------ #


class LassoModel(MLModel):
    """Lasso (L1) linear regression with standard scaling.

    L1 penalty drives sparse solutions — many coefficients are exactly zero,
    effectively selecting a subset of features.

    Args:
        alpha: Regularisation strength. Larger = sparser solution.
        max_iter: Maximum solver iterations (increase if convergence warnings appear).
    """

    def __init__(self, alpha: float = 0.001, max_iter: int = 5000) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self._pipeline = None

    @property
    def name(self) -> str:
        return "Lasso"

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import Lasso
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=self.alpha, max_iter=self.max_iter, random_state=42)),
        ])
        self._pipeline.fit(X, y)
        n_nonzero = int(np.sum(self._pipeline.named_steps["model"].coef_ != 0))
        logger.debug(f"Lasso trained: alpha={self.alpha}, {n_nonzero} non-zero coefs")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("LassoModel.train() must be called before predict()")
        return self._pipeline.predict(X).astype(np.float64)

    def get_feature_importance(self, feature_names: list[str] | None = None) -> dict | None:
        if self._pipeline is None:
            return None
        coefs = np.abs(self._pipeline.named_steps["model"].coef_)
        if feature_names and len(feature_names) == len(coefs):
            return dict(zip(feature_names, coefs.tolist()))
        return {f"f{i}": float(c) for i, c in enumerate(coefs)}


# ------------------------------------------------------------------ #
# ElasticNet
# ------------------------------------------------------------------ #


class ElasticNetModel(MLModel):
    """ElasticNet (L1 + L2) linear regression with standard scaling.

    Combines Ridge and Lasso regularisation. *l1_ratio* controls the mix:
    0 = pure Ridge, 1 = pure Lasso, values between give a blend.

    Args:
        alpha: Overall regularisation strength.
        l1_ratio: Mix of L1 vs L2 (0 = Ridge, 1 = Lasso).
        max_iter: Maximum solver iterations.
    """

    def __init__(
        self,
        alpha: float = 0.001,
        l1_ratio: float = 0.5,
        max_iter: int = 5000,
    ) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self._pipeline = None

    @property
    def name(self) -> str:
        return "ElasticNet"

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                random_state=42,
            )),
        ])
        self._pipeline.fit(X, y)
        n_nonzero = int(np.sum(self._pipeline.named_steps["model"].coef_ != 0))
        logger.debug(
            f"ElasticNet trained: alpha={self.alpha}, l1_ratio={self.l1_ratio}, "
            f"{n_nonzero} non-zero coefs"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("ElasticNetModel.train() must be called before predict()")
        return self._pipeline.predict(X).astype(np.float64)

    def get_feature_importance(self, feature_names: list[str] | None = None) -> dict | None:
        if self._pipeline is None:
            return None
        coefs = np.abs(self._pipeline.named_steps["model"].coef_)
        if feature_names and len(feature_names) == len(coefs):
            return dict(zip(feature_names, coefs.tolist()))
        return {f"f{i}": float(c) for i, c in enumerate(coefs)}
