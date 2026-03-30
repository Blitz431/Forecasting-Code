"""XGBoost and LightGBM gradient-boosting models for Phase 4 ML Engine.

Both models implement the MLModel ABC. XGBoost uses GPU acceleration
(device="cuda") when a CUDA GPU is available; LightGBM uses GPU via
device="gpu" if available.

Classes
-------
XGBoostModel  — XGBoost regressor (GPU-accelerated on 3060 Ti)
LightGBMModel — LightGBM regressor (GPU-accelerated when available)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ml.base import MLModel, MLResult
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ------------------------------------------------------------------ #
# XGBoost
# ------------------------------------------------------------------ #


class XGBoostModel(MLModel):
    """XGBoost gradient-boosting regressor.

    Uses CUDA GPU acceleration on the 3060 Ti when available.

    Args:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage.
        subsample: Row subsampling ratio per tree.
        colsample_bytree: Feature subsampling ratio per tree.
        min_child_weight: Minimum sum of instance weight in a child.
        reg_alpha: L1 regularisation.
        reg_lambda: L2 regularisation.
        early_stopping_rounds: Stop if val metric doesn't improve.
        use_gpu: If True, attempt CUDA acceleration (auto-detected).
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        use_gpu: bool | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.use_gpu = _cuda_available() if use_gpu is None else use_gpu
        self._model = None
        self._feature_names: list[str] | None = None

    @property
    def name(self) -> str:
        return "XGBoost"

    def _build_params(self) -> dict:
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
        if self.use_gpu:
            params["device"] = "cuda"
            logger.debug("XGBoost: using CUDA GPU")
        return params

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        import xgboost as xgb

        # Use 10% of training data as an internal validation set for early stopping
        val_size = max(1, int(len(X) * 0.1))
        X_fit, X_val = X[:-val_size], X[-val_size:]
        y_fit, y_val = y[:-val_size], y[-val_size:]

        self._model = xgb.XGBRegressor(
            **self._build_params(),
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric="rmse",
            verbosity=0,
        )
        self._model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        logger.debug(f"XGBoost trained: {self._model.best_iteration} rounds")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("XGBoostModel.train() must be called before predict()")
        return self._model.predict(X).astype(np.float64)

    def get_feature_importance(self, feature_names: list[str] | None = None) -> dict | None:
        if self._model is None:
            return None
        scores = self._model.feature_importances_
        if feature_names and len(feature_names) == len(scores):
            return dict(zip(feature_names, scores.tolist()))
        return {f"f{i}": float(s) for i, s in enumerate(scores)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        logger.debug(f"XGBoost model saved to {path}")


# ------------------------------------------------------------------ #
# LightGBM
# ------------------------------------------------------------------ #


class LightGBMModel(MLModel):
    """LightGBM gradient-boosting regressor.

    Faster than XGBoost for large feature counts; GPU-accelerated when available.

    Args:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth (-1 = unlimited).
        learning_rate: Step size shrinkage.
        num_leaves: Maximum number of leaves per tree.
        subsample: Row subsampling ratio.
        colsample_bytree: Feature subsampling ratio.
        reg_alpha: L1 regularisation.
        reg_lambda: L2 regularisation.
        min_child_samples: Minimum data in a leaf.
        use_gpu: If True, attempt GPU acceleration.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_samples: int = 20,
        use_gpu: bool | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
        self.use_gpu = _cuda_available() if use_gpu is None else use_gpu
        self._model = None

    @property
    def name(self) -> str:
        return "LightGBM"

    def _build_params(self) -> dict:
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_samples": self.min_child_samples,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        if self.use_gpu:
            params["device"] = "gpu"
            logger.debug("LightGBM: using GPU")
        return params

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        import lightgbm as lgb

        val_size = max(1, int(len(X) * 0.1))
        X_fit, X_val = X[:-val_size], X[-val_size:]
        y_fit, y_val = y[:-val_size], y[-val_size:]

        self._model = lgb.LGBMRegressor(**self._build_params())
        self._model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        logger.debug(f"LightGBM trained: {self._model.best_iteration_} rounds")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LightGBMModel.train() must be called before predict()")
        return self._model.predict(X).astype(np.float64)

    def get_feature_importance(self, feature_names: list[str] | None = None) -> dict | None:
        if self._model is None:
            return None
        scores = self._model.feature_importances_
        if feature_names and len(feature_names) == len(scores):
            return dict(zip(feature_names, scores.tolist()))
        return {f"f{i}": float(s) for i, s in enumerate(scores)}
