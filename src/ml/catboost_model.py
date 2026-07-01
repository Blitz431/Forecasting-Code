from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ml.base import MLModel
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: CatBoost gradient-boosting model with ordered boosting — reduces overfitting on time-series data for Phase 4.

Connections:
  - src/ml/base.py: subclasses MLModel, returns MLResult
  - src/ml/runner.py: instantiated and trained via run_ticker()
  - src/ml/ensemble_model.py: used as one of the base models in StackingEnsemble

In:  (X_train, y_train) numpy arrays
Out: MLResult with predictions, RMSE/MAE/MAPE; .pkl saved to disk
"""


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class CatBoostModel(MLModel):
    """CatBoost gradient-boosting regressor.

    Uses ordered boosting mode and GPU acceleration when available.

    Args:
        iterations: Number of boosting rounds.
        depth: Tree depth.
        learning_rate: Step size shrinkage.
        l2_leaf_reg: L2 regularisation coefficient.
        early_stopping_rounds: Stop if validation metric doesn't improve.
        use_gpu: If True, attempt CUDA acceleration (auto-detected).
    """

    def __init__(
        self,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        early_stopping_rounds: int = 50,
        use_gpu: bool | None = None,
    ) -> None:
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.early_stopping_rounds = early_stopping_rounds
        self.use_gpu = _cuda_available() if use_gpu is None else use_gpu
        self._model = None
        self._feature_names: list[str] | None = None

    @property
    def name(self) -> str:
        return "CatBoost"

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            from catboost import CatBoostRegressor, Pool
        except ImportError:
            raise ImportError("catboost not installed. Run: pip install catboost")

        val_size = max(1, int(len(X) * 0.1))
        X_fit, X_val = X[:-val_size], X[-val_size:]
        y_fit, y_val = y[:-val_size], y[-val_size:]

        task_type = "GPU" if self.use_gpu else "CPU"
        if self.use_gpu:
            logger.debug("CatBoost: using GPU")

        self._model = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            task_type=task_type,
            random_seed=42,
            verbose=False,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric="RMSE",
            boosting_type="Ordered",
        )
        self._model.fit(
            Pool(X_fit, y_fit),
            eval_set=Pool(X_val, y_val),
        )
        logger.debug(f"CatBoost trained: {self._model.best_iteration_} rounds")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("CatBoostModel.train() must be called before predict()")
        return self._model.predict(X).astype(np.float64)

    def get_feature_importance(self, feature_names: list[str] | None = None) -> dict | None:
        if self._model is None:
            return None
        scores = self._model.get_feature_importance()
        names = feature_names if (feature_names and len(feature_names) == len(scores)) \
            else [f"f{i}" for i in range(len(scores))]
        return dict(zip(names, scores.tolist()))

    def save(self, path: Path) -> None:
        if self._model is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        logger.debug(f"CatBoost model saved to {path}")
