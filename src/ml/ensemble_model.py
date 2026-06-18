"""Stacking ensemble model for Phase 4 ML Engine.

Trains a set of base models (XGBoost, LightGBM, CatBoost, Random Forest)
and fits a Ridge meta-model on their out-of-fold predictions. This reduces
the variance of any single model and typically beats all individual models.

The stacking uses 5-fold time-series cross-validation (no data leakage —
each fold's meta-features are predicted by a model that never saw those rows).
"""

from __future__ import annotations

import numpy as np

from src.ml.base import MLModel
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

_N_FOLDS = 5


class StackingEnsemble(MLModel):
    """Stacking ensemble: base models + Ridge meta-learner.

    Base models: XGBoost, LightGBM, CatBoost, Random Forest.
    Meta-learner: Ridge regression trained on out-of-fold base predictions.

    Args:
        use_catboost: Include CatBoost in base models (requires catboost package).
        n_folds: Number of time-series CV folds for generating meta-features.
    """

    def __init__(
        self,
        use_catboost: bool = True,
        n_folds: int = _N_FOLDS,
    ) -> None:
        self.use_catboost = use_catboost
        self.n_folds = n_folds
        self._base_models: list[MLModel] = []
        self._meta_model = None
        self._feature_names: list[str] | None = None

    @property
    def name(self) -> str:
        return "Stacking Ensemble"

    def _build_base_models(self) -> list[MLModel]:
        from src.ml.xgboost_model import XGBoostModel, LightGBMModel
        from src.ml.random_forest import RandomForestModel

        models: list[MLModel] = [
            XGBoostModel(),
            LightGBMModel(),
            RandomForestModel(),
        ]
        if self.use_catboost:
            try:
                from src.ml.catboost_model import CatBoostModel
                models.append(CatBoostModel())
            except ImportError:
                logger.warning("CatBoost not installed — skipping in ensemble")
        return models

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import Ridge

        n = len(X)
        self._base_models = self._build_base_models()
        n_base = len(self._base_models)

        # Out-of-fold predictions matrix: (n_samples, n_base_models)
        oof = np.full((n, n_base), np.nan)

        # Time-series folds — always train on past, validate on future
        fold_size = n // self.n_folds
        if fold_size < 10:
            # Not enough data for stacking — fall back to simple average
            logger.warning(
                f"[Stacking] Only {n} samples — too few for {self.n_folds}-fold CV. "
                "Training base models on full data without stacking."
            )
            self._train_base_full(X, y)
            self._meta_model = None
            return

        for fold in range(self.n_folds):
            val_start = (fold + 1) * fold_size
            val_end = val_start + fold_size if fold < self.n_folds - 1 else n
            if val_start >= n:
                break

            X_tr, y_tr = X[:val_start], y[:val_start]
            X_val = X[val_start:val_end]

            for j, model in enumerate(self._base_models):
                try:
                    # Fresh instance per fold so weights don't bleed across folds
                    fold_model = model.__class__()
                    fold_model.train(X_tr, y_tr)
                    oof[val_start:val_end, j] = fold_model.predict(X_val)
                except Exception as exc:
                    logger.warning(f"[Stacking] fold={fold} model={model.name} failed: {exc}")

        # Fill any remaining NaN OOF slots with column medians
        for j in range(n_base):
            col = oof[:, j]
            finite = col[np.isfinite(col)]
            fill = float(np.median(finite)) if len(finite) else 0.0
            col[~np.isfinite(col)] = fill

        # Train meta-model on OOF predictions
        self._meta_model = Ridge(alpha=1.0)
        self._meta_model.fit(oof, y)
        logger.debug(
            f"[Stacking] Meta-model trained on OOF ({n} samples, {n_base} base models)"
        )

        # Retrain all base models on full training data for final predictions
        self._train_base_full(X, y)

    def _train_base_full(self, X: np.ndarray, y: np.ndarray) -> None:
        for model in self._base_models:
            try:
                model.train(X, y)
                logger.debug(f"[Stacking] Base model {model.name} trained on full data")
            except Exception as exc:
                logger.warning(f"[Stacking] Base model {model.name} full-train failed: {exc}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._base_models:
            raise RuntimeError("StackingEnsemble.train() must be called before predict()")

        base_preds = np.column_stack([
            _safe_predict(m, X) for m in self._base_models
        ])

        if self._meta_model is not None:
            return self._meta_model.predict(base_preds).astype(np.float64)

        # Fallback: simple mean of base predictions
        return np.nanmean(base_preds, axis=1).astype(np.float64)

    def get_feature_importance(self, feature_names: list[str] | None = None) -> dict | None:
        # Average feature importances across base models that support it
        importances: list[dict] = []
        for model in self._base_models:
            fi = model.get_feature_importance(feature_names)
            if fi:
                importances.append(fi)
        if not importances:
            return None

        all_keys = set().union(*importances)
        averaged = {
            k: float(np.mean([d.get(k, 0.0) for d in importances]))
            for k in all_keys
        }
        total = sum(averaged.values())
        if total > 0:
            averaged = {k: v / total for k, v in averaged.items()}
        return averaged


def _safe_predict(model: MLModel, X: np.ndarray) -> np.ndarray:
    try:
        preds = model.predict(X)
        # Replace NaN/inf with 0 so they don't poison the meta-model
        bad = ~np.isfinite(preds)
        if bad.any():
            preds[bad] = 0.0
        return preds
    except Exception as exc:
        logger.warning(f"[Stacking] predict failed for {model.name}: {exc}")
        return np.zeros(len(X))
