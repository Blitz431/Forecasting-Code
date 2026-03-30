"""Optuna hyperparameter optimization for Phase 4 ML models.

Each model type has a defined search space. Tuning uses time-series-safe
cross-validation (single split with a small internal val set) to prevent
data leakage.

Public API
----------
tune(model_name, X_train, y_train, n_trials, timeout)
    -> dict   (best hyperparameter dict)

get_tuned_model(model_name, X_train, y_train, n_trials, timeout)
    -> MLModel (trained with best hyperparameters)
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


# ------------------------------------------------------------------ #
# Search space definitions
# ------------------------------------------------------------------ #


def _xgboost_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _lightgbm_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _rf_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
    }


def _ridge_space(trial) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True)}


def _lasso_space(trial) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-6, 1.0, log=True)}


def _elasticnet_space(trial) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 1e-6, 1.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }


def _lstm_space(trial) -> dict:
    return {
        "seq_len": trial.suggest_categorical("seq_len", [10, 20, 40]),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def _gru_space(trial) -> dict:
    return _lstm_space(trial)


def _transformer_space(trial) -> dict:
    d_model_choices = [32, 64, 128]
    nhead_map = {32: [2, 4], 64: [4, 8], 128: [4, 8]}
    d_model = trial.suggest_categorical("d_model", d_model_choices)
    nhead = trial.suggest_categorical("nhead", nhead_map[d_model])
    return {
        "seq_len": trial.suggest_categorical("seq_len", [10, 20, 40]),
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [128, 256, 512]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


_SEARCH_SPACES: dict[str, Callable] = {
    "xgboost": _xgboost_space,
    "lightgbm": _lightgbm_space,
    "random_forest": _rf_space,
    "ridge": _ridge_space,
    "lasso": _lasso_space,
    "elasticnet": _elasticnet_space,
    "lstm": _lstm_space,
    "gru": _gru_space,
    "transformer": _transformer_space,
}


# ------------------------------------------------------------------ #
# Model factory
# ------------------------------------------------------------------ #


def _build_model(model_name: str, params: dict):
    """Instantiate the correct MLModel subclass with *params*."""
    name = model_name.lower()
    if name == "xgboost":
        from src.ml.xgboost_model import XGBoostModel
        return XGBoostModel(**params)
    if name == "lightgbm":
        from src.ml.xgboost_model import LightGBMModel
        return LightGBMModel(**params)
    if name == "random_forest":
        from src.ml.random_forest import RandomForestModel
        return RandomForestModel(**params)
    if name == "ridge":
        from src.ml.linear_models import RidgeModel
        return RidgeModel(**params)
    if name == "lasso":
        from src.ml.linear_models import LassoModel
        return LassoModel(**params)
    if name == "elasticnet":
        from src.ml.linear_models import ElasticNetModel
        return ElasticNetModel(**params)
    if name == "lstm":
        from src.ml.lstm_model import LSTMModel
        return LSTMModel(**params)
    if name == "gru":
        from src.ml.lstm_model import GRUModel
        return GRUModel(**params)
    if name == "transformer":
        from src.ml.transformer_model import TransformerModel
        return TransformerModel(**params)
    raise ValueError(f"Unknown model name: '{model_name}'. "
                     f"Valid: {list(_SEARCH_SPACES.keys())}")


# ------------------------------------------------------------------ #
# Objective function
# ------------------------------------------------------------------ #


def _make_objective(model_name: str, X_train: np.ndarray, y_train: np.ndarray):
    """Return an Optuna objective function for *model_name*."""
    space_fn = _SEARCH_SPACES[model_name.lower()]

    # Internal val split: last 20% of training data (time-ordered)
    n = len(X_train)
    val_size = max(10, int(n * 0.2))
    X_fit, X_val = X_train[:-val_size], X_train[-val_size:]
    y_fit, y_val = y_train[:-val_size], y_train[-val_size:]

    def objective(trial):
        params = space_fn(trial)
        model = _build_model(model_name, params)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.train(X_fit, y_fit)
                preds = model.predict(X_val)

            # Strip NaN padding from RNN models
            valid = ~np.isnan(preds)
            if valid.sum() == 0:
                return float("inf")
            rmse = float(np.sqrt(np.mean((y_val[valid] - preds[valid]) ** 2)))
            return rmse
        except Exception as exc:
            logger.debug(f"[{model_name}] trial failed: {exc}")
            return float("inf")

    return objective


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #


def tune(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 50,
    timeout: int = 300,
) -> dict:
    """Run Optuna hyperparameter search for *model_name*.

    Args:
        model_name: One of: xgboost, lightgbm, random_forest, ridge, lasso,
                    elasticnet, lstm, gru, transformer.
        X_train: Training feature matrix.
        y_train: Training target vector.
        n_trials: Maximum number of Optuna trials.
        timeout: Maximum wall-clock seconds to spend tuning.

    Returns:
        Dict of best hyperparameters found.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    model_key = model_name.lower()
    if model_key not in _SEARCH_SPACES:
        raise ValueError(
            f"No search space for '{model_name}'. Valid: {list(_SEARCH_SPACES.keys())}"
        )

    objective = _make_objective(model_key, X_train, y_train)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    logger.info(
        f"[{model_name}] Optuna search: n_trials={n_trials}, timeout={timeout}s"
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best = study.best_params
    logger.info(
        f"[{model_name}] Best trial: RMSE={study.best_value:.6f}, params={best}"
    )
    return best


def get_tuned_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 50,
    timeout: int = 300,
):
    """Tune then train and return a model with the best hyperparameters.

    Args:
        model_name: Model identifier (see :func:`tune`).
        X_train: Training feature matrix.
        y_train: Training target vector.
        n_trials: Optuna trial budget.
        timeout: Maximum tuning seconds.

    Returns:
        A fitted :class:`MLModel` using the best found hyperparameters.
    """
    best_params = tune(model_name, X_train, y_train, n_trials=n_trials, timeout=timeout)
    model = _build_model(model_name, best_params)
    logger.info(f"[{model_name}] Training with best params: {best_params}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train(X_train, y_train)
    return model
