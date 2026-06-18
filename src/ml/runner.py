"""ML Forecasting Engine runner — orchestrates all models for one or many tickers.

Public API
----------
run_ticker(ticker, settings, target_days, tune, feature_select, deep_learning, save)
    -> list[MLResult]

run_tickers(tickers, settings, ...)
    -> dict[str, list[MLResult]]

comparison_table(results) -> pd.DataFrame
best_result(results)      -> MLResult | None

predict_latest(ticker, model_name, settings, target_days)
    -> float   (predicted next-N-day return for the most recent available data)

save_results(results, ticker, output_dir) -> Path
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import get_settings
from src.ml.base import MLModel, MLResult
from src.ml.feature_engineer import load_and_build
from src.ml.walk_forward import apply_split, settings_split
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


# ------------------------------------------------------------------ #
# Model registry
# ------------------------------------------------------------------ #


def _classical_models() -> list[MLModel]:
    """Instantiate all classical ML models."""
    from src.ml.linear_models import ElasticNetModel, LassoModel, RidgeModel
    from src.ml.random_forest import RandomForestModel
    from src.ml.xgboost_model import LightGBMModel, XGBoostModel

    models = [
        XGBoostModel(),
        LightGBMModel(),
        RandomForestModel(),
        RidgeModel(),
        LassoModel(),
        ElasticNetModel(),
    ]

    try:
        from src.ml.catboost_model import CatBoostModel
        models.append(CatBoostModel())
    except ImportError:
        pass

    return models


def _deep_learning_models() -> list[MLModel]:
    """Instantiate all PyTorch deep learning models."""
    from src.ml.lstm_model import GRUModel, LSTMModel
    from src.ml.transformer_model import TransformerModel

    return [
        LSTMModel(),
        GRUModel(),
        TransformerModel(),
    ]


def _ensemble_models() -> list[MLModel]:
    """Instantiate ensemble models (run after base models are done)."""
    from src.ml.ensemble_model import StackingEnsemble
    return [StackingEnsemble()]


# ------------------------------------------------------------------ #
# Core runner
# ------------------------------------------------------------------ #


def _run_model(
    model: MLModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ticker: str,
    feature_names: list[str],
    split,
) -> MLResult:
    """Train one model and return its MLResult. Catches all exceptions."""
    logger.info(f"[{ticker}] Training {model.name}...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.train(X_train, y_train)

        result = model.evaluate(
            X_test, y_test,
            ticker=ticker,
            feature_names=feature_names,
            train_start=str(split.train_start.date()),
            train_end=str(split.train_end.date()),
            test_start=str(split.test_start.date()),
            test_end=str(split.test_end.date()),
        )
        logger.info(
            f"[{ticker}] {model.name:20s}  RMSE={result.rmse:.5f}  "
            f"MAE={result.mae:.5f}  MAPE={result.mape:.2%}"
        )
        return result

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.warning(f"[{ticker}] {model.name} failed: {error_msg}")
        return MLResult(
            model_name=model.name,
            ticker=ticker,
            error=error_msg,
            train_start=str(split.train_start.date()),
            train_end=str(split.train_end.date()),
            test_start=str(split.test_start.date()),
            test_end=str(split.test_end.date()),
        )


def _sanitize_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ticker: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove NaN/inf from train rows; impute NaN/inf in test with train medians.

    This is a safety net — feature_engineer should produce clean arrays, but
    edge cases (very short histories, all-zero volume windows, pandas NA in
    integer columns) can still produce NaN after the float64 cast.
    """
    # ---- Training: drop any row with NaN or inf ----
    train_finite = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    n_bad_train = int((~train_finite).sum())
    if n_bad_train:
        logger.warning(f"[{ticker}] Dropping {n_bad_train} training rows with NaN/inf")
        X_train = X_train[train_finite]
        y_train = y_train[train_finite]

    if len(X_train) == 0:
        raise ValueError(f"[{ticker}] No valid training rows after NaN cleanup")

    # ---- Test: impute NaN/inf columns using training column medians ----
    col_medians = np.nanmedian(X_train, axis=0)
    X_test = X_test.copy()
    for j in range(X_test.shape[1]):
        bad = ~np.isfinite(X_test[:, j])
        if bad.any():
            fill = col_medians[j] if np.isfinite(col_medians[j]) else 0.0
            X_test[bad, j] = fill

    return X_train, y_train, X_test, y_test


def run_all_models(
    ticker: str,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    settings=None,
    tune: bool = False,
    feature_select: bool = True,
    deep_learning: bool = True,
    save: bool = True,
) -> list[MLResult]:
    """Run all ML models on pre-built feature data for *ticker*.

    Args:
        ticker: Stock symbol.
        X: Feature DataFrame (output of feature_engineer).
        y: Target Series.
        feature_names: Feature column names.
        settings: Optional settings override.
        tune: If True, run Optuna tuning before training classical models.
        feature_select: If True, apply correlation + SHAP feature selection.
        deep_learning: If True, include LSTM/GRU/Transformer models.
        save: If True, save trained models and results to disk.

    Returns:
        List of :class:`MLResult`, one per model.
    """
    if settings is None:
        settings = get_settings()

    # Train/test split
    split = settings_split(X, y, settings)
    X_train, y_train, X_test, y_test = apply_split(X, y, split)
    X_train, y_train, X_test, y_test = _sanitize_arrays(
        X_train, y_train, X_test, y_test, ticker
    )
    feat_names = list(feature_names)

    logger.info(
        f"[{ticker}] Data split: "
        f"train={split.n_train} samples, test={split.n_test} samples, "
        f"features={len(feat_names)}"
    )

    # Feature selection
    if feature_select and len(feat_names) > 50:
        try:
            from src.ml.feature_selection import select_features
            X_train, feat_names = select_features(X_train, y_train, feat_names)
            # Apply same column mask to test set
            col_idx = [list(feature_names).index(n) for n in feat_names]
            X_test = X_test[:, col_idx]
            logger.info(f"[{ticker}] Features after selection: {len(feat_names)}")
        except Exception as exc:
            logger.warning(f"[{ticker}] Feature selection failed: {exc} — using all features")

    # Decide which models to run
    models = _classical_models()
    if deep_learning:
        models += _deep_learning_models()
    models += _ensemble_models()

    # Optuna tuning for classical models (deep learning always uses defaults or own early stopping)
    if tune:
        tuned_models = []
        for model in models:
            if model.name in ("XGBoost", "LightGBM", "Random Forest",
                              "Ridge", "Lasso", "ElasticNet"):
                try:
                    from src.ml.tuner import get_tuned_model
                    tuned = get_tuned_model(
                        model.name.lower().replace(" ", "_"),
                        X_train, y_train,
                        n_trials=30,
                        timeout=120,
                    )
                    tuned_models.append(tuned)
                except Exception as exc:
                    logger.warning(f"[{ticker}] Tuning {model.name} failed: {exc}")
                    tuned_models.append(model)
            else:
                tuned_models.append(model)
        models = tuned_models

    # Run all models
    results: list[MLResult] = []
    for model in models:
        result = _run_model(
            model, X_train, y_train, X_test, y_test,
            ticker, feat_names, split
        )
        results.append(result)

        # Save model artifact
        if save and result.succeeded:
            _save_model_artifact(model, ticker, settings)

    # Save results summary
    if save:
        save_results(results, ticker)

    return results


def run_ticker(
    ticker: str,
    settings=None,
    target_days: int = 1,
    tune: bool = False,
    feature_select: bool = True,
    deep_learning: bool = True,
    save: bool = True,
) -> list[MLResult]:
    """Load data, build features, and run all ML models for *ticker*.

    Args:
        ticker: Stock symbol (e.g. "AAPL").
        settings: Optional settings override.
        target_days: Forward return horizon (1=next-day, 5=next-week, 20=next-month).
        tune: Enable Optuna hyperparameter search.
        feature_select: Enable SHAP + correlation feature selection.
        deep_learning: Include LSTM/GRU/Transformer models.
        save: Persist model artifacts and results to disk.

    Returns:
        List of :class:`MLResult` objects, one per model.
    """
    if settings is None:
        settings = get_settings()

    logger.info(f"[{ticker}] ML pipeline: target_days={target_days}, tune={tune}, "
                f"feature_select={feature_select}, deep_learning={deep_learning}")

    try:
        X, y, feature_names = load_and_build(ticker, settings, target_days=target_days)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        raise

    return run_all_models(
        ticker, X, y, feature_names,
        settings=settings,
        tune=tune,
        feature_select=feature_select,
        deep_learning=deep_learning,
        save=save,
    )


def run_tickers(
    tickers: list[str],
    settings=None,
    **kwargs,
) -> dict[str, list[MLResult]]:
    """Run the ML pipeline for each ticker in *tickers*.

    Args:
        tickers: List of stock symbols.
        settings: Optional settings override.
        **kwargs: Forwarded to :func:`run_ticker`.

    Returns:
        Dict mapping ticker -> list of MLResult.
    """
    if settings is None:
        settings = get_settings()

    output: dict[str, list[MLResult]] = {}
    for ticker in tickers:
        try:
            output[ticker] = run_ticker(ticker, settings=settings, **kwargs)
        except Exception as exc:
            logger.error(f"[{ticker}] pipeline failed: {exc}")
    return output


# ------------------------------------------------------------------ #
# Prediction for live use
# ------------------------------------------------------------------ #


def predict_latest(
    ticker: str,
    model_name: str = "XGBoost",
    settings=None,
    target_days: int = 1,
) -> float | None:
    """Return the predicted next-N-day return using the most recent data row.

    Loads stored feature data, trains (or loads) the specified model on the
    full available history, then predicts using the latest feature vector.

    Args:
        ticker: Stock symbol.
        model_name: Model to use (must be a valid model name).
        settings: Optional settings override.
        target_days: Forward return horizon.

    Returns:
        Predicted percentage return (e.g. 0.012 = +1.2%), or None on failure.
    """
    if settings is None:
        settings = get_settings()

    try:
        X, y, feature_names = load_and_build(ticker, settings, target_days=target_days)
    except Exception as exc:
        logger.error(f"[{ticker}] predict_latest: {exc}")
        return None

    X_arr = X.values.astype(np.float64)
    y_arr = y.values.astype(np.float64)

    # Train on full history
    from src.ml.tuner import _build_model as build_model
    try:
        model = build_model(model_name.lower().replace(" ", "_"), {})
    except ValueError:
        from src.ml.xgboost_model import XGBoostModel
        model = XGBoostModel()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train(X_arr, y_arr)

    # Predict on the latest row
    preds = model.predict(X_arr[-1:])
    valid = preds[~np.isnan(preds)]
    if len(valid) == 0:
        return None
    return float(valid[-1])


# ------------------------------------------------------------------ #
# Output helpers
# ------------------------------------------------------------------ #


def comparison_table(results: list[MLResult]) -> pd.DataFrame:
    """Build a human-readable comparison DataFrame from model results.

    Args:
        results: Output of :func:`run_ticker` or :func:`run_all_models`.

    Returns:
        DataFrame sorted ascending by RMSE. Failures shown at bottom.
    """
    rows = []
    for r in results:
        rows.append({
            "Model": r.model_name,
            "RMSE": round(r.rmse, 5) if r.rmse != float("inf") else "—",
            "MAE": round(r.mae, 5) if r.mae != float("inf") else "—",
            "MAPE": f"{r.mape:.2%}" if r.mape != float("inf") else "—",
            "Test Period": (
                f"{r.test_start} → {r.test_end}"
                if r.test_start else "—"
            ),
            "Status": "OK" if r.error is None else f"ERR: {r.error[:60]}",
        })

    df = pd.DataFrame(rows)
    ok_mask = df["Status"] == "OK"
    ok_df = df[ok_mask].copy()
    err_df = df[~ok_mask].copy()

    ok_df["_sort"] = pd.to_numeric(ok_df["RMSE"], errors="coerce")
    ok_df = ok_df.sort_values("_sort").drop(columns="_sort")

    return pd.concat([ok_df, err_df], ignore_index=True)


def best_result(results: list[MLResult]) -> MLResult | None:
    """Return the MLResult with the lowest finite RMSE."""
    valid = [r for r in results if r.error is None and r.rmse != float("inf")]
    if not valid:
        return None
    return min(valid, key=lambda r: r.rmse)


def save_results(
    results: list[MLResult],
    ticker: str,
    output_dir: Path | None = None,
) -> Path:
    """Persist ML results to a Parquet file.

    Args:
        results: List of MLResult for *ticker*.
        ticker: Ticker symbol used in the filename.
        output_dir: Directory for output (defaults to data/ml/results/).

    Returns:
        Path to the written Parquet file.
    """
    if output_dir is None:
        settings = get_settings()
        output_dir = settings.data_dir / "ml" / "results"

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)
    out_path = output_dir / f"{ticker}_ml_results.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"[{ticker}] Saved ML results to {out_path}")
    return out_path


def _save_model_artifact(model: MLModel, ticker: str, settings) -> None:
    """Save a trained model artifact to the models directory."""
    models_dir = settings.data_dir / "ml" / "models"
    safe_name = model.name.lower().replace(" ", "_")

    # Prefer .pt for PyTorch models, .pkl for others
    ext = ".pt" if model.name in ("LSTM", "GRU", "Transformer") else ".pkl"
    path = models_dir / ticker / f"{safe_name}{ext}"

    try:
        model.save(path)
    except Exception as exc:
        logger.warning(f"[{ticker}] Could not save {model.name} artifact: {exc}")
