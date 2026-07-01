from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: SHAP-based + correlation feature selection — removes redundant features to reduce overfitting in Phase 4.

Connections:
  - src/ml/runner.py: calls select_features() when feature_select=True in run_ticker()
  - src/ml/xgboost_model.py: fast XGBoost surrogate trained internally to generate SHAP values

In:  (X_train, y_train) numpy arrays + feature names list
Out: (X_selected, selected_names) — reduced feature matrix keeping top-k SHAP-important, low-correlation features
"""


# ------------------------------------------------------------------ #
# Correlation filter
# ------------------------------------------------------------------ #


def correlation_filter(
    X: np.ndarray,
    threshold: float = 0.95,
) -> list[int]:
    """Return column indices to retain after dropping near-duplicate features.

    For each pair of features with |Pearson r| > *threshold*, the one with
    lower variance is dropped.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        threshold: Correlation threshold above which one feature is dropped.

    Returns:
        Sorted list of column indices to keep.
    """
    if X.shape[1] <= 1:
        return list(range(X.shape[1]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.corrcoef(X.T)

    n = corr.shape[0]
    variances = X.var(axis=0)
    to_drop = set()

    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if j in to_drop:
                continue
            if abs(corr[i, j]) > threshold:
                # Drop the lower-variance feature
                if variances[i] >= variances[j]:
                    to_drop.add(j)
                else:
                    to_drop.add(i)
                    break  # i is dropped, no need to check further pairs with i

    keep = sorted(set(range(n)) - to_drop)
    logger.debug(
        f"Correlation filter (threshold={threshold}): "
        f"{n} → {len(keep)} features ({n - len(keep)} dropped)"
    )
    return keep


# ------------------------------------------------------------------ #
# SHAP importance
# ------------------------------------------------------------------ #


def shap_importance(
    model,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    max_samples: int = 2000,
) -> pd.Series:
    """Compute mean absolute SHAP values for a trained tree-based model.

    Supports XGBoost, LightGBM, and Random Forest (via TreeExplainer).
    Falls back to permutation importance if SHAP is unavailable.

    Args:
        model: A trained model with a ``_model`` attribute (sklearn/xgb/lgb).
        X: Feature matrix to explain (can be a subset for speed).
        feature_names: Optional list of feature names for the output index.
        max_samples: Maximum samples to pass to SHAP (sub-samples if larger).

    Returns:
        pd.Series of mean |SHAP value| per feature, sorted descending.
        Index is feature names if provided, else integer indices.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — falling back to model's built-in importance")
        return _fallback_importance(model, X, feature_names)

    # Sub-sample for speed
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    # Get the underlying sklearn/xgb/lgb model
    raw_model = getattr(model, "_model", None) or getattr(model, "_pipeline", None)
    if raw_model is None:
        logger.warning("Cannot extract underlying model for SHAP — using fallback")
        return _fallback_importance(model, X, feature_names)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.TreeExplainer(raw_model)
            shap_values = explainer.shap_values(X_sample)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        if feature_names and len(feature_names) == len(mean_abs_shap):
            result = pd.Series(mean_abs_shap, index=feature_names)
        else:
            result = pd.Series(mean_abs_shap, index=[f"f{i}" for i in range(len(mean_abs_shap))])

        return result.sort_values(ascending=False)

    except Exception as exc:
        logger.warning(f"SHAP TreeExplainer failed ({exc}) — using fallback")
        return _fallback_importance(model, X, feature_names)


def _fallback_importance(model, X: np.ndarray, feature_names: list[str] | None) -> pd.Series:
    """Return built-in feature importances when SHAP is unavailable."""
    imp = model.get_feature_importance(feature_names)
    if imp is None:
        n = X.shape[1]
        imp = {f"f{i}": 1.0 / n for i in range(n)}

    s = pd.Series(imp)
    return s.sort_values(ascending=False)


# ------------------------------------------------------------------ #
# Combined pipeline
# ------------------------------------------------------------------ #


def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    top_k: int = 50,
    corr_threshold: float = 0.95,
    shap_surrogate: str = "xgboost",
) -> tuple[np.ndarray, list[str]]:
    """Select the top-k most predictive features using SHAP + correlation filter.

    Steps:
    1. Apply correlation filter to remove near-duplicate features.
    2. Train a fast surrogate model (XGBoost by default) on the filtered features.
    3. Compute SHAP importances and keep the top-*k* features.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        feature_names: List of feature column names (length == X_train.shape[1]).
        top_k: Number of features to retain after SHAP ranking.
        corr_threshold: Pearson |r| threshold for the correlation filter step.
        shap_surrogate: Model type for SHAP surrogate ("xgboost" or "random_forest").

    Returns:
        (X_selected, selected_names):
            X_selected — column-filtered training matrix.
            selected_names — list of retained feature names.
    """
    n_orig = X_train.shape[1]
    logger.info(
        f"Feature selection: {n_orig} features → target top_k={top_k}, "
        f"corr_threshold={corr_threshold}"
    )

    # Step 1: correlation filter
    keep_idx = correlation_filter(X_train, threshold=corr_threshold)
    X_filtered = X_train[:, keep_idx]
    filtered_names = [feature_names[i] for i in keep_idx]

    # If already at or below top_k, skip SHAP
    if X_filtered.shape[1] <= top_k:
        logger.info(f"Feature selection complete: {X_filtered.shape[1]} features (no SHAP needed)")
        return X_filtered, filtered_names

    # Step 2: train a fast surrogate for SHAP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if shap_surrogate == "xgboost":
            from src.ml.xgboost_model import XGBoostModel
            surrogate = XGBoostModel(n_estimators=200, max_depth=5)
        else:
            from src.ml.random_forest import RandomForestModel
            surrogate = RandomForestModel(n_estimators=100)

        surrogate.train(X_filtered, y_train)

    # Step 3: compute SHAP and rank
    importance = shap_importance(surrogate, X_filtered, filtered_names)

    # Keep top_k by SHAP rank
    top_names = importance.head(top_k).index.tolist()
    top_col_idx = [filtered_names.index(n) for n in top_names if n in filtered_names]

    X_selected = X_filtered[:, top_col_idx]
    selected_names = [filtered_names[i] for i in top_col_idx]

    logger.info(
        f"Feature selection complete: {n_orig} → {len(selected_names)} features "
        f"(corr_filter kept {len(filtered_names)}, SHAP kept {len(selected_names)})"
    )
    return X_selected, selected_names


def get_shap_summary(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
    top_k: int = 20,
) -> pd.DataFrame:
    """Build a SHAP summary table for the top-k features on the test set.

    Args:
        model: Trained MLModel.
        X_test: Test feature matrix.
        feature_names: Feature names.
        top_k: Number of top features to include.

    Returns:
        DataFrame with columns: feature, mean_abs_shap, rank.
    """
    importance = shap_importance(model, X_test, feature_names)
    top = importance.head(top_k).reset_index()
    top.columns = ["feature", "mean_abs_shap"]
    top["rank"] = range(1, len(top) + 1)
    return top
