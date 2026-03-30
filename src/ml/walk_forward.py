"""Walk-forward cross-validation for the ML Forecasting Engine (Phase 4).

Walk-forward (a.k.a. time-series split) ensures no future data leaks into training
by sliding the train/test boundary forward in time.

Two modes:
  - single_split: one clean train/test boundary (default: 2015-2020 / 2020-2026)
  - walk_forward_splits: expanding-window folds stepping forward by *step_days*

Public API
----------
single_split(X, y, train_end, test_start)
    -> WFSplit

walk_forward_splits(X, y, min_train_days, step_days, n_test_days)
    -> list[WFSplit]

apply_split(X, y, split)
    -> (X_train, y_train, X_test, y_test)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class WFSplit:
    """One train/test split produced by walk-forward validation.

    Attributes:
        train_idx: Integer indices into X/y for the training set.
        test_idx: Integer indices into X/y for the test set.
        train_start: Date of first training sample.
        train_end: Date of last training sample.
        test_start: Date of first test sample.
        test_end: Date of last test sample.
    """

    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)

    def __repr__(self) -> str:
        return (
            f"WFSplit(train={self.train_start.date()}→{self.train_end.date()} "
            f"[{self.n_train}d], "
            f"test={self.test_start.date()}→{self.test_end.date()} "
            f"[{self.n_test}d])"
        )


# ------------------------------------------------------------------ #
# Single split
# ------------------------------------------------------------------ #


def single_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_end: str = "2019-12-31",
    test_start: str = "2020-01-01",
) -> WFSplit:
    """Produce one train/test split at a fixed date boundary.

    Args:
        X: Feature DataFrame with DatetimeIndex.
        y: Target Series with DatetimeIndex aligned to X.
        train_end: Last date (inclusive) to include in the training set.
        test_start: First date (inclusive) to include in the test set.

    Returns:
        :class:`WFSplit` with train and test index arrays.

    Raises:
        ValueError: If either split is empty after applying the boundary.
    """
    idx = X.index
    train_mask = idx <= pd.Timestamp(train_end)
    test_mask = idx >= pd.Timestamp(test_start)

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    if len(train_idx) == 0:
        raise ValueError(
            f"Training set is empty with train_end='{train_end}'. "
            f"Data starts at {idx.min().date()}."
        )
    if len(test_idx) == 0:
        raise ValueError(
            f"Test set is empty with test_start='{test_start}'. "
            f"Data ends at {idx.max().date()}."
        )

    split = WFSplit(
        train_idx=train_idx,
        test_idx=test_idx,
        train_start=idx[train_idx[0]],
        train_end=idx[train_idx[-1]],
        test_start=idx[test_idx[0]],
        test_end=idx[test_idx[-1]],
    )
    logger.info(f"Single split: {split}")
    return split


# ------------------------------------------------------------------ #
# Walk-forward splits (expanding window)
# ------------------------------------------------------------------ #


def walk_forward_splits(
    X: pd.DataFrame,
    y: pd.Series,
    min_train_days: int = 504,
    step_days: int = 63,
    n_test_days: int = 63,
) -> list[WFSplit]:
    """Generate expanding-window walk-forward splits.

    Each fold:
      - Training set: all data from the start up to the current boundary.
      - Test set: the next *n_test_days* trading days after the boundary.
      - The boundary advances by *step_days* for each subsequent fold.

    Args:
        X: Feature DataFrame with DatetimeIndex.
        y: Target Series with DatetimeIndex aligned to X.
        min_train_days: Minimum number of training samples required before
                        the first test fold begins (default 504 ≈ 2 years).
        step_days: Number of days to advance the boundary per fold
                   (default 63 ≈ 1 quarter).
        n_test_days: Number of test samples per fold (default 63 ≈ 1 quarter).

    Returns:
        List of :class:`WFSplit` objects, one per fold.  Empty list if there
        are not enough samples for even one fold.
    """
    n = len(X)
    if n < min_train_days + n_test_days:
        logger.warning(
            f"Not enough data for walk-forward splits "
            f"(need {min_train_days + n_test_days}, got {n})"
        )
        return []

    splits: list[WFSplit] = []
    idx = X.index
    start = min_train_days

    while start + n_test_days <= n:
        end = min(start + n_test_days, n)
        train_idx = np.arange(0, start)
        test_idx = np.arange(start, end)

        splits.append(
            WFSplit(
                train_idx=train_idx,
                test_idx=test_idx,
                train_start=idx[0],
                train_end=idx[start - 1],
                test_start=idx[start],
                test_end=idx[end - 1],
            )
        )
        start += step_days

    logger.info(
        f"Walk-forward splits: {len(splits)} folds, "
        f"min_train={min_train_days}d, step={step_days}d, test={n_test_days}d"
    )
    return splits


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def apply_split(
    X: pd.DataFrame,
    y: pd.Series,
    split: WFSplit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (X_train, y_train, X_test, y_test) numpy arrays from a split.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        split: :class:`WFSplit` with index arrays.

    Returns:
        (X_train, y_train, X_test, y_test) as numpy float64 arrays.
    """
    X_arr = X.values.astype(np.float64)
    y_arr = y.values.astype(np.float64)

    X_train = X_arr[split.train_idx]
    y_train = y_arr[split.train_idx]
    X_test = X_arr[split.test_idx]
    y_test = y_arr[split.test_idx]

    return X_train, y_train, X_test, y_test


def settings_split(X: pd.DataFrame, y: pd.Series, settings=None) -> WFSplit:
    """Produce a single split using the train/test dates from settings.

    Args:
        X: Feature DataFrame with DatetimeIndex.
        y: Target Series.
        settings: Optional settings object; uses default if None.

    Returns:
        :class:`WFSplit` matching the configured train/test boundaries.
    """
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    return single_split(
        X, y,
        train_end=settings.ml_train_end,
        test_start=settings.ml_test_start,
    )
