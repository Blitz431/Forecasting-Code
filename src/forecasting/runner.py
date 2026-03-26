"""Forecast runner — orchestrates all 12 methods for one or many tickers.

Public API
----------
run_all_methods(series, horizons, holdout)
    -> list[ForecastResult]          (one result per method)

run_ticker(ticker, settings, horizons, holdout)
    -> list[ForecastResult]          (loads data + calls run_all_methods)

run_tickers(tickers, settings, horizons, holdout)
    -> dict[str, list[ForecastResult]]

comparison_table(results)
    -> pd.DataFrame                  (sorted by RMSE for easy printing)

save_forecasts(results, ticker, output_dir)
    -> Path                          (Parquet file path)
"""

from __future__ import annotations

import traceback
import warnings
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.forecasting.auto_best import AutoBest
from src.forecasting.base import ForecastMethod, ForecastResult
from src.forecasting.decomposition import (
    AdditiveDecomposition,
    EnsembleDecomposition,
    FlatTrendDecomposition,
    LinearTrendDecomposition,
    MultiplicativeDecomposition,
)
from src.forecasting.exponential_smoothing import (
    HoltLinearTrend,
    HoltWinters,
    SimpleExpSmoothing,
)
from src.forecasting.metrics import metrics_summary
from src.forecasting.moving_average import SimpleMovingAverage, WeightedMovingAverage
from src.forecasting.regression import OLSRegression
from src.scraper.storage import get_ticker_filepath, load_dataframe
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# Ordered list of all 12 methods — index + 1 == method number
ALL_METHODS: list[type[ForecastMethod]] = [
    AdditiveDecomposition,       # 1
    MultiplicativeDecomposition, # 2
    FlatTrendDecomposition,      # 3
    LinearTrendDecomposition,    # 4
    EnsembleDecomposition,       # 5
    SimpleMovingAverage,         # 6
    WeightedMovingAverage,       # 7
    SimpleExpSmoothing,          # 8
    HoltLinearTrend,             # 9
    HoltWinters,                 # 10
    OLSRegression,               # 11
    AutoBest,                    # 12
]


# ------------------------------------------------------------------ #
# Core runner
# ------------------------------------------------------------------ #


def run_all_methods(
    series: pd.Series,
    horizons: int = 4,
    holdout: int = 8,
) -> list[ForecastResult]:
    """Run all 12 forecast methods on *series*.

    Each method is instantiated fresh, evaluated on a holdout window to
    compute RMSE/MAE/MAPE, then refit on the full series to produce the
    final *horizons*-quarter forecast.  Methods that raise exceptions
    produce a result with ``error`` set and ``rmse=inf``.

    Args:
        series: Quarterly close-price series (DatetimeIndex, no NaNs).
        horizons: Number of future quarters to forecast.
        holdout: Number of trailing quarters used for error estimation.

    Returns:
        List of :class:`ForecastResult`, one per method (length 12).
    """
    results: list[ForecastResult] = []

    for method_cls in ALL_METHODS:
        method = method_cls()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = method.evaluate(series, holdout=holdout, horizons=horizons)
            results.append(result)
            logger.debug(
                f"[{method.name:40s}]  RMSE={result.rmse:8.2f}  "
                f"MAE={result.mae:8.2f}  MAPE={result.mape:.1%}"
            )
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.warning(f"[{method.name}] failed — {error_msg}")
            results.append(
                ForecastResult(
                    method_name=method.name,
                    method_number=method.number,
                    forecasts=pd.Series(dtype=float),
                    fitted_values=pd.Series(dtype=float),
                    rmse=float("inf"),
                    mae=float("inf"),
                    mape=float("inf"),
                    error=error_msg,
                )
            )

    return results


# ------------------------------------------------------------------ #
# Ticker-level helpers
# ------------------------------------------------------------------ #


def _load_quarterly_series(ticker: str, settings) -> pd.Series:
    """Load quarterly close prices for *ticker*."""
    filepath = get_ticker_filepath(ticker, settings.raw_quarterly_dir)
    df = load_dataframe(filepath)

    if df.empty:
        raise FileNotFoundError(
            f"No quarterly data for {ticker}. "
            "Run `cli/scrape.py` first to populate data/raw/quarterly/."
        )

    if "Close" not in df.columns:
        raise ValueError(f"Quarterly data for {ticker} missing 'Close' column")

    series = df["Close"].dropna()
    series.name = ticker
    return series


def run_ticker(
    ticker: str,
    settings=None,
    horizons: int | None = None,
    holdout: int | None = None,
) -> list[ForecastResult]:
    """Load data and run all 12 methods for a single *ticker*.

    Args:
        ticker: Stock symbol (e.g. "AAPL").
        settings: Optional settings override.
        horizons: Quarters ahead to forecast (defaults to settings.forecast_horizons).
        holdout: Holdout size (defaults to settings.holdout_periods).

    Returns:
        List of 12 :class:`ForecastResult` objects.
    """
    if settings is None:
        settings = get_settings()
    if horizons is None:
        horizons = settings.forecast_horizons
    if holdout is None:
        holdout = settings.holdout_periods

    series = _load_quarterly_series(ticker, settings)
    logger.info(
        f"[{ticker}] Running all 12 methods  "
        f"({len(series)} quarters available, horizon={horizons}q, holdout={holdout})"
    )

    results = run_all_methods(series, horizons=horizons, holdout=holdout)

    # Tag each result with the ticker
    for r in results:
        r.ticker = ticker

    return results


def run_tickers(
    tickers: list[str],
    settings=None,
    horizons: int | None = None,
    holdout: int | None = None,
) -> dict[str, list[ForecastResult]]:
    """Run all 12 methods for each ticker in *tickers*.

    Args:
        tickers: List of ticker symbols.
        settings: Optional settings override.
        horizons: Quarters ahead.
        holdout: Holdout size.

    Returns:
        Dict mapping ticker -> list of 12 ForecastResult.
    """
    if settings is None:
        settings = get_settings()

    output: dict[str, list[ForecastResult]] = {}
    for ticker in tickers:
        try:
            output[ticker] = run_ticker(ticker, settings, horizons, holdout)
        except Exception as exc:
            logger.error(f"[{ticker}] run_ticker failed: {exc}")

    return output


# ------------------------------------------------------------------ #
# Output helpers
# ------------------------------------------------------------------ #


def comparison_table(results: list[ForecastResult]) -> pd.DataFrame:
    """Build a human-readable comparison DataFrame from method results.

    Args:
        results: Output from :func:`run_all_methods` or :func:`run_ticker`.

    Returns:
        DataFrame with columns: method_number, method_name, rmse, mae, mape,
        sorted ascending by RMSE.  Rows with errors are shown at the bottom.
    """
    rows = []
    for r in results:
        rows.append({
            "#": r.method_number,
            "Method": r.method_name,
            "RMSE": round(r.rmse, 4) if r.rmse != float("inf") else "—",
            "MAE": round(r.mae, 4) if r.mae != float("inf") else "—",
            "MAPE": f"{r.mape:.1%}" if r.mape != float("inf") else "—",
            "Status": "OK" if r.error is None else f"ERR: {r.error[:60]}",
        })

    df = pd.DataFrame(rows)

    # Sort: successes first by RMSE, failures last
    ok_mask = df["Status"] == "OK"
    ok_df = df[ok_mask].copy()
    err_df = df[~ok_mask].copy()

    # Convert RMSE to numeric for sorting
    ok_df["_rmse_num"] = pd.to_numeric(ok_df["RMSE"], errors="coerce")
    ok_df = ok_df.sort_values("_rmse_num").drop(columns="_rmse_num")

    return pd.concat([ok_df, err_df], ignore_index=True)


def best_result(results: list[ForecastResult]) -> ForecastResult | None:
    """Return the result with the lowest finite RMSE."""
    valid = [r for r in results if r.error is None and r.rmse != float("inf")]
    if not valid:
        return None
    return min(valid, key=lambda r: r.rmse)


def save_forecasts(
    results: list[ForecastResult],
    ticker: str,
    output_dir: Path | None = None,
) -> Path:
    """Persist forecast results to a Parquet file.

    Each method's horizon-period forecasts are stored as rows.

    Args:
        results: List of ForecastResult for *ticker*.
        ticker: Ticker symbol used in the file name.
        output_dir: Output directory (defaults to ``settings.forecasts_dir``).

    Returns:
        Path to the written Parquet file.
    """
    settings = get_settings()
    if output_dir is None:
        output_dir = settings.forecasts_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        if r.error is not None or r.forecasts.empty:
            continue
        for date, value in r.forecasts.items():
            rows.append({
                "Ticker": ticker,
                "Method_Number": r.method_number,
                "Method_Name": r.method_name,
                "Forecast_Date": date,
                "Forecast_Price": value,
                "RMSE": r.rmse,
                "MAE": r.mae,
                "MAPE": r.mape,
            })

    df = pd.DataFrame(rows)
    out_path = output_dir / f"{ticker}_forecasts.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"[{ticker}] Saved {len(df)} forecast rows to {out_path}")
    return out_path
