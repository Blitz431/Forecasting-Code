"""CLI entry point for Phase 4 — ML Forecasting Engine.

Usage examples
--------------
    # Train all models for AAPL and evaluate on 2020-2026 test set
    python cli/ml.py --ticker AAPL --train

    # Train with Optuna hyperparameter tuning
    python cli/ml.py --ticker AAPL --train --tune

    # Train without deep learning models (faster)
    python cli/ml.py --ticker AAPL --train --no-deep

    # Predict next-day return using the best model
    python cli/ml.py --ticker AAPL --predict

    # Predict 5-day (weekly) return
    python cli/ml.py --ticker AAPL --predict --days 5

    # Train multiple tickers
    python cli/ml.py --tickers AAPL,MSFT,GOOG --train

    # Train all tickers with stored data
    python cli/ml.py --all --train

    # Show SHAP feature importance after training
    python cli/ml.py --ticker AAPL --train --shap
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.ml.runner import best_result, comparison_table, predict_latest, run_ticker, run_tickers
from src.scraper.storage import list_stored_tickers
from src.utils.logging import setup_logger

logger = setup_logger("cli.ml")


# ------------------------------------------------------------------ #
# Display helpers
# ------------------------------------------------------------------ #


def _print_header(ticker: str, target_days: int) -> None:
    print()
    print("=" * 74)
    print("  AutoStockAnalyzer — Phase 4: ML Forecasting Engine")
    print(f"  Ticker  : {ticker}")
    print(f"  Target  : next-{target_days}-day return")
    print("=" * 74)


def _print_results(ticker: str, results, show_shap: bool = False) -> None:
    """Print comparison table and best model."""
    table = comparison_table(results)
    print()
    print(f"  {'Model':<22}  {'RMSE':>10}  {'MAE':>10}  {'MAPE':>8}  Status")
    print("  " + "-" * 68)

    for _, row in table.iterrows():
        status = row["Status"]
        if status == "OK":
            print(
                f"  {row['Model']:<22}  {str(row['RMSE']):>10}  "
                f"{str(row['MAE']):>10}  {str(row['MAPE']):>8}  {status}"
            )
        else:
            print(f"  {row['Model']:<22}  {'—':>10}  {'—':>10}  {'—':>8}  {status[:55]}")

    best = best_result(results)
    if best:
        print()
        print(f"  Best model : {best.model_name}  (RMSE={best.rmse:.5f})")
        print(f"  Test period: {best.test_start} → {best.test_end}")

        if show_shap and best.feature_importance:
            top_n = 15
            print(f"\n  Top-{top_n} features by importance ({best.model_name}):")
            print("  " + "-" * 52)
            sorted_fi = sorted(best.feature_importance.items(), key=lambda x: x[1], reverse=True)
            for rank, (feat, score) in enumerate(sorted_fi[:top_n], 1):
                bar = "#" * int(score / max(v for _, v in sorted_fi[:top_n]) * 20)
                print(f"  {rank:2d}. {feat:<38}  {score:.4f}  [{bar}]")

    print()


def _print_prediction(ticker: str, pred: float | None, target_days: int) -> None:
    """Print a formatted prediction result."""
    print()
    print(f"  {ticker} — Predicted next-{target_days}d return:", end="  ")
    if pred is None:
        print("N/A (model not trained or insufficient data)")
    else:
        arrow = "▲" if pred > 0 else ("▼" if pred < 0 else "—")
        print(f"{pred:+.4%}  {arrow}")
    print()


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoStockAnalyzer — Phase 4: ML Forecasting Engine"
    )

    # Ticker selection
    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument("--ticker", "-t", type=str, metavar="SYMBOL",
                               help="Single ticker symbol")
    ticker_group.add_argument("--tickers", type=str, metavar="A,B,C",
                               help="Comma-separated list of tickers")
    ticker_group.add_argument("--all", action="store_true",
                               help="Run for all tickers with stored daily data")

    # Mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true",
                             help="Train all models and evaluate on test set")
    mode_group.add_argument("--predict", action="store_true",
                             help="Predict next-N-day return for the latest data point")

    # Options
    parser.add_argument("--days", type=int, default=1, metavar="N",
                        help="Forward return horizon in trading days (default 1)")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning (slower, better accuracy)")
    parser.add_argument("--no-deep", action="store_true",
                        help="Skip LSTM/GRU/Transformer (faster training)")
    parser.add_argument("--no-feature-select", action="store_true",
                        help="Disable SHAP + correlation feature selection")
    parser.add_argument("--shap", action="store_true",
                        help="Print SHAP feature importance after training")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save model artifacts to disk")

    args = parser.parse_args()
    settings = get_settings()

    # Resolve ticker list
    if args.ticker:
        tickers = [args.ticker.strip().upper()]
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = list_stored_tickers(settings.raw_daily_dir)
        if not tickers:
            logger.error("No stored daily data found. Run `python cli/scrape.py` first.")
            sys.exit(1)

    target_days = args.days
    deep_learning = not args.no_deep
    feature_select = not args.no_feature_select
    save = not args.no_save

    t_start = time.time()

    # ---- TRAIN mode ----
    if args.train:
        for ticker in tickers:
            _print_header(ticker, target_days)
            try:
                results = run_ticker(
                    ticker,
                    settings=settings,
                    target_days=target_days,
                    tune=args.tune,
                    feature_select=feature_select,
                    deep_learning=deep_learning,
                    save=save,
                )
                _print_results(ticker, results, show_shap=args.shap)
            except FileNotFoundError as exc:
                print(f"  ERROR: {exc}")
                if len(tickers) == 1:
                    sys.exit(1)
            except Exception as exc:
                logger.error(f"[{ticker}] {exc}")
                if len(tickers) == 1:
                    raise

    # ---- PREDICT mode ----
    else:
        for ticker in tickers:
            try:
                pred = predict_latest(
                    ticker,
                    model_name="XGBoost",
                    settings=settings,
                    target_days=target_days,
                )
                _print_prediction(ticker, pred, target_days)
            except Exception as exc:
                logger.error(f"[{ticker}] predict_latest: {exc}")
                if len(tickers) == 1:
                    sys.exit(1)

    elapsed = time.time() - t_start
    if len(tickers) > 1:
        mode = "trained" if args.train else "predicted"
        print(f"Completed {len(tickers)} tickers ({mode}) in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
