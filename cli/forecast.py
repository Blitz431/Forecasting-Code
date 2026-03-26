"""CLI entry point for the Long-Term Forecasting Engine (Phase 2).

Usage examples
--------------
# Run all 12 methods for a single ticker and print comparison table:
    python cli/forecast.py --ticker AAPL

# Run for multiple tickers and save results:
    python cli/forecast.py --tickers AAPL,MSFT,GOOG --save

# Run for all stored tickers:
    python cli/forecast.py --all

# Custom horizons / holdout:
    python cli/forecast.py --ticker AAPL --horizons 8 --holdout 4

# Show only AutoBest result:
    python cli/forecast.py --ticker AAPL --best-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root so imports resolve regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.forecasting.runner import (
    best_result,
    comparison_table,
    run_ticker,
    run_tickers,
    save_forecasts,
)
from src.scraper.storage import list_stored_tickers
from src.utils.logging import setup_logger
from src.utils.tickers import get_tickers

logger = setup_logger("cli.forecast")


# ------------------------------------------------------------------ #
# Formatters
# ------------------------------------------------------------------ #


def _print_header(ticker: str, n_quarters: int, horizons: int, holdout: int) -> None:
    print()
    print("=" * 72)
    print(f"  AutoStockAnalyzer — Phase 2: Long-Term Forecasting Engine")
    print(f"  Ticker : {ticker}")
    print(f"  Data   : {n_quarters} quarters available")
    print(f"  Horizon: {horizons} quarters ahead")
    print(f"  Holdout: {holdout} quarters")
    print("=" * 72)


def _print_comparison(results, ticker: str) -> None:
    df = comparison_table(results)
    print(f"\n{'Method Comparison — ' + ticker}")
    print("-" * 72)
    print(df.to_string(index=False))
    print()


def _print_forecasts(results, ticker: str) -> None:
    """Print each successful method's quarter-by-quarter forecast."""
    print(f"\n{'Quarterly Forecasts — ' + ticker}")
    print("-" * 72)

    valid = [r for r in results if r.error is None and not r.forecasts.empty]
    if not valid:
        print("  No valid forecasts produced.")
        return

    # Build a pivoted table: rows = future dates, cols = methods
    import pandas as pd

    frames = {r.method_name: r.forecasts for r in valid}
    df = pd.DataFrame(frames)
    def _qfmt(ts) -> str:
        q = (ts.month - 1) // 3 + 1
        return f"{ts.year}-Q{q}"

    df.index = [_qfmt(ts) for ts in df.index]

    # Format as currency
    formatted = df.map(lambda v: f"${v:,.2f}" if isinstance(v, float) else v)
    print(formatted.to_string())
    print()


def _print_best(results, ticker: str) -> None:
    br = best_result(results)
    if br is None:
        print(f"\n[{ticker}] No successful method — cannot determine best.")
        return

    print(f"\n{'Best Method — ' + ticker}")
    print("-" * 72)
    print(f"  Method  : #{br.method_number} — {br.method_name}")
    print(f"  RMSE    : {br.rmse:.4f}")
    print(f"  MAE     : {br.mae:.4f}")
    print(f"  MAPE    : {br.mape:.1%}")
    print(f"\n  {br.method_name} Forecast:")
    for date, price in br.forecasts.items():
        q = (date.month - 1) // 3 + 1 if hasattr(date, "month") else "?"
        label = f"{date.year}-Q{q}" if hasattr(date, "year") else str(date)
        print(f"    {label}  ->  ${price:,.2f}")
    print()


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoStockAnalyzer — Phase 2: Long-Term Forecasting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Ticker selection
    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument(
        "--ticker", "-t",
        type=str,
        metavar="SYMBOL",
        help="Single ticker (e.g. AAPL)",
    )
    ticker_group.add_argument(
        "--tickers",
        type=str,
        metavar="A,B,C",
        help="Comma-separated list of tickers",
    )
    ticker_group.add_argument(
        "--all",
        action="store_true",
        help="Run for all tickers that have stored quarterly data",
    )

    # Forecast parameters
    parser.add_argument(
        "--horizons",
        type=int,
        default=None,
        help="Quarters ahead to forecast (default: settings.forecast_horizons)",
    )
    parser.add_argument(
        "--holdout",
        type=int,
        default=None,
        help="Holdout periods for error estimation (default: settings.holdout_periods)",
    )

    # Output options
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save forecasts to data/forecasts/<TICKER>_forecasts.parquet",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Print only the AutoBest winning method (suppresses full comparison)",
    )
    parser.add_argument(
        "--no-forecasts",
        action="store_true",
        help="Suppress the per-quarter forecast table (show comparison only)",
    )

    args = parser.parse_args()
    settings = get_settings()
    horizons = args.horizons or settings.forecast_horizons
    holdout = args.holdout or settings.holdout_periods

    # ------------------------------------------------------------------ #
    # Resolve ticker list
    # ------------------------------------------------------------------ #
    if args.ticker:
        tickers = [args.ticker.strip().upper()]
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:  # --all
        tickers = list_stored_tickers(settings.raw_quarterly_dir)
        if not tickers:
            logger.error(
                "No quarterly data found. Run `python cli/scrape.py` first."
            )
            sys.exit(1)
        logger.info(f"Running forecasts for all {len(tickers)} stored tickers")

    # ------------------------------------------------------------------ #
    # Run forecasts
    # ------------------------------------------------------------------ #
    t_start = time.time()

    for ticker in tickers:
        try:
            results = run_ticker(ticker, settings, horizons=horizons, holdout=holdout)

            # Determine series length for header
            from src.scraper.storage import get_ticker_filepath, load_dataframe
            qpath = get_ticker_filepath(ticker, settings.raw_quarterly_dir)
            qdf = load_dataframe(qpath)
            n_quarters = len(qdf) if not qdf.empty else 0

            _print_header(ticker, n_quarters, horizons, holdout)

            if args.best_only:
                _print_best(results, ticker)
            else:
                _print_comparison(results, ticker)
                if not args.no_forecasts:
                    _print_forecasts(results, ticker)
                _print_best(results, ticker)

            if args.save:
                out_path = save_forecasts(results, ticker)
                print(f"  Saved → {out_path}")

        except FileNotFoundError as exc:
            logger.error(str(exc))
            if len(tickers) == 1:
                sys.exit(1)
        except Exception as exc:
            logger.error(f"[{ticker}] Unexpected error: {exc}")
            if len(tickers) == 1:
                raise

    elapsed = time.time() - t_start
    if len(tickers) > 1:
        print(f"\nCompleted {len(tickers)} tickers in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
