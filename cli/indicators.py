from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.indicators.signal_aggregator import AggregateResult, run_and_aggregate
from src.scraper.storage import get_ticker_filepath, list_stored_tickers, load_dataframe
from src.utils.logging import setup_logger

logger = setup_logger("cli.indicators")

"""
Purpose: Phase 3 CLI — run all 9 technical indicators and print composite signal table per ticker.

Connections:
  - src/indicators/signal_aggregator.py: run_and_aggregate() for per-ticker composite
  - src/scraper/storage.py: list_stored_tickers(), load_dataframe()
  - config/settings.py: raw_daily_dir
  - cli/scheduler.py: invoked as subprocess in job_indicators()

In:  data/raw/daily/*.parquet (daily OHLCV)
Out: indicator table printed to stdout; no files written
"""

# Bar chart for the score display
_BAR_FULL = "#"
_BAR_EMPTY = "."
_BAR_WIDTH = 20


def _score_bar(score_normalized: float) -> str:
    """Build a visual progress bar centred at 0."""
    # Convert from [−1, +1] to [0, 1]
    pct = (score_normalized + 1) / 2
    filled = round(pct * _BAR_WIDTH)
    return _BAR_FULL * filled + _BAR_EMPTY * (_BAR_WIDTH - filled)


def _signal_color_prefix(signal_value: int) -> str:
    return {2: "++", 1: "+", 0: "~", -1: "-", -2: "--"}.get(signal_value, "~")


def _print_full(agg: AggregateResult) -> None:
    """Print detailed indicator breakdown + composite."""
    print()
    print("=" * 72)
    print(f"  AutoStockAnalyzer — Phase 3: Short-Term Technical Indicators")
    print(f"  Ticker  : {agg.ticker}")
    print(f"  Signals : {agg.succeeded} computed, {agg.failed} failed")
    print("=" * 72)

    # Per-indicator table
    print(f"\n  {'Indicator':<38}  {'Signal':<12}  Interpretation")
    print("  " + "-" * 68)

    for r in agg.results:
        if r.error:
            print(f"  {r.indicator_name:<38}  {'ERROR':<12}  {r.error[:50]}")
        else:
            prefix = _signal_color_prefix(int(r.signal))
            print(
                f"  {r.indicator_name:<38}  "
                f"[{prefix}] {r.signal.label():<10}  "
                f"{r.interpretation[:55]}"
            )

    # Composite
    bar = _score_bar(agg.score_normalized)
    print()
    print("  " + "-" * 68)
    prefix = _signal_color_prefix(int(agg.signal))
    print(
        f"  {'COMPOSITE SIGNAL':<38}  "
        f"[{prefix}] {agg.signal.label():<10}  "
        f"Score: {agg.score:+.2f}/2.0"
    )
    print(f"\n  [{bar}]  {agg.score_normalized:+.0%}")
    print()


def _print_compact(agg: AggregateResult) -> None:
    """One-line summary per indicator + composite."""
    prefix = _signal_color_prefix(int(agg.signal))
    bar = _score_bar(agg.score_normalized)

    lines = [f"{agg.ticker:6s}  [{bar}] {agg.signal.label():12s} ({agg.score:+.2f})"]
    for r in agg.results:
        if not r.error:
            p = _signal_color_prefix(int(r.signal))
            lines.append(f"         [{p}] {r.indicator_name:<38} {r.signal.label()}")

    print("\n".join(lines))
    print()


def _load_daily(ticker: str, settings) -> "pd.DataFrame":
    import pandas as pd

    filepath = get_ticker_filepath(ticker, settings.raw_daily_dir)
    df = load_dataframe(filepath)

    if df.empty:
        raise FileNotFoundError(
            f"No daily data for {ticker}. "
            "Run `python cli/scrape.py` first."
        )

    if "Close" not in df.columns:
        raise ValueError(f"Daily data for {ticker} missing 'Close' column")

    return df.sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoStockAnalyzer — Phase 3: Short-Term Technical Indicators"
    )

    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument("--ticker", "-t", type=str, metavar="SYMBOL")
    ticker_group.add_argument("--tickers", type=str, metavar="A,B,C")
    ticker_group.add_argument(
        "--all", action="store_true",
        help="Run for all tickers with stored daily data"
    )

    parser.add_argument(
        "--compact", action="store_true",
        help="One-line output per ticker instead of full breakdown"
    )

    args = parser.parse_args()
    settings = get_settings()

    if args.ticker:
        tickers = [args.ticker.strip().upper()]
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = list_stored_tickers(settings.raw_daily_dir)
        if not tickers:
            logger.error("No daily data found. Run `python cli/scrape.py` first.")
            sys.exit(1)

    t_start = time.time()
    for ticker in tickers:
        try:
            df = _load_daily(ticker, settings)
            agg = run_and_aggregate(df, ticker)

            if args.compact:
                _print_compact(agg)
            else:
                _print_full(agg)

        except FileNotFoundError as exc:
            logger.error(str(exc))
            if len(tickers) == 1:
                sys.exit(1)
        except Exception as exc:
            logger.error(f"[{ticker}] {exc}")
            if len(tickers) == 1:
                raise

    elapsed = time.time() - t_start
    if len(tickers) > 1:
        print(f"Completed {len(tickers)} tickers in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
