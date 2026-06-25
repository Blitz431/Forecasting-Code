"""Two-phase ranking pipeline.

Phase 1 — fast: rank all S&P 500 stocks using indicators, forecasts,
           sentiment, options flow, and insider signals (no ML training).
Phase 2 — slow: train full ML models on the top-N candidates from phase 1,
           then re-rank with ML signal included.

Final output: top 50 stocks in order with composite score, current price,
ML-predicted price target, forecast price target, and target date.

Usage
-----
    python cli/pipeline.py               # default: phase-1 pool=150, final top 50
    python cli/pipeline.py --pool 100    # smaller phase-1 pool
    python cli/pipeline.py --top 30      # show top 30 instead of 50
    python cli/pipeline.py --no-deep     # skip LSTM/GRU/Transformer (faster ML)
    python cli/pipeline.py --output data/rankings/pipeline.csv
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.ranking.ranker import rank_tickers, save_ranking_cache
from src.scraper.storage import load_dataframe
from src.utils.logging import setup_logger
from src.utils.tickers import get_tickers

logger = setup_logger("cli.pipeline")


# ---------------------------------------------------------------------------#
# Helpers
# ---------------------------------------------------------------------------#

def _current_price(ticker: str, settings) -> float | None:
    path = settings.raw_daily_dir / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = load_dataframe(path)
    if df.empty or "Close" not in df.columns:
        return None
    return float(df["Close"].dropna().iloc[-1])


def _forecast_target(ticker: str, settings) -> tuple[float | None, str | None]:
    """Return (forecast_price, forecast_date_str) from stored forecast parquet."""
    path = settings.forecasts_dir / f"{ticker}_forecasts.parquet"
    if not path.exists():
        return None, None
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        if df.empty or "Forecast_Price" not in df.columns:
            return None, None
        if "Forecast_Date" in df.columns:
            df["Forecast_Date"] = pd.to_datetime(df["Forecast_Date"])
            now = pd.Timestamp.now()
            future = df[df["Forecast_Date"] > now].sort_values("Forecast_Date")
            if future.empty:
                future = df.sort_values("Forecast_Date").tail(1)
        else:
            future = df.tail(1)
        row = future.iloc[0]
        price = float(row["Forecast_Price"])
        date = str(row["Forecast_Date"].date()) if "Forecast_Date" in row.index else None
        return price, date
    except Exception:
        return None, None


def _ml_price_target(ticker: str, settings, target_days: int = 20) -> tuple[float | None, str | None]:
    """Return (predicted_price, target_date) using stored ML model prediction."""
    from src.ml.runner import predict_latest
    current = _current_price(ticker, settings)
    if current is None:
        return None, None
    predicted_return = predict_latest(
        ticker, model_name="XGBoost", settings=settings, target_days=target_days
    )
    if predicted_return is None:
        return None, None
    target_price = round(current * (1 + predicted_return), 2)
    target_date = (
        datetime.date.today() + datetime.timedelta(days=target_days)
    ).isoformat()
    return target_price, target_date


# ---------------------------------------------------------------------------#
# Printer
# ---------------------------------------------------------------------------#

def _print_results(entries, current_prices, ml_targets, forecast_targets) -> None:
    header = (
        f"{'Rank':>4}  {'Ticker':<6}  {'Score':>6}  "
        f"{'Sigs':>4}  {'Price':>8}  "
        f"{'ML Target':>10}  {'ML Date':<12}  "
        f"{'Fcst Target':>11}  {'Fcst Date':<12}  "
        f"{'ML Upside':>9}  {'Fcst Upside':>11}"
    )
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("  AutoStockAnalyzer -- Two-Phase Pipeline Results")
    print(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * len(header))
    print(header)
    print(sep)

    for e in entries:
        t = e.ticker
        cur = current_prices.get(t)
        ml_price, ml_date = ml_targets.get(t, (None, None))
        fc_price, fc_date = forecast_targets.get(t, (None, None))

        cur_str    = f"${cur:>7.2f}"  if cur else "       N/A"
        ml_str     = f"${ml_price:>9.2f}" if ml_price else "        N/A"
        ml_dt_str  = ml_date or "N/A"
        fc_str     = f"${fc_price:>10.2f}" if fc_price else "         N/A"
        fc_dt_str  = fc_date or "N/A"

        ml_upside = (
            f"{(ml_price - cur) / cur * 100:+.1f}%"
            if (ml_price and cur) else "N/A"
        )
        fc_upside = (
            f"{(fc_price - cur) / cur * 100:+.1f}%"
            if (fc_price and cur) else "N/A"
        )

        print(
            f"{e.rank:>4}  {t:<6}  {e.composite_score:>+6.3f}  "
            f"{e.signals_available:>4}  {cur_str}  "
            f"{ml_str}  {ml_dt_str:<12}  "
            f"{fc_str}  {fc_dt_str:<12}  "
            f"{ml_upside:>9}  {fc_upside:>11}"
        )

    print(sep)
    print()


# ---------------------------------------------------------------------------#
# Main pipeline
# ---------------------------------------------------------------------------#

def run_pipeline(
    pool: int = 150,
    top: int = 50,
    no_deep: bool = False,
    phase1_only: bool = False,
    output: str | None = None,
) -> None:
    settings = get_settings()

    # ------------------------------------------------------------------ #
    # Phase 1 — fast rank of full universe
    # ------------------------------------------------------------------ #
    all_tickers = get_tickers(settings.ticker_source)
    available = [
        t for t in all_tickers
        if (settings.raw_daily_dir / f"{t}.parquet").exists()
    ]

    print(f"\n[Phase 1] Fast-ranking {len(available)} tickers (no ML) ...")
    t0 = time.time()
    phase1 = rank_tickers(
        available, settings,
        include_ml=False,
        include_indicators=True,
    )
    pool_tickers = [e.ticker for e in phase1[:pool]]
    print(f"[Phase 1] Done in {time.time() - t0:.0f}s -- pool: top {len(pool_tickers)} candidates")

    if phase1_only:
        final = phase1[:top]
        # Still collect prices and save cache so trading loop can use it
        print("\n[Phase 1 Only] Collecting price targets ...")
        current_prices = {}
        ml_targets: dict = {}
        forecast_targets = {}
        for e in final:
            t = e.ticker
            current_prices[t] = _current_price(t, settings)
            forecast_targets[t] = _forecast_target(t, settings)
        cache_path = save_ranking_cache(
            phase1,
            settings,
            current_prices=current_prices,
            ml_targets=ml_targets,
            forecast_targets=forecast_targets,
        )
        print(f"[Cache] Ranking cache saved -> {cache_path}")
        _print_results(final, current_prices, ml_targets, forecast_targets)
        return

    # ------------------------------------------------------------------ #
    # Phase 2 — train ML on pool, then re-rank with ML signal
    # ------------------------------------------------------------------ #
    print(f"\n[Phase 2] Training ML models on {len(pool_tickers)} candidates ...")
    t1 = time.time()

    from src.ml.runner import run_tickers as ml_run_tickers
    ml_run_tickers(
        pool_tickers, settings,
        target_days=20,
        tune=False,
        feature_select=True,
        deep_learning=not no_deep,
        save=True,
    )
    print(f"[Phase 2] ML training done in {time.time() - t1:.0f}s")

    print(f"\n[Phase 2] Re-ranking top {len(pool_tickers)} with ML signal ...")
    t2 = time.time()
    phase2 = rank_tickers(
        pool_tickers, settings,
        include_ml=True,
        include_indicators=True,
    )
    final = phase2[:top]
    print(f"[Phase 2] Re-rank done in {time.time() - t2:.0f}s")

    # ------------------------------------------------------------------ #
    # Collect price data
    # ------------------------------------------------------------------ #
    print("\n[Output] Collecting price targets …")
    current_prices = {}
    ml_targets = {}
    forecast_targets = {}

    for e in final:
        t = e.ticker
        current_prices[t] = _current_price(t, settings)
        ml_targets[t] = _ml_price_target(t, settings, target_days=20)
        forecast_targets[t] = _forecast_target(t, settings)

    # ------------------------------------------------------------------ #
    # Save ranking cache (parquet)
    # ------------------------------------------------------------------ #
    cache_path = save_ranking_cache(
        phase2,           # full re-ranked pool, not just top-N
        settings,
        current_prices=current_prices,
        ml_targets=ml_targets,
        forecast_targets=forecast_targets,
    )
    print(f"[Cache] Ranking cache saved -> {cache_path}")

    # ------------------------------------------------------------------ #
    # Print
    # ------------------------------------------------------------------ #
    _print_results(final, current_prices, ml_targets, forecast_targets)

    # ------------------------------------------------------------------ #
    # Optional CSV save
    # ------------------------------------------------------------------ #
    if output:
        import pandas as pd
        rows = []
        for e in final:
            t = e.ticker
            cur = current_prices.get(t)
            ml_p, ml_d = ml_targets.get(t, (None, None))
            fc_p, fc_d = forecast_targets.get(t, (None, None))
            rows.append({
                "Rank":           e.rank,
                "Ticker":         t,
                "Score":          round(e.composite_score, 4),
                "Signals":        e.signals_available,
                "Current_Price":  cur,
                "ML_Target":      ml_p,
                "ML_Target_Date": ml_d,
                "ML_Upside_Pct":  round((ml_p - cur) / cur * 100, 2) if (ml_p and cur) else None,
                "Fcst_Target":    fc_p,
                "Fcst_Date":      fc_d,
                "Fcst_Upside_Pct": round((fc_p - cur) / cur * 100, 2) if (fc_p and cur) else None,
            })
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------#
# Entry point
# ---------------------------------------------------------------------------#

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-phase stock ranking pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pool", type=int, default=150,
                        help="Phase-1 pool size fed into ML training (default: 150).")
    parser.add_argument("--top", type=int, default=50,
                        help="Number of final results to print (default: 50).")
    parser.add_argument("--no-deep", action="store_true",
                        help="Skip LSTM/GRU/Transformer -- faster ML training.")
    parser.add_argument("--phase1-only", action="store_true",
                        help="Run Phase 1 fast rank only, skip ML training.")
    parser.add_argument("--output", type=str, default=None, metavar="PATH",
                        help="Optional CSV path to save results.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        pool=args.pool,
        top=args.top,
        no_deep=args.no_deep,
        phase1_only=args.phase1_only,
        output=args.output,
    )
