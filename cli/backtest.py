from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Purpose: Backtest CLI — run a paper-trading walk-forward simulation over a configurable train/test window.

Connections:
  - src/trading/backtester.py: BacktestConfig, Backtester.run()
  - config/settings.py: data_dir for output paths

In:  all signal parquets in data/ + daily OHLCV prices (via Backtester internally)
Out: equity curve metrics printed to stdout; optionally saves to data/backtest_results/
"""


def parse_year_range(s: str) -> tuple[str, str]:
    """Parse '2015-2020' → ('2015-01-01', '2020-12-31')."""
    parts = s.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected YYYY-YYYY format, got: {s}"
        )
    return f"{parts[0]}-01-01", f"{parts[1]}-12-31"


def _print_metrics(label: str, m) -> None:
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Total Return  : {m.total_return*100:+.2f}%")
    print(f"  CAGR          : {m.cagr*100:+.2f}%")
    print(f"  Annualised Vol: {m.annualised_vol*100:.2f}%")
    print(f"  Sharpe Ratio  : {m.sharpe:.3f}")
    print(f"  Sortino Ratio : {m.sortino:.3f}")
    print(f"  Calmar Ratio  : {m.calmar:.3f}")
    print(f"  Max Drawdown  : {m.max_drawdown*100:.2f}%")
    print(f"  Max DD Days   : {m.max_drawdown_duration_days}")
    print(f"  Beta vs SPY   : {m.beta:.3f}")
    print(f"  Alpha (ann.)  : {m.alpha*100:+.2f}%")
    print(f"  Win Rate      : {m.win_rate*100:.1f}%")
    print(f"  Profit Factor : {m.profit_factor:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoStockAnalyzer — Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["full-sim"], default="full-sim",
                        help="Simulation mode (default: full-sim)")
    parser.add_argument("--train", default="2015-2020",
                        help="Train window YYYY-YYYY (default: 2015-2020)")
    parser.add_argument("--test",  default="2020-2026",
                        help="Test window YYYY-YYYY (default: 2020-2026)")
    parser.add_argument("--capital", type=float, default=100_000.0,
                        help="Initial portfolio value (default: 100000)")
    parser.add_argument("--top-n",   type=int,   default=20,
                        help="Max concurrent positions (default: 20)")
    parser.add_argument("--slippage-bps", type=float, default=5.0,
                        help="One-way slippage in bps (default: 5)")
    parser.add_argument("--spread-bps",   type=float, default=2.0,
                        help="Half-spread per side in bps (default: 2)")
    parser.add_argument("--circuit-pct",  type=float, default=0.10,
                        help="Circuit-breaker daily drop threshold (default: 0.10)")
    parser.add_argument("--trailing-stop", type=float, default=0.05,
                        help="Trailing stop %% below peak (default: 0.05)")
    parser.add_argument("--no-trades", action="store_true",
                        help="Skip printing the full trade log")
    parser.add_argument("--save", action="store_true",
                        help="Save results to data/backtest_results/")
    args = parser.parse_args()

    # Parse windows
    try:
        train_start, train_end = parse_year_range(args.train)
        test_start, test_end   = parse_year_range(args.test)
    except argparse.ArgumentTypeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("AutoStockAnalyzer — Backtester")
    print(f"  Mode          : {args.mode}")
    print(f"  Train window  : {train_start} → {train_end}")
    print(f"  Test window   : {test_start} → {test_end}")
    print(f"  Initial capital: ${args.capital:,.0f}")
    print(f"  Max positions : {args.top_n}")
    print(f"  Slippage      : {args.slippage_bps} bps")
    print(f"  Circuit breaker: {args.circuit_pct*100:.0f}%")

    # Load settings
    from config.settings import get_settings
    settings = get_settings()

    # Build config
    from src.trading.backtester import BacktestConfig, Backtester

    config = BacktestConfig(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        initial_capital=args.capital,
        max_position_pct=1.0 / max(args.top_n, 1),
        top_n=args.top_n,
        slippage_bps=args.slippage_bps,
        spread_bps=args.spread_bps,
        trailing_stop_pct=args.trailing_stop,
        circuit_breaker_pct=args.circuit_pct,
    )

    # Run
    import time
    print("\nRunning simulation …  (this may take several minutes for 6-year windows)")

    last_pct = [-1]
    def progress(day: int, total: int) -> None:
        pct = int(day / max(total, 1) * 100)
        if pct != last_pct[0] and pct % 5 == 0:
            print(f"  {pct:3d}% ({day}/{total} days)", end="\r", flush=True)
            last_pct[0] = pct

    t0 = time.time()
    bt = Backtester(config, settings)
    result = bt.run(progress_callback=progress)
    elapsed = time.time() - t0

    print(f"\nSimulation complete in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    _print_metrics("Strategy Performance", result.metrics)

    if not result.spy_curve.empty:
        _print_metrics("SPY Buy-and-Hold (Benchmark)", result.spy_metrics)

        print(f"\n  Excess Return vs SPY : {result.metrics.excess_return*100:+.2f}%")
        print(f"  Beta                 : {result.metrics.beta:.3f}")
        print(f"  Alpha (annualised)   : {result.metrics.alpha*100:+.2f}%")

    print(f"\n{'='*50}")
    print("  Trade Summary")
    print(f"{'='*50}")
    print(f"  Total trades      : {result.total_trades}")
    print(f"  Winning           : {result.winning_trades}")
    print(f"  Losing            : {result.losing_trades}")
    print(f"  Circuit breaker   : {result.circuit_breaker_hits} hits")

    if not args.no_trades and not result.trade_log_df.empty:
        print(f"\n{'='*50}")
        print("  Last 20 Trades")
        print(f"{'='*50}")
        import pandas as pd
        with pd.option_context("display.max_columns", None, "display.width", 120):
            print(result.trade_log_df.tail(20).to_string(index=False))

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if args.save:
        out_dir = settings.data_dir / "backtest_results"
        out_dir.mkdir(parents=True, exist_ok=True)

        result.equity_curve.to_frame("portfolio_value").to_parquet(
            out_dir / "equity_curve.parquet"
        )
        if not result.spy_curve.empty:
            result.spy_curve.to_frame("spy_value").to_parquet(
                out_dir / "spy_curve.parquet"
            )
        if not result.trade_log_df.empty:
            result.trade_log_df.to_parquet(out_dir / "trade_log.parquet", index=False)
        if not result.drawdown_curve.empty:
            result.drawdown_curve.to_frame("drawdown").to_parquet(
                out_dir / "drawdown_curve.parquet"
            )

        import json
        metrics_dict = result.metrics.to_dict()
        metrics_dict["spy"] = result.spy_metrics.to_dict()
        metrics_dict["total_trades"] = result.total_trades
        metrics_dict["winning_trades"] = result.winning_trades
        metrics_dict["losing_trades"] = result.losing_trades
        metrics_dict["circuit_breaker_hits"] = result.circuit_breaker_hits
        (out_dir / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))

        print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
