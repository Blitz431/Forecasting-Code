from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

"""
Purpose: Synthesis CLI — aggregate all signal parquets into a composite score and print the top-N ranked picks.

Connections:
  - src/ranking/ranker.py: top_picks(), rank_tickers(), to_dataframe()
  - src/utils/tickers.py: get_tickers() for default S&P 500 list
  - config/settings.py: data_dir, top_n_picks

In:  all signal parquets in data/ (indicators, forecasts, ML, news, political, options)
Out: ranked table printed to stdout; optional CSV to --output path
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank S&P 500 stocks by composite multi-signal score.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Number of top picks to show (default: 20).",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        metavar="A,B,C",
        help="Comma-separated ticker list. Defaults to full S&P 500.",
    )
    parser.add_argument(
        "--ml",
        action="store_true",
        default=False,
        help="Include ML predictions (calls predict_latest() — slow).",
    )
    parser.add_argument(
        "--no-indicators",
        action="store_true",
        default=False,
        help="Skip running the indicator pipeline (use only stored signals).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="CSV file path to save the ranking table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from config.settings import get_settings
    from src.ranking.ranker import top_picks, rank_tickers, to_dataframe
    from src.utils.tickers import get_tickers

    settings = get_settings()

    # Resolve ticker list
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        print(f"\nRanking {len(tickers)} custom tickers …")
    else:
        tickers = None  # top_picks() will fetch S&P 500

    # Run ranking
    picks = top_picks(
        n=args.top,
        settings=settings,
        include_ml=args.ml,
        include_indicators=not args.no_indicators,
        tickers=tickers,
    )

    if not picks:
        print(
            "\n[!] No picks returned — make sure you have scraped data first:\n"
            "    python cli/scrape.py --backfill 2015\n"
        )
        sys.exit(1)

    df = to_dataframe(picks)

    # Pretty-print
    print(f"\n{'─'*70}")
    print(f"  TOP {args.top} PICKS  (signals: {'ML+' if args.ml else ''}indicators+forecasts+political+options)")
    print(f"{'─'*70}")
    try:
        print(df.to_string(index=False, max_colwidth=14))
    except Exception:
        print(df.to_string(index=False))
    print(f"{'─'*70}\n")

    # Optional CSV export
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"[✓] Saved ranking to {out}")

    print("Done.")


if __name__ == "__main__":
    main()
