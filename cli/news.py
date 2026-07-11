import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.news.runner import run_news_pipeline, run_sentiment_only, run_short_interest_only
from src.utils.logging import setup_logger
from src.utils.tickers import get_tickers

logger = setup_logger("cli.news")

"""
Purpose: Phase 5 CLI — scrape news headlines, score FinBERT sentiment, and fetch short interest per ticker.

Connections:
  - src/news/runner.py: run_news_pipeline(), run_sentiment_only(), run_short_interest_only()
  - src/utils/tickers.py: get_tickers() for default S&P 500 list
  - config/settings.py: ticker_source, news_articles_dir
  - cli/scheduler.py: invoked as subprocess in job_analysis()

In:  news feeds (RSS/web), Quiver Quant API (short interest)
Out: data/news/articles/*.parquet, data/news/short_interest/*.parquet
"""


def _print_results(results: dict[str, dict]) -> None:
    """Print a summary table of pipeline results to stdout."""
    print(f"\n{'Ticker':<8} {'Sentiment':<12} {'Score':>7} {'Articles':>9} {'Short Ratio':>12} {'High SI':>8}")
    print("-" * 65)
    for ticker, r in sorted(results.items()):
        if r.get("error"):
            print(f"{ticker:<8} ERROR: {r['error']}")
            continue
        sentiment = r.get("sentiment", "n/a")
        score = r.get("avg_score", 0.0)
        articles = r.get("article_count", 0)
        short_ratio = r.get("short_ratio")
        high_si = r.get("high_short_interest", False)
        ratio_str = f"{short_ratio:.1f}" if short_ratio is not None else "n/a"
        high_str = "YES" if high_si else "no"
        print(f"{ticker:<8} {sentiment:<12} {score:>7.4f} {articles:>9} {ratio_str:>12} {high_str:>8}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoStockAnalyzer — Phase 5: News & Sentiment")
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated ticker list (default: S&P 500)",
    )
    parser.add_argument(
        "--premarket",
        action="store_true",
        help="Run full pipeline for all S&P 500 tickers (pre-market mode)",
    )
    parser.add_argument(
        "--sentiment-only",
        action="store_true",
        help="Fetch news and score sentiment; skip short interest",
    )
    parser.add_argument(
        "--short-interest-only",
        action="store_true",
        help="Fetch short interest only; skip news and sentiment",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max articles to fetch per ticker (default: from settings)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Sentiment rolling window in days (default: 7)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between ticker requests (default: 1.0)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print results table to terminal after the run",
    )
    args = parser.parse_args()

    settings = get_settings()

    # Resolve tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = get_tickers(settings.ticker_source)

    logger.info(f"Phase 5 starting — {len(tickers)} tickers")

    # Dispatch to the right runner
    if args.short_interest_only:
        results = run_short_interest_only(tickers)
    elif args.sentiment_only:
        results = run_sentiment_only(
            tickers,
            max_articles=args.max,
            sentiment_window_days=args.days,
            request_delay=args.delay,
        )
    else:
        results = run_news_pipeline(
            tickers,
            max_articles=args.max,
            sentiment_window_days=args.days,
            request_delay=args.delay,
        )

    if args.show:
        _print_results(results)

    # Summary counts
    errors = sum(1 for r in results.values() if r.get("error"))
    success = len(results) - errors
    positive = sum(1 for r in results.values() if r.get("sentiment") == "positive")
    negative = sum(1 for r in results.values() if r.get("sentiment") == "negative")
    high_si = sum(1 for r in results.values() if r.get("high_short_interest"))

    logger.info(
        f"Done — {success} ok / {errors} errors | "
        f"positive={positive} negative={negative} | high_short_interest={high_si}"
    )


if __name__ == "__main__":
    main()
