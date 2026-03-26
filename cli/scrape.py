"""CLI entry point for data scraping.

Usage:
    python cli/scrape.py                           # Incremental update for S&P 500
    python cli/scrape.py --backfill                # Full backfill from 2015
    python cli/scrape.py --tickers AAPL,MSFT,GOOG  # Specific tickers only
    python cli/scrape.py --prices-only             # Skip macro and dividends
    python cli/scrape.py --macro-only              # Only FRED macro data
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.scraper.dividend_scraper import scrape_dividends
from src.scraper.macro_scraper import scrape_macro
from src.scraper.price_scraper import aggregate_to_quarterly, scrape_prices
from src.utils.logging import setup_logger
from src.utils.tickers import get_tickers

logger = setup_logger("cli.scrape")


def main():
    parser = argparse.ArgumentParser(description="AutoStockAnalyzer Data Scraper")
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated ticker list (default: S&P 500)",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Force full historical download from 2015",
    )
    parser.add_argument(
        "--prices-only",
        action="store_true",
        help="Only scrape stock prices (skip macro and dividends)",
    )
    parser.add_argument(
        "--macro-only",
        action="store_true",
        help="Only scrape FRED macro data",
    )
    parser.add_argument(
        "--dividends-only",
        action="store_true",
        help="Only scrape dividend data",
    )
    parser.add_argument(
        "--no-quarterly",
        action="store_true",
        help="Skip quarterly aggregation after price scrape",
    )

    args = parser.parse_args()
    settings = get_settings()
    start_time = time.time()

    # Determine tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        logger.info(f"Using custom ticker list: {len(tickers)} tickers")
    else:
        tickers = get_tickers(settings.ticker_source)
        logger.info(f"Using {settings.ticker_source} ticker list: {len(tickers)} tickers")

    mode = "backfill" if args.backfill else "incremental"
    logger.info(f"Scrape mode: {mode}")

    # --- Prices ---
    if not args.macro_only and not args.dividends_only:
        logger.info("=" * 60)
        logger.info("SCRAPING STOCK PRICES")
        logger.info("=" * 60)

        price_results = scrape_prices(tickers, backfill=args.backfill)
        logger.info(f"Prices: {len(price_results)} tickers updated")

        # Quarterly aggregation
        if not args.no_quarterly and price_results:
            logger.info("Aggregating to quarterly data...")
            aggregate_to_quarterly(
                daily_dir=settings.raw_daily_dir,
                quarterly_dir=settings.raw_quarterly_dir,
                tickers=list(price_results.keys()),
            )

    # --- Macro ---
    if not args.prices_only and not args.dividends_only:
        logger.info("=" * 60)
        logger.info("SCRAPING MACRO DATA (FRED)")
        logger.info("=" * 60)

        try:
            macro_results = scrape_macro(backfill=args.backfill)
            logger.info(f"Macro: {len(macro_results)} series updated")
        except ValueError as e:
            logger.warning(f"Macro scrape skipped: {e}")

    # --- Dividends ---
    if not args.prices_only and not args.macro_only:
        logger.info("=" * 60)
        logger.info("SCRAPING DIVIDENDS")
        logger.info("=" * 60)

        div_results = scrape_dividends(tickers, backfill=args.backfill)
        logger.info(f"Dividends: {len(div_results)} tickers with dividend data")

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"SCRAPE COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
