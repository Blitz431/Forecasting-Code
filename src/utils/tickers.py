"""Ticker universe management — fetch S&P 500 component list."""

import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# Fallback list if Wikipedia scrape fails (top 50 by market cap as of 2024)
FALLBACK_SP500_SAMPLE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "XOM",
    "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
    "PEP", "KO", "AVGO", "COST", "TMO", "MCD", "WMT", "CSCO", "ACN", "ABT",
    "DHR", "CRM", "NEE", "LIN", "TXN", "AMD", "PM", "UNP", "RTX", "HON",
    "LOW", "INTC", "AMGN", "UPS", "BA", "CAT", "GS", "BLK", "ISRG", "MDLZ",
]


def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 ticker list from Wikipedia.

    Returns:
        List of ticker symbols.
    """
    try:
        import requests
        from io import StringIO
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 AutoStockAnalyzer/1.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return sorted(tickers)
    except Exception as e:
        logger.warning(f"Failed to fetch S&P 500 list: {e}. Using fallback sample.")
        return FALLBACK_SP500_SAMPLE


def get_tickers(source: str = "sp500") -> list[str]:
    """Get ticker list based on configured source.

    Args:
        source: Either "sp500" or a comma-separated list of tickers.

    Returns:
        List of ticker symbols.
    """
    if source == "sp500":
        return get_sp500_tickers()
    else:
        # Treat as comma-separated custom list
        return [t.strip().upper() for t in source.split(",") if t.strip()]
