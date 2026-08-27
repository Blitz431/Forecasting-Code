import threading
import time

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from config.settings import get_settings
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Shared, rate-limit-aware yfinance access layer — bounds effective
concurrency across all callers and centralizes the MultiIndex-extraction
logic previously duplicated in price_scraper.py.

Connections:
  - config/settings.py: scrape_max_workers (outer semaphore size), yf_inner_threads
    (per-call internal thread cap)
  - Used by: src/scraper/price_scraper.py (planned follow-up), and any other
    scraper module that needs to call yf.download() safely

In:  ticker list, start/end dates
Out: pd.DataFrame (batch, MultiIndex-columned) / pd.DataFrame (single ticker, title-cased columns)
"""

_YF_SEMAPHORE: threading.Semaphore | None = None
_SEMAPHORE_LOCK = threading.Lock()


def _get_semaphore() -> threading.Semaphore:
    """Lazily build a module-level semaphore sized to settings.scrape_max_workers.

    Resolved once on first use (thread-safe double-checked init under
    _SEMAPHORE_LOCK) so total in-flight yfinance batches are globally capped
    regardless of how many callers/threads invoke yf_download concurrently.
    """
    global _YF_SEMAPHORE
    if _YF_SEMAPHORE is None:
        with _SEMAPHORE_LOCK:
            if _YF_SEMAPHORE is None:
                settings = get_settings()
                _YF_SEMAPHORE = threading.Semaphore(settings.scrape_max_workers)
    return _YF_SEMAPHORE


def extract_ticker_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Pull a single ticker out of a yfinance MultiIndex DataFrame.

    yfinance 1.x is inconsistent about column level order:
      - Multi-ticker download  → ['Ticker', 'Price']  e.g. ('AAPL', 'Close')
      - Single-ticker download → ['Price', 'Ticker']  e.g. ('Close', 'AAPL')

    We detect which level holds the ticker names at call time so neither
    layout causes a KeyError.
    """
    level_names = data.columns.names  # e.g. ['Ticker', 'Price'] or ['Price', 'Ticker']

    if level_names[0] == "Ticker":
        ticker_level = 0
    else:
        ticker_level = 1

    available = data.columns.get_level_values(ticker_level).unique()
    if ticker not in available:
        return pd.DataFrame()

    df = data.xs(ticker, level=ticker_level, axis=1).dropna(how="all")
    # Normalise column names to title-case (Open/High/Low/Close/Volume)
    df.columns = [str(c).title() for c in df.columns]
    return df


def yf_download(
    tickers: list[str],
    start: str,
    end: str | None = None,
    retries: int = 2,
    actions: bool = False,
) -> pd.DataFrame:
    """Rate-limit-aware wrapper around yf.download.

    Bounds effective concurrency by acquiring a module-level semaphore (sized
    to settings.scrape_max_workers) for the duration of the call, and caps
    yfinance's own internal per-call thread pool via settings.yf_inner_threads
    (passed as an int, not bare threads=True) so effective peak concurrency is
    scrape_max_workers x yf_inner_threads, bounded and environment-independent.
    """
    settings = get_settings()
    semaphore = _get_semaphore()

    data = pd.DataFrame()
    with semaphore:
        for attempt in range(retries + 1):
            try:
                data = yf.download(
                    tickers=tickers,
                    start=start,
                    end=end,
                    group_by="ticker",
                    auto_adjust=True,
                    actions=actions,
                    threads=settings.yf_inner_threads,
                    progress=False,
                    timeout=30,
                )
                break  # success
            except YFRateLimitError as exc:
                if attempt < retries:
                    wait = 30 * (attempt + 1)
                    logger.warning(
                        f"yf_download attempt {attempt+1} rate-limited: {exc}. Retrying in {wait}s …"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"yf_download rate-limited after {retries+1} attempts: {exc}")
                    return pd.DataFrame()
            except Exception as exc:
                if attempt < retries:
                    wait = 5 * (attempt + 1)
                    logger.warning(
                        f"yf_download attempt {attempt+1} failed: {exc}. Retrying in {wait}s …"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"yf_download failed after {retries+1} attempts: {exc}")
                    return pd.DataFrame()

    return data
