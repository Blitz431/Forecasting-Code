from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils.logging import setup_logger
from src.utils.parallel import thread_map

logger = setup_logger(__name__)

"""
Purpose: Live/near-real-time quote fetching via Alpaca's market-data API (separate from
         src/trading/alpaca_client.py, which only wraps the trading/orders API).

Uses the DELAYED_SIP feed (SIP-consolidated across all exchanges, ~15 min delayed) rather
than the default IEX feed — IEX only carries trades routed through that one exchange, so its
"latest trade" can go stale for many minutes even while the stock is actively trading
elsewhere. DELAYED_SIP is available on Alpaca's free data plan; live/real-time SIP requires a
paid subscription. Falls back to IEX if DELAYED_SIP is ever unavailable for an account.

Connections:
  - config/settings.py: AlpacaSettings (api_key, secret_key), live_quotes_dir
  - src/trading/alpaca_client.py: AlpacaClient.list_positions() to find held tickers
  - src/watchlist/watchlist.py: WatchlistManager.load() to find watched tickers
  - cli/scheduler.py: job_live_quotes() calls refresh_quote_cache() every 5 minutes
  - dashboard/pages/2a_stock_summary.py: calls get_live_quote() directly for the selected ticker
  - dashboard/pages/15_watchlist.py: calls load_quote_cache() for the 5-min cached prices

In:  ticker symbols, ALPACA_API_KEY / ALPACA_SECRET_KEY
Out: LiveQuote dataclasses; JSON cache at <live_quotes_dir>/quotes.json
"""


@dataclass
class LiveQuote:
    ticker: str
    price: float
    prev_close: float | None
    change: float | None
    change_pct: float | None
    timestamp: str  # ISO 8601


# Module-level cache of Alpaca data clients keyed by (api_key, secret_key), so
# repeated calls reuse the same client object instead of reconstructing it
# every time. Guarded by _data_client_lock for thread-safety (get_live_quotes
# may be called concurrently, e.g. dashboard + scheduler).
_data_client_cache: dict[tuple[str, str], object] = {}
_data_client_lock = threading.Lock()

# In-process cache of previous closes keyed by ticker -> (file_mtime_ns, value).
# Avoids re-reading a ticker's parquet file when it hasn't changed on disk.
_prev_close_cache: dict[str, tuple[int, float | None]] = {}


def _build_data_client(settings=None):
    """Return a StockHistoricalDataClient, or None if unavailable/misconfigured."""
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    if not settings.alpaca.api_key or not settings.alpaca.secret_key:
        return None

    cache_key = (settings.alpaca.api_key, settings.alpaca.secret_key)
    with _data_client_lock:
        cached = _data_client_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            from alpaca.data.historical import StockHistoricalDataClient
            client = StockHistoricalDataClient(
                api_key=settings.alpaca.api_key,
                secret_key=settings.alpaca.secret_key,
            )
        except ImportError:
            logger.error("alpaca-py not installed. Run: pip install alpaca-py")
            return None
        except Exception as exc:
            logger.error(f"Failed to initialise Alpaca data client: {exc}")
            return None

        _data_client_cache[cache_key] = client
        return client


def _prev_close(ticker: str, settings) -> float | None:
    """Return the most recent stored daily close for *ticker*, or None."""
    fp = settings.raw_daily_dir / f"{ticker}.parquet"
    if not fp.exists():
        return None
    try:
        mtime_ns = fp.stat().st_mtime_ns
    except OSError:
        return None

    cached = _prev_close_cache.get(ticker)
    if cached is not None and cached[0] == mtime_ns:
        return cached[1]

    try:
        df = pd.read_parquet(fp, columns=["Close"])
        closes = df["Close"].dropna() if "Close" in df.columns else pd.Series(dtype=float)
        value = float(closes.iloc[-1]) if len(closes) else None
    except Exception:
        value = None

    _prev_close_cache[ticker] = (mtime_ns, value)
    return value


def _prev_closes(tickers: list[str], settings) -> dict[str, float | None]:
    """Batched, threaded version of _prev_close() for multiple tickers."""
    if not tickers:
        return {}
    results = thread_map(
        lambda t: _prev_close(t, settings),
        tickers,
        max_workers=settings.scrape_io_workers,
        label="prev-close",
    )
    return dict(zip(tickers, results))


def get_live_quotes(tickers: list[str], settings=None) -> dict[str, LiveQuote]:
    """Fetch the latest trade price for each ticker in *tickers* via Alpaca.

    Returns an empty dict if Alpaca isn't configured or the request fails.
    Tickers with no available trade are simply omitted from the result.
    """
    if not tickers:
        return {}

    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    client = _build_data_client(settings)
    if client is None:
        return {}

    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockLatestTradeRequest

    trades = None
    for feed in (DataFeed.DELAYED_SIP, DataFeed.IEX):
        try:
            req = StockLatestTradeRequest(symbol_or_symbols=tickers, feed=feed)
            trades = client.get_stock_latest_trade(req)
            break
        except Exception as exc:
            logger.error(f"get_live_quotes({tickers}) feed={feed.value} failed: {exc}")

    if trades is None:
        return {}

    now_iso = datetime.now(timezone.utc).isoformat()
    prevs = _prev_closes(tickers, settings)
    result: dict[str, LiveQuote] = {}
    for ticker, trade in trades.items():
        try:
            price = float(trade.price)
        except Exception:
            continue
        prev = prevs.get(ticker)
        change = (price - prev) if prev is not None else None
        change_pct = (change / prev * 100) if change is not None and prev else None
        result[ticker] = LiveQuote(
            ticker=ticker,
            price=price,
            prev_close=prev,
            change=change,
            change_pct=change_pct,
            timestamp=now_iso,
        )
    return result


def get_live_quote(ticker: str, settings=None) -> LiveQuote | None:
    """Fetch a single live quote, or None if unavailable."""
    return get_live_quotes([ticker], settings=settings).get(ticker)


def get_tracked_tickers(settings=None) -> list[str]:
    """Return the union of open Alpaca positions and the watchlist, sorted."""
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    tickers: set[str] = set()

    try:
        from src.trading.alpaca_client import AlpacaClient
        client = AlpacaClient(settings)
        tickers.update(p.ticker for p in client.list_positions())
    except Exception as exc:
        logger.error(f"get_tracked_tickers: failed to list Alpaca positions: {exc}")

    try:
        from src.watchlist.watchlist import WatchlistManager
        tickers.update(WatchlistManager(settings).load())
    except Exception as exc:
        logger.error(f"get_tracked_tickers: failed to load watchlist: {exc}")

    return sorted(tickers)


def _cache_path(settings) -> Path:
    return settings.live_quotes_dir / "quotes.json"


def refresh_quote_cache(settings=None) -> dict[str, LiveQuote]:
    """Fetch live quotes for all tracked tickers and persist them to disk."""
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    tickers = get_tracked_tickers(settings)
    quotes = get_live_quotes(tickers, settings=settings)

    fp = _cache_path(settings)
    fp.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "quotes": {t: asdict(q) for t, q in quotes.items()},
    }
    tmp_fp = fp.with_suffix(fp.suffix + ".tmp")
    tmp_fp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp_fp, fp)
    logger.info(f"Refreshed live quote cache: {len(quotes)}/{len(tickers)} tickers -> {fp}")

    return quotes


def load_quote_cache(settings=None) -> tuple[dict[str, dict], str | None]:
    """Read the cached quotes written by refresh_quote_cache().

    Returns (quotes_by_ticker, updated_at_iso), or ({}, None) if no cache exists.
    """
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    fp = _cache_path(settings)
    if not fp.exists():
        return {}, None
    try:
        payload = json.loads(fp.read_text())
        return payload.get("quotes", {}), payload.get("updated_at")
    except Exception:
        return {}, None
