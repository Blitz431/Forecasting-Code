from __future__ import annotations

import datetime
import json
from pathlib import Path

from src.analytics.sector_analysis import SectorAnalyzer
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Company metadata cache — long name, sector, and market cap per ticker, fetched on demand from yfinance.

Connections:
  - src/analytics/sector_analysis.py: SectorAnalyzer.get_sector() supplies the sector (do not fork the sector map)
  - config/settings.py: data_dir for the cache file location
  - src/analytics/supply_chain.py: reads meta to size and colour graph nodes
  - dashboard/pages/21_supply_chain.py: calls ensure()/refresh() behind a button

In:  list of ticker symbols
Out: dict of {ticker: {name, sector, market_cap, fetched}}; cache written to data/company_meta.json
"""

_DEFAULT_META_FILE = "company_meta.json"
_STALE_AFTER_DAYS = 30


class CompanyMeta:
    """Persistent per-ticker metadata (name, sector, market cap).

    Market cap is not produced by any scraper in this project, so it is fetched
    lazily from yfinance and cached on disk. Only tickers actually asked for are
    fetched — never the full 500-ticker universe.
    """

    def __init__(self, settings=None, meta_path: Path | None = None):
        self._settings = settings
        if meta_path is not None:
            self._path = meta_path
        elif settings is not None:
            self._path = settings.data_dir / _DEFAULT_META_FILE
        else:
            self._path = Path(_DEFAULT_META_FILE)
        self._sectors = SectorAnalyzer(settings)
        self._cache: dict[str, dict] | None = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> dict[str, dict]:
        """Return the whole cache, reading from disk on first access."""
        if self._cache is not None:
            return self._cache
        if not self._path.exists():
            self._cache = {}
            return self._cache
        try:
            raw = json.loads(self._path.read_text())
            self._cache = raw if isinstance(raw, dict) else {}
        except Exception:
            logger.warning("Could not parse %s — starting from an empty cache", self._path)
            self._cache = {}
        return self._cache

    def save(self) -> None:
        """Persist the cache."""
        if self._cache is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._cache, indent=2, sort_keys=True))

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, ticker: str) -> dict:
        """Return metadata for one ticker. Never raises — falls back to defaults."""
        t = ticker.upper().strip()
        cached = self.load().get(t)
        if cached:
            return cached
        return {
            "name": t,
            "sector": self._sectors.get_sector(t),
            "market_cap": None,
            "fetched": None,
        }

    def _is_stale(self, entry: dict) -> bool:
        fetched = entry.get("fetched")
        if not fetched:
            return True
        try:
            age = datetime.date.today() - datetime.date.fromisoformat(fetched)
        except Exception:
            return True
        return age.days > _STALE_AFTER_DAYS

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def ensure(self, tickers: list[str] | set[str]) -> dict[str, dict]:
        """Fetch metadata for any ticker that is missing or stale. Returns the cache."""
        return self._fetch_many(tickers, force=False)

    def refresh(self, tickers: list[str] | set[str]) -> dict[str, dict]:
        """Force a refetch for every ticker given. Returns the cache."""
        return self._fetch_many(tickers, force=True)

    def _fetch_many(self, tickers: list[str] | set[str], force: bool) -> dict[str, dict]:
        cache = self.load()
        wanted = sorted({str(t).upper().strip() for t in tickers if t})
        todo = [
            t for t in wanted
            if force or t not in cache or self._is_stale(cache[t])
        ]
        if not todo:
            return cache

        logger.info("Fetching company metadata for %d ticker(s)", len(todo))
        for t in todo:
            cache[t] = self._fetch_one(t)
        self._cache = cache
        self.save()
        return cache

    def _fetch_one(self, ticker: str) -> dict:
        """Fetch one ticker from yfinance. Exception-safe — returns defaults on failure."""
        name, market_cap, sector = ticker, None, None
        try:
            import yfinance as yf

            info = yf.Ticker(ticker).info or {}
            name = info.get("longName") or info.get("shortName") or ticker
            market_cap = info.get("marketCap")
            sector = info.get("sector")
        except Exception as exc:
            logger.warning("yfinance lookup failed for %s: %s", ticker, exc)

        return {
            "name": name,
            # Fall back to the shared sector map rather than leaving it Unknown.
            "sector": sector or self._sectors.get_sector(ticker),
            "market_cap": market_cap,
            "fetched": datetime.date.today().isoformat(),
        }
