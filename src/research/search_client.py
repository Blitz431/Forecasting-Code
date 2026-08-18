from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup

from src.research import ResearchError
from src.research.cache import ResearchCache
from src.research.sources import tier_label, trust_tier
from src.utils.input_sanitize import clean_text
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Query a local SearXNG instance for web search results, preferring its JSON API and falling
         back to scraping the HTML results page when JSON output is disabled server-side.

Connections:
  - config/settings.py: settings.searx.{base_url, request_timeout}, research_request_delay
  - src/research/cache.py: identical queries are served from cache (no repeat search)
  - src/research/sources.py: each result is tagged with a trust tier for the agent + review UI
  - src/research/supply_chain_agent.py: consumes SearchResult lists each round

In:  a query string and a result count
Out: list[SearchResult] (title, url, content, trust tier); ResearchError if SearXNG is unreachable
"""

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AutoStockAnalyzer/1.0; research)",
    "Accept": "application/json, text/html",
}


@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    tier: int = 1
    tier_name: str = "Other"


class SearxClient:
    """SearXNG search client (JSON-first, HTML fallback)."""

    def __init__(self, settings, cache: Optional[ResearchCache] = None, use_cache: bool = True):
        self._settings = settings
        self._base = settings.searx.base_url.rstrip("/")
        self._timeout = settings.searx.request_timeout
        self._delay = settings.research_request_delay
        self._cache = cache if cache is not None else (ResearchCache(settings) if use_cache else None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, count: int = 6) -> list[SearchResult]:
        """Return up to `count` results for `query`. Cached; raises ResearchError if unreachable."""
        query = (query or "").strip()
        if not query:
            return []

        def _produce() -> list[dict]:
            raw = self._search_json(query, count)
            if raw is None:
                raw = self._search_html(query, count)
            time.sleep(self._delay)  # politeness between outbound searches
            return raw

        if self._cache is not None:
            rows = self._cache.cached("searx", {"q": query, "n": count}, _produce)
        else:
            rows = _produce()

        results: list[SearchResult] = []
        for r in (rows or [])[:count]:
            url = r.get("url", "")
            if not url:
                continue
            results.append(SearchResult(
                title=clean_text(r.get("title", ""), 300),
                url=url,
                content=clean_text(r.get("content", ""), 500),
                tier=trust_tier(url),
                tier_name=tier_label(url),
            ))
        return results

    def health(self) -> tuple[bool, str]:
        """(reachable, message) — used by the dashboard to show a friendly error."""
        try:
            self.search("test", count=1)
        except ResearchError as exc:
            return False, str(exc)
        return True, f"SearXNG reachable at {self._base}."

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _search_json(self, query: str, count: int) -> Optional[list[dict]]:
        """Try the JSON API. Returns None (not []) if JSON is unavailable, so the caller falls back."""
        params = {"q": query, "format": "json", "safesearch": "1"}
        try:
            resp = requests.get(
                f"{self._base}/search", params=params, headers=_HEADERS, timeout=self._timeout
            )
        except requests.RequestException as exc:
            raise ResearchError(
                f"SearXNG not reachable at {self._base} ({exc}). Start SearXNG or set SEARX_BASE_URL."
            ) from exc
        if resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except ValueError:
            return None  # JSON format not enabled on this instance → HTML fallback
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
            for r in data.get("results", [])
        ]

    def _search_html(self, query: str, count: int) -> list[dict]:
        """Fallback: scrape the default SearXNG results page."""
        params = {"q": query, "safesearch": "1"}
        try:
            resp = requests.get(
                f"{self._base}/search", params=params, headers=_HEADERS, timeout=self._timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise ResearchError(
                f"SearXNG search failed at {self._base} ({exc})."
            ) from exc

        soup = BeautifulSoup(resp.text, "html.parser")
        out: list[dict] = []
        for art in soup.select("article.result, div.result"):
            link = art.select_one("h3 a, a.url_wrapper, h4 a")
            if not link or not link.get("href"):
                continue
            snippet = art.select_one("p.content, p.result-content")
            out.append({
                "title": link.get_text(" ", strip=True),
                "url": link.get("href", ""),
                "content": snippet.get_text(" ", strip=True) if snippet else "",
            })
            if len(out) >= count:
                break
        return out
