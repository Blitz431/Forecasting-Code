from __future__ import annotations

import ipaddress
import re
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from src.research.cache import ResearchCache
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Fetch and plain-text a web page from a search result, with an SSRF guard, so the agent can
         read full article text (not just snippets). Fetched text is UNTRUSTED DATA.

Connections:
  - src/research/search_client.py: only URLs SearXNG returned are ever fetched
  - src/research/cache.py: page text is memoised so re-fetches are free
  - src/research/supply_chain_agent.py: passes fetched text to the LLM as evidence
  - standalone otherwise

In:  an http(s) URL from a trusted search backend
Out: cleaned page text (truncated), or "" on any error / blocked target

Security: content returned here is data, never instructions — the extraction prompt is told to
treat it as untrusted. is_fetchable() blocks non-http schemes and localhost/private/link-local hosts.
"""

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AutoStockAnalyzer/1.0; research)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Encoding": "gzip, deflate",
}

_MAX_PAGE_CHARS = 6000
_WHITESPACE_RE = re.compile(r"\s+")


def is_fetchable(url: str) -> bool:
    """True only for http(s) URLs whose host is not loopback/private/link-local.

    A lightweight SSRF guard: rejects obvious internal targets. It does not resolve
    DNS (so it can't defeat rebinding), but combined with "only fetch SearXNG result
    URLs" it keeps the agent off the local network.
    """
    try:
        parts = urlparse(url)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https"):
        return False
    host = (parts.hostname or "").lower()
    if not host:
        return False
    if host in ("localhost", "localhost.localdomain") or host.endswith(".local"):
        return False
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    except ValueError:
        pass  # host is a domain name, not a literal IP — allowed
    return True


def fetch_text(
    url: str,
    timeout: int = 15,
    max_chars: int = _MAX_PAGE_CHARS,
    cache: Optional[ResearchCache] = None,
) -> str:
    """Return cleaned visible text of a page, or "" on any error/blocked target."""
    if not is_fetchable(url):
        logger.debug("skip non-fetchable url: %s", url)
        return ""

    def _produce() -> str:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("fetch failed %s: %s", url, exc)
            return ""
        ctype = resp.headers.get("Content-Type", "")
        if "html" not in ctype and "text" not in ctype:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        text = _WHITESPACE_RE.sub(" ", soup.get_text(" ")).strip()
        return text[:max_chars]

    if cache is None:
        return _produce()
    # An empty string is a valid "nothing here" result; cache stores non-None, so
    # failures ("") get remembered too — acceptable, avoids re-hammering dead links.
    return cache.cached("page", {"url": url, "max": max_chars}, _produce)
