from __future__ import annotations

import re
from typing import Optional

from src.analytics.company_meta import CompanyMeta
from src.utils.input_sanitize import clean_ticker
from src.utils.tickers import get_tickers

"""
Purpose: Resolve a free-text company name (and/or an LLM-suggested ticker) to a real ticker symbol,
         so agent-discovered suppliers land on the right graph node.

Connections:
  - src/analytics/company_meta.py: reverse name→ticker map from data/company_meta.json
  - src/utils/tickers.py: the S&P 500 universe (validates known tickers)
  - src/utils/input_sanitize.py: clean_ticker for format validation
  - src/research/supply_chain_agent.py: the only caller

In:  a company name string and an optional raw ticker from the LLM
Out: an upper-case ticker, or None when the company can't be resolved (kept as an external node)
"""

# Common short names / aliases the LLM tends to emit, mapped to their ticker.
_ALIASES = {
    "google": "GOOGL", "alphabet": "GOOGL",
    "facebook": "META", "meta": "META",
    "tsmc": "TSM", "taiwan semiconductor": "TSM",
    "amazon web services": "AMZN", "aws": "AMZN",
    "berkshire": "BRK-B", "berkshire hathaway": "BRK-B",
}

# Corporate suffixes stripped before matching.
_SUFFIX_RE = re.compile(
    r"\b(incorporated|inc|corporation|corp|company|co|companies|"
    r"limited|ltd|llc|plc|holdings|holding|group|the|sa|nv|ag)\b",
    re.IGNORECASE,
)
_NONWORD_RE = re.compile(r"[^a-z0-9 ]+")


def normalize_name(name: str) -> str:
    """Lowercase, drop punctuation and corporate suffixes, collapse whitespace."""
    s = (name or "").lower()
    s = _NONWORD_RE.sub(" ", s)
    s = _SUFFIX_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


class EntityResolver:
    """Name/ticker → canonical ticker, backed by the S&P universe + company_meta cache."""

    def __init__(self, settings):
        self._settings = settings
        self._universe: set[str] | None = None
        self._name_to_ticker: dict[str, str] | None = None

    def _load(self) -> None:
        if self._universe is not None:
            return
        try:
            self._universe = {t.upper() for t in get_tickers()}
        except Exception:
            self._universe = set()
        name_map: dict[str, str] = {}
        try:
            for ticker, info in CompanyMeta(self._settings).load().items():
                nm = normalize_name(info.get("name", "")) if isinstance(info, dict) else ""
                if nm:
                    name_map.setdefault(nm, ticker.upper())
        except Exception:
            pass
        self._name_to_ticker = name_map

    def resolve(self, name: str, raw_ticker: Optional[str] = None) -> Optional[str]:
        """Return a ticker for the company, or None if unresolved."""
        self._load()
        assert self._name_to_ticker is not None and self._universe is not None

        # 1. A well-formed ticker from the LLM wins (covers foreign ADRs like TSM that
        #    aren't in the S&P universe — they become external nodes downstream).
        t = clean_ticker(raw_ticker) if raw_ticker else None
        if t:
            return t

        norm = normalize_name(name)
        if not norm:
            return None

        # 2. Exact normalized-name match from the metadata cache.
        if norm in self._name_to_ticker:
            return self._name_to_ticker[norm]

        # 3. Curated alias.
        if norm in _ALIASES:
            return _ALIASES[norm]

        # 4. Substring match against known company names (both directions).
        for known_norm, ticker in self._name_to_ticker.items():
            if known_norm and (known_norm in norm or norm in known_norm):
                return ticker

        return None
