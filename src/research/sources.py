from __future__ import annotations

from urllib.parse import urlparse

"""
Purpose: Rank evidence URLs by source trust — SEC/regulatory & investor-relations pages outrank major
         financial wires, which outrank everything else. Feeds both the review UI and corroboration scoring.

Connections:
  - src/research/supply_chain_agent.py: weights a finding's confidence by the tier of its evidence
  - dashboard/components/research_panel.py: shows the tier badge next to each source link
  - standalone — no project-internal imports

In:  an http(s) URL
Out: an integer tier (3 best … 1) and a short human label
"""

TIER_PRIMARY = 3   # regulatory filings + company investor-relations pages
TIER_WIRE = 2      # major financial press / official press-release wires
TIER_OTHER = 1     # blogs, forums, aggregators, everything else

_TIER_LABELS = {
    TIER_PRIMARY: "Primary (SEC/IR)",
    TIER_WIRE: "Major press",
    TIER_OTHER: "Other",
}

# Host substrings that denote major financial press / official wire services.
_WIRE_DOMAINS = (
    "reuters.com", "bloomberg.com", "apnews.com", "wsj.com", "ft.com",
    "cnbc.com", "barrons.com", "marketwatch.com", "forbes.com", "economist.com",
    "businesswire.com", "prnewswire.com", "globenewswire.com", "nasdaq.com",
    "finance.yahoo.com", "fool.com", "investing.com",
)


def domain_of(url: str) -> str:
    """Registrable-ish host for a URL: lowercased netloc without a leading www."""
    try:
        host = (urlparse(url).netloc or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def trust_tier(url: str) -> int:
    """Classify a URL into a trust tier (higher = more authoritative)."""
    host = domain_of(url)
    if not host:
        return TIER_OTHER
    path = ""
    try:
        path = (urlparse(url).path or "").lower()
    except ValueError:
        path = ""

    # Primary: SEC / government filings, or a company investor-relations surface.
    if host.endswith(".gov") or "sec.gov" in host:
        return TIER_PRIMARY
    if host.startswith("investor.") or host.startswith("ir.") or "/investor" in path or "/investor-relations" in path:
        return TIER_PRIMARY

    # Major financial press / official wires.
    if any(w in host for w in _WIRE_DOMAINS):
        return TIER_WIRE

    return TIER_OTHER


def tier_label(url: str) -> str:
    """Short human label for a URL's trust tier."""
    return _TIER_LABELS.get(trust_tier(url), _TIER_LABELS[TIER_OTHER])
