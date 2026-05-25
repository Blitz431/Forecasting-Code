"""User-input sanitizers for Streamlit pages.

Validates and normalises free-text inputs before they reach any downstream
code (tickers used in file paths, cache keys, API calls, etc.). All helpers
fail closed: invalid items are dropped (and reported back via the
``rejected`` channel where applicable) rather than raising.
"""

from __future__ import annotations

import re
from typing import Iterable

# Ticker symbols we accept: 1-10 chars, start with a letter, may include
# digits, dot, or hyphen (covers tickers like BRK.B, BF-B, RDS.A).
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPLIT_RE = re.compile(r"[\s,;]+")

MAX_TICKER_LIST = 50
MAX_TEXT_CHARS = 500


def clean_ticker(value: str | None) -> str | None:
    """Return a normalised ticker or ``None`` if invalid."""
    if value is None:
        return None
    cleaned = _CONTROL_RE.sub("", str(value)).strip().upper()
    if not cleaned:
        return None
    return cleaned if _TICKER_RE.match(cleaned) else None


def clean_ticker_list(
    raw: str | Iterable[str] | None,
    max_len: int = MAX_TICKER_LIST,
) -> tuple[list[str], list[str]]:
    """Parse a user-typed ticker list.

    Accepts comma/space/newline-delimited input. Returns
    ``(accepted, rejected)`` — rejected contains the original tokens that
    failed validation so the UI can surface them as warnings.
    """
    if raw is None:
        return [], []
    if isinstance(raw, str):
        tokens = [t for t in _SPLIT_RE.split(raw) if t]
    else:
        tokens = [str(t) for t in raw if t is not None]

    accepted: list[str] = []
    rejected: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        if len(accepted) >= max_len:
            rejected.append(token)
            continue
        normalised = clean_ticker(token)
        if normalised is None:
            rejected.append(token)
        elif normalised in seen:
            continue
        else:
            accepted.append(normalised)
            seen.add(normalised)

    return accepted, rejected


def clean_text(value: str | None, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Strip control chars and enforce a length cap."""
    if value is None:
        return ""
    s = _CONTROL_RE.sub("", str(value)).strip()
    return s[:max_chars]
