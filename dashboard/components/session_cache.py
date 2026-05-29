"""Daily session cache — tracks which tickers were analyzed today.

Stores a tiny JSON at data/run_cache/session.json. Resets automatically
when the date changes (lazy reset on first load after midnight).
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path


def _cache_path() -> Path:
    from config.settings import get_settings
    settings = get_settings()
    return settings.data_dir / "run_cache" / "session.json"


def load_session() -> dict:
    """Load today's ticker entries. Returns empty dict if no session or date changed."""
    fp = _cache_path()
    if not fp.exists():
        return {}
    try:
        data = json.loads(fp.read_text())
    except Exception:
        return {}
    if data.get("date") != str(date.today()):
        try:
            fp.unlink()
        except Exception:
            pass
        return {}
    return data.get("tickers", {})


def record_run(
    ticker: str,
    *,
    forecast: bool = False,
    ml: str | None = None,
    news: bool = False,
    options: bool = False,
    data_through: str | None = None,
) -> None:
    """Upsert today's run record for *ticker*."""
    fp = _cache_path()
    fp.parent.mkdir(parents=True, exist_ok=True)

    tickers = load_session()
    entry = tickers.get(ticker, {})
    entry["run_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if forecast:
        entry["forecast"] = True
    if ml is not None:
        entry["ml"] = ml
    if news:
        entry["news"] = True
    if options:
        entry["options"] = True
    if data_through is not None:
        entry["data_through"] = data_through
    tickers[ticker] = entry

    fp.write_text(json.dumps({"date": str(date.today()), "tickers": tickers}, indent=2))


def get_ticker_status(ticker: str) -> dict | None:
    """Return today's session record for *ticker*, or None if not analyzed yet."""
    return load_session().get(ticker)


def format_freshness(fp: Path) -> str:
    """Return a human-readable mtime string for *fp*, or 'no data'."""
    if not fp.exists():
        return "no data"
    ts = datetime.fromtimestamp(fp.stat().st_mtime)
    return ts.strftime("%Y-%m-%d %H:%M")
