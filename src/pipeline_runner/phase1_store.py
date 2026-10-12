"""
Purpose: Save, load and validate the dated Phase 1 ranking artifact so a later run can skip Phase 1.

Connections:
  - cli/pipeline.py: calls save_phase1_result after Phase 1 and check_phase1_reusable for --skip-phase1
  - launcher (src/pipeline_runner): calls check_phase1_reusable before choosing a run mode
  - config/settings.py: settings supply data_dir, raw_daily_dir and ticker_source

In:  RankEntry-like objects (.ticker, .rank, .composite_score) in rank_tickers order, settings
Out: data/run_cache/phase1_result.json; a Phase1Reuse verdict (never raises)
"""

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.pipeline_runner.run_state import write_json_atomic

PHASE1_SCHEMA_VERSION = 1
PHASE1_FILENAME = "phase1_result.json"      # stored in <data_dir>/run_cache/


def phase1_result_path(data_dir: Path) -> Path:
    """Return the artifact path under <data_dir>/run_cache/."""
    return Path(data_dir) / "run_cache" / PHASE1_FILENAME


def _score_or_none(value: Any) -> float | None:
    """Round a finite score to 6 decimals; -inf, nan, None become None."""
    if value is None:
        return None
    score = float(value)
    if not math.isfinite(score):
        return None
    return round(score, 6)


def save_phase1_result(entries: Sequence[Any], settings: Any, universe_size: int,
                       now: datetime | None = None, day: str | None = None) -> Path:
    """Write the full ordered Phase 1 ranking (including failures) atomically; return the path."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    run_date = day or datetime.now().date().isoformat()

    path = phase1_result_path(settings.data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": PHASE1_SCHEMA_VERSION,
        "created_at": now.astimezone(timezone.utc).isoformat(),
        "run_date": run_date,
        "ticker_source": settings.ticker_source,
        "universe_size": int(universe_size),
        "include_indicators": True,
        "entries": [
            {
                "ticker": str(e.ticker),
                "rank": int(e.rank),
                "score": _score_or_none(e.composite_score),
            }
            for e in entries
        ],
    }
    write_json_atomic(path, payload)
    return path


@dataclass(frozen=True)
class Phase1Reuse:
    usable: bool
    reason: str                     # "ok" or a short human-readable reason (no newlines)
    pool_tickers: list[str] | None  # entries[:pool] tickers in saved order when usable and pool given
    created_at: str | None
    ranked_count: int


def _no(reason: str, created_at: str | None = None, ranked_count: int = 0) -> Phase1Reuse:
    return Phase1Reuse(False, reason, None, created_at, ranked_count)


def check_phase1_reusable(settings: Any, pool: int | None = None,
                          today: str | None = None) -> Phase1Reuse:
    """Decide whether today's saved Phase 1 result can be reused. Never raises."""
    try:
        path = phase1_result_path(settings.data_dir)
        if not path.exists():
            return _no("missing")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return _no("unreadable")
        if not isinstance(data, dict) or data.get("schema_version") != PHASE1_SCHEMA_VERSION:
            return _no("unreadable")

        created_at = data.get("created_at")
        run_date = data.get("run_date")
        source = data.get("ticker_source")
        entries = data.get("entries")
        if (not isinstance(created_at, str) or not isinstance(run_date, str)
                or not isinstance(source, str) or not isinstance(entries, list)):
            return _no("unreadable")
        for e in entries:
            if (not isinstance(e, dict) or not isinstance(e.get("ticker"), str)
                    or not e["ticker"]):
                return _no("unreadable")
        created_dt = datetime.fromisoformat(created_at)
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)

        ranked_count = len(entries)
        day = today or datetime.now().date().isoformat()
        if run_date != day:
            return _no(f"not from today ({run_date})", created_at, ranked_count)
        if source != settings.ticker_source:
            return _no("ticker source changed", created_at, ranked_count)
        if ranked_count == 0:
            return _no("empty", created_at, 0)

        created_ts = created_dt.timestamp()
        newest = 0.0
        raw_dir = Path(settings.raw_daily_dir)
        if raw_dir.is_dir():
            for f in raw_dir.glob("*.parquet"):
                try:
                    newest = max(newest, f.stat().st_mtime)
                except OSError:
                    continue
        if newest > created_ts:
            return _no("price data updated after Phase 1 ran", created_at, ranked_count)

        subset = entries if pool is None else entries[:pool]
        tickers = None if pool is None else [e["ticker"] for e in subset]
        return Phase1Reuse(True, "ok", tickers, created_at, ranked_count)
    except Exception:
        return _no("unreadable")
