from __future__ import annotations

import json
from pathlib import Path

"""
Purpose: Rolling-average duration tracking for ML training/prediction runs — powers the
         upfront ETA shown before "Run All Analysis" / "Train ML Models" / "Predict Only".

Connections:
  - dashboard/components/ticker_selector.py: calls estimate_duration() before a run starts
    (to display an ETA) and record_duration() after it finishes
  - config/settings.py: settings.data_dir (history lives at data/ml/run_durations.json)

In:  an operation key (e.g. "run_all", "train_ml", "predict_only") + elapsed seconds
Out: data/ml/run_durations.json — rolling history capped to the last _MAX_HISTORY runs per key
"""

_MAX_HISTORY = 20


def _durations_path(settings) -> Path:
    return settings.data_dir / "ml" / "run_durations.json"


def _load(settings) -> dict:
    fp = _durations_path(settings)
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text())
    except Exception:
        return {}


def _save(settings, data: dict) -> None:
    fp = _durations_path(settings)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(data, indent=2))


def record_duration(op_key: str, seconds: float, settings=None) -> None:
    """Append a completed run's duration to the rolling history for *op_key*."""
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()
    data = _load(settings)
    history = data.get(op_key, [])
    history.append(round(seconds, 1))
    data[op_key] = history[-_MAX_HISTORY:]
    _save(settings, data)


def estimate_duration(op_key: str, settings=None) -> float | None:
    """Return the average of past durations for *op_key*, or None with no history yet."""
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()
    history = _load(settings).get(op_key, [])
    if not history:
        return None
    return sum(history) / len(history)


def format_duration(seconds: float) -> str:
    """Format seconds as e.g. '1m 24s' or '48s'."""
    total = round(seconds)
    mins, secs = divmod(total, 60)
    return f"{mins}m {secs}s" if mins else f"{secs}s"
