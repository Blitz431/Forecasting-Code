"""Shared fixtures for the pipeline_runner tests.

Every fixture builds its state under tmp_path. Nothing here touches the repo's data/ directory,
the network, or .env.
"""

from __future__ import annotations

import os
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.pipeline_runner import run_state
from src.pipeline_runner.run_state import RunPaths, get_run_paths, local_today, try_acquire_lock, write_status

NOW = datetime(2026, 10, 11, 15, 0, 0, tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


@pytest.fixture
def paths(tmp_path: Path) -> RunPaths:
    return get_run_paths(tmp_path / "data")


@pytest.fixture
def settings(tmp_path: Path) -> types.SimpleNamespace:
    """Settings stand-in with one scraped parquet file (backdated so it never looks 'new')."""
    data_dir = tmp_path / "data"
    raw = data_dir / "raw" / "daily"
    raw.mkdir(parents=True, exist_ok=True)
    f = raw / "AAA.parquet"
    f.write_bytes(b"x")
    old = (datetime.now(timezone.utc) - timedelta(hours=5)).timestamp()
    os.utime(f, (old, old))
    return types.SimpleNamespace(
        data_dir=data_dir, raw_daily_dir=raw, ticker_source="sp500", auto_rank_enabled=True)


@pytest.fixture
def make_status():
    """Factory for a valid status dict; keyword overrides replace fields."""
    def _make(**kw) -> dict:
        base = {
            "schema_version": 1, "run_id": "run1", "state": "running", "mode": "full",
            "trigger": "auto", "phase1_reused": False, "pool": 100, "top": 50,
            "pid": None, "child_pid": None, "step_index": 0, "step_total": 9,
            "label": "Phase 1: Fast-ranking all tickers with indicators...",
            "run_date": local_today(), "started_at": iso(NOW), "step_started_at": iso(NOW),
            "heartbeat_at": iso(NOW), "finished_at": None, "returncode": None, "error": None,
            "log_path": "data/logs/pipeline_job_run1.log",
        }
        base.update(kw)
        return base
    return _make


@pytest.fixture
def make_lock(paths):
    """Factory: acquire the lock for run_id at a chosen time."""
    def _make(run_id="run1", mode="full", trigger="auto", now=NOW, run_date=None) -> None:
        assert try_acquire_lock(paths, run_id, run_date or local_today(), mode, trigger, now)
    return _make


@pytest.fixture
def only_self_alive(monkeypatch):
    """Make pid_alive deterministic: only this test process counts as alive."""
    monkeypatch.setattr(run_state, "pid_alive", lambda pid: pid == os.getpid())


@pytest.fixture
def write_st(paths):
    def _w(status: dict) -> None:
        write_status(paths, status)
    return _w
