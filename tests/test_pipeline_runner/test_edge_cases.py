"""Tester edge cases for the background ranking pipeline (plans/auto_rank_background.md).

Everything lives under tmp_path. No network, no .env, no real cli/pipeline.py, no real data/.
Tests marked xfail(strict=True, reason="BUG: ...") describe real defects found by the tester; the
rest document actual behaviour (including the known residual gaps) or confirm safe behaviour.

Note: importing cli.pipeline_job (and src.utils.logging) opens logs/autostockanalyzer.log in the
repo, exactly as the existing suite already does. Nothing in this file writes to it on purpose.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from cli import pipeline_job as pj
from src.pipeline_runner import launcher, run_state
from src.pipeline_runner.launcher import maybe_start_daily_run, start_pipeline_run
from src.pipeline_runner.phase1_store import check_phase1_reusable, phase1_result_path
from src.pipeline_runner.run_state import (
    break_stale_lock,
    get_run_paths,
    get_run_view,
    has_done_marker,
    is_run_active,
    local_today,
    read_day_state,
    read_lock,
    read_log_tail,
    read_status,
    record_full_failure,
    try_acquire_lock,
    write_status,
)

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261011T150000Z-abc123"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


class FakeSpawn:
    def __init__(self, pid: int | None = None):
        self.calls: list[list[str]] = []
        self.pid = os.getpid() if pid is None else pid

    def __call__(self, argv, cwd):
        self.calls.append(argv)
        return self.pid


def _mode_of(argv: list[str]) -> str:
    return argv[argv.index("--mode") + 1]


@pytest.fixture(autouse=True)
def _isolate_process_state(monkeypatch):
    monkeypatch.setattr(pj.os, "nice", lambda n: 0, raising=False)
    old = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, old)


# ====================================================================== (A) stale lock gaps


def _old_lock(paths, run_id="deadrun", age_min=5, mode="full", trigger="auto"):
    assert try_acquire_lock(paths, run_id, local_today(), mode, trigger, _now() - timedelta(minutes=age_min))


def test_A_stale_lock_without_status_next_load_starts_second_full_run(paths, settings, only_self_alive):
    """No status file at all + lock older than the spawn grace: no failure is recorded, so the next
    page load starts ANOTHER automatic full run."""
    _old_lock(paths)
    spawn = FakeSpawn()
    res = maybe_start_daily_run(paths, settings, spawn=spawn)
    assert res.started and res.mode == "full"
    assert _mode_of(spawn.calls[0]) == "full"
    assert read_day_state(paths).full_failures == 0          # the dead run left no failure behind


@pytest.mark.parametrize("garbage", ["", "{not json", "[1,2]", '{"schema_version": 2, "state": "running"}'])
def test_A_stale_lock_with_corrupt_status_behaves_like_missing_status(paths, settings, only_self_alive, garbage):
    _old_lock(paths)
    paths.status_file.write_text(garbage, encoding="utf-8")
    spawn = FakeSpawn()
    res = maybe_start_daily_run(paths, settings, spawn=spawn)
    assert res.started and _mode_of(spawn.calls[0]) == "full"
    assert read_day_state(paths).full_failures == 0


def test_A_stale_lock_with_other_run_id_status_records_no_failure(paths, settings, only_self_alive, make_status):
    _old_lock(paths, run_id="deadrun")
    write_status(paths, make_status(run_id="someone-else", state="running", pid=os.getpid()))
    spawn = FakeSpawn()
    res = maybe_start_daily_run(paths, settings, spawn=spawn)
    assert res.started and _mode_of(spawn.calls[0]) == "full"
    assert read_day_state(paths).full_failures == 0


def test_A_fresh_corrupt_lock_is_trusted_for_spawn_grace_then_broken(paths, settings, only_self_alive):
    paths.lock_file.write_text("", encoding="utf-8")          # zero byte lock
    spawn = FakeSpawn()
    res = maybe_start_daily_run(paths, settings, spawn=spawn)
    assert not res.started and res.reason == "running"          # fresh: treated as alive (<60 s)
    old = time.time() - 600
    os.utime(paths.lock_file, (old, old))
    res = maybe_start_daily_run(paths, settings, spawn=spawn)
    assert res.started                                          # >60 s old: broken, silently re-run
    assert read_day_state(paths).full_failures == 0


def test_A_contrast_matching_running_status_with_dead_pid_does_count_a_failure(paths, settings, make_status):
    _old_lock(paths, run_id="deadrun")
    write_status(paths, make_status(run_id="deadrun", state="running", pid=2_000_000_000,
                                    heartbeat_at=_iso(_now() - timedelta(seconds=30))))
    spawn = FakeSpawn()
    res = maybe_start_daily_run(paths, settings, spawn=spawn)
    assert res.started and res.reason == "started"
    assert _mode_of(spawn.calls[0]) == "phase1"                 # fallback, NOT a second full run
    assert read_day_state(paths).full_failures == 1


def test_A_wrapper_never_started_for_over_60s_counts_failure_then_old_wrapper_exits(paths, settings, make_status):
    """Slow spawn (>60 s, pid never recorded): the lock is broken and a failure IS recorded."""
    _old_lock(paths, run_id="slow")
    write_status(paths, make_status(run_id="slow", state="running", pid=None,
                                    started_at=_iso(_now() - timedelta(minutes=2))))
    assert break_stale_lock(paths) is True
    assert read_day_state(paths).full_failures == 1
    assert read_status(paths)["state"] == "failed"


def test_A_wrapper_killed_after_final_status_before_release(paths, settings, only_self_alive, make_status):
    """Killed between the terminal status write and release_lock: marker already there, so no re-run."""
    _old_lock(paths, run_id="r1")
    run_state.write_done_markers(paths, [local_today()])
    write_status(paths, make_status(run_id="r1", state="done", pid=2_000_000_000,
                                    finished_at=_iso(_now() - timedelta(minutes=5))))
    res = maybe_start_daily_run(paths, settings, spawn=FakeSpawn())
    assert not res.started and res.reason == "already_done"


# ====================================================================== (B) done marker failure


def _fake_child(tmp: Path, lines: list[str], code: str = "", exit_code: int = 0, name="fake.py") -> Path:
    body = ["import sys, os, time, json, signal, subprocess"]
    body += [f"print({ln!r}, flush=True)" for ln in lines]
    if code:
        body.append(textwrap.dedent(code))
    body.append(f"sys.exit({exit_code})")
    p = tmp / name
    p.write_text("\n".join(body) + "\n", encoding="utf-8")
    return p


FULL_LINES = [
    "[Phase 1] Fast-ranking 500 tickers (no ML) ...", "[Phase 1] Done in 12s -- pool: top 100 candidates",
    "[Phase 2] Training ML models on 100 candidates ...", "[Phase 2] ML training done in 40s",
    "[Phase 2] Re-ranking top 100 with ML signal ...", "[Phase 2] Re-rank done in 3s",
    "[Output] Collecting price targets", "[Cache] Ranking cache saved -> x", "  AutoStockAnalyzer -- Results",
]


def _job_args(paths, script, *, run_id=RUN_ID, mode="full", run_date=None, extra=None, trigger="auto"):
    argv = ["--run-id", run_id, "--run-date", run_date or local_today(), "--mode", mode,
            "--pool", "100", "--top", "50", "--trigger", trigger, "--data-dir", str(paths.data_dir),
            "--pipeline-script", str(script), "--heartbeat-s", "0.5"]
    return pj.parse_args(argv + (extra or []))


def test_B_done_marker_write_failure_then_next_page_load(paths, settings, tmp_path, monkeypatch, only_self_alive):
    """Full run exits 0 but the marker cannot be written (OSError). Status says done, lock released.
    Document what the next page load does."""
    assert try_acquire_lock(paths, RUN_ID, local_today(), "full", "auto")
    script = _fake_child(tmp_path, FULL_LINES)
    orig = Path.write_text

    def flaky(self, *a, **k):
        if self.name.startswith(".done_rank_"):
            raise OSError("disk full")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", flaky)
    assert pj.run_job(_job_args(paths, script)) == 0
    monkeypatch.setattr(Path, "write_text", orig)
    assert read_status(paths)["state"] == "done"
    assert read_lock(paths) is None
    assert not has_done_marker(paths)
    assert read_day_state(paths).full_success          # backup success signal in the day state
    spawn = FakeSpawn()
    res = maybe_start_daily_run(paths, settings, spawn=spawn)
    # Fixed: the day-state full_success flag stops a second automatic full run.
    assert not res.started and res.reason == "already_done" and spawn.calls == []


def test_B_successful_full_run_must_not_be_auto_restarted_when_marker_cannot_be_written(
        paths, settings, tmp_path, monkeypatch, only_self_alive):
    assert try_acquire_lock(paths, RUN_ID, local_today(), "full", "auto")
    script = _fake_child(tmp_path, FULL_LINES)
    orig = Path.write_text

    def flaky(self, *a, **k):
        if self.name.startswith(".done_rank_"):
            raise OSError("disk full")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", flaky)
    assert pj.run_job(_job_args(paths, script)) == 0
    monkeypatch.setattr(Path, "write_text", orig)
    spawn = FakeSpawn()
    res = maybe_start_daily_run(paths, settings, spawn=spawn)
    assert not res.started and spawn.calls == []


# ====================================================================== (C) day-state corruption


@pytest.mark.parametrize("text", [
    "", "   ", "{", '{"schema_version": 1, "full_failures": ', "null", "[]", "42",
    '{"schema_version": 1, "full_failures": -1}', '{"schema_version": 1, "full_failures": "2"}',
    '{"schema_version": 1, "full_failures": 1.0}', '{"schema_version": 1, "full_failures": true}',
    '{"schema_version": 1, "fallback_started": 1}', '{"schema_version": 1, "phase1_success": "yes"}',
    '{"schema_version": 7, "full_failures": 3}', '{"full_failures": 3}', "﻿{}",
    '{"schema_version": 1, "full_failures": NaN}',
])
def test_C_corrupt_day_state_reads_as_defaults_never_raises(paths, text):
    paths.day_state_file(local_today()).write_text(text, encoding="utf-8")
    st = read_day_state(paths)
    assert (st.full_failures, st.fallback_started, st.phase1_success) == (0, False, False)


def test_C_binary_day_state_never_raises(paths):
    paths.day_state_file(local_today()).write_bytes(b"\xff\xfe\x00\x01garbage")
    assert read_day_state(paths).full_failures == 0


def test_C_corrupt_day_state_after_a_failed_full_run_allows_another_full_run(paths, settings):
    record_full_failure(paths, local_today())
    paths.day_state_file(local_today()).write_bytes(b"")        # zero-byte after a crash/power loss
    spawn = FakeSpawn()
    res = maybe_start_daily_run(paths, settings, spawn=spawn)
    assert res.started and _mode_of(spawn.calls[0]) == "full"   # failure count forgotten


def test_C_corrupt_day_state_heals_on_next_write(paths):
    paths.day_state_file(local_today()).write_text("{", encoding="utf-8")
    st = record_full_failure(paths, local_today())
    assert st.full_failures == 1 and read_day_state(paths).full_failures == 1


def test_C_day_state_dir_in_place_of_file_never_raises(paths):
    paths.day_state_file(local_today()).mkdir()
    assert read_day_state(paths).full_failures == 0


# ====================================================================== pid fill race


def _race_start(paths, settings, make_status_dict, monkeypatch, wrapper_effect):
    """Interleave: launcher read_status -> (wrapper writes) -> launcher write_status with stale copy."""
    real_read = launcher.read_status
    state = {"n": 0}

    def racing_read(p):
        s = real_read(p)
        state["n"] += 1
        if state["n"] == 1 and s is not None and s.get("pid") is None:
            wrapper_effect(p, s)          # wrapper gets in after the launcher's read
        return s                          # launcher keeps its stale snapshot

    monkeypatch.setattr(launcher, "read_status", racing_read)
    return start_pipeline_run("full", 100, 50, trigger="auto", paths=paths, settings=settings,
                              spawn=FakeSpawn())


def test_pidfill_race_stale_snapshot_does_not_overwrite_fresher_wrapper_progress(paths, settings, monkeypatch, only_self_alive):
    def wrapper(p, s):
        write_status(p, {**s, "pid": os.getpid(), "step_index": 4, "label": "Phase 2: Re-ranking",
                         "heartbeat_at": _iso(_now())})

    res = _race_start(paths, settings, None, monkeypatch, wrapper)
    assert res.started
    final = read_status(paths)
    # Fixed: the launcher re-reads before filling and leaves the fresher wrapper record alone.
    assert final["step_index"] == 4 and final["pid"] == os.getpid()
    view = get_run_view(paths)
    assert view.state == "running" and view.step_index == 4


def test_pidfill_race_must_not_regress_wrapper_progress(paths, settings, monkeypatch, only_self_alive):
    def wrapper(p, s):
        write_status(p, {**s, "pid": os.getpid(), "step_index": 4})

    _race_start(paths, settings, None, monkeypatch, wrapper)
    assert read_status(paths)["step_index"] == 4


def test_pidfill_race_fast_finishing_wrapper_terminal_status_kept_and_no_second_run(
        paths, settings, monkeypatch, only_self_alive):
    """Wrapper finishes (done status, marker, lock released) between launcher read and write."""
    def wrapper(p, s):
        run_state.write_done_markers(p, [local_today()])
        write_status(p, {**s, "pid": os.getpid(), "state": "done", "finished_at": _iso(_now())})
        run_state.release_lock(p, s["run_id"])

    res = _race_start(paths, settings, None, monkeypatch, wrapper)
    assert res.started
    final = read_status(paths)
    assert final["state"] == "done"                            # terminal state kept (fixed)
    assert read_lock(paths) is None
    assert get_run_view(paths).state == "done"
    again = maybe_start_daily_run(paths, settings, spawn=FakeSpawn())
    assert not again.started and again.reason == "already_done"   # marker protects: no second run


def test_pidfill_race_fast_failing_wrapper_failure_status_and_failure_count_kept(
        paths, settings, monkeypatch, only_self_alive):
    def wrapper(p, s):
        record_full_failure(p, local_today())
        write_status(p, {**s, "pid": os.getpid(), "state": "failed", "error": "boom",
                         "finished_at": _iso(_now())})
        run_state.release_lock(p, s["run_id"])

    _race_start(paths, settings, None, monkeypatch, wrapper)
    assert get_run_view(paths).state == "failed"                # failure bar kept (fixed)
    spawn = FakeSpawn()
    res = maybe_start_daily_run(paths, settings, spawn=spawn)
    assert res.started and _mode_of(spawn.calls[0]) == "phase1"  # fallback, not another full run


# ====================================================================== concurrency on the launcher


def test_concurrent_start_requests_spawn_exactly_once(paths, settings, only_self_alive):
    spawn = FakeSpawn()
    orig = spawn.__call__
    results: list = []
    barrier = threading.Barrier(8)

    def slow_spawn(argv, cwd):
        time.sleep(0.05)
        return orig(argv, cwd)

    def worker():
        barrier.wait()
        results.append(maybe_start_daily_run(paths, settings, spawn=slow_spawn))

    ts = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in ts]
    [t.join(20) for t in ts]
    assert len(spawn.calls) == 1
    assert sum(r.started for r in results) == 1


def test_double_click_button_second_start_refused_while_active(paths, settings, only_self_alive):
    spawn = FakeSpawn()
    a = start_pipeline_run("phase1", 100, 50, trigger="button", paths=paths, settings=settings, spawn=spawn)
    b = start_pipeline_run("phase1", 100, 50, trigger="button", paths=paths, settings=settings, spawn=spawn)
    assert a.started and not b.started and b.reason == "running"
    assert len(spawn.calls) == 1


def test_auto_rank_disabled_while_run_active_does_not_touch_run(paths, settings, only_self_alive, make_status):
    spawn = FakeSpawn()
    start_pipeline_run("full", 100, 50, trigger="auto", paths=paths, settings=settings, spawn=spawn)
    settings.auto_rank_enabled = False
    res = maybe_start_daily_run(paths, settings, spawn=spawn)
    assert res.reason == "disabled" and len(spawn.calls) == 1
    assert get_run_view(paths).state == "running"               # bar still tracks the live run
    assert read_lock(paths) is not None


def test_auto_rank_disabled_leaves_stale_lock_for_buttons_to_break(paths, settings, only_self_alive):
    _old_lock(paths)
    settings.auto_rank_enabled = False
    res = maybe_start_daily_run(paths, settings, spawn=FakeSpawn())
    assert res.reason == "disabled" and read_lock(paths) is not None
    assert is_run_active(paths) is False                        # buttons enabled
    r2 = start_pipeline_run("phase1", 100, 50, trigger="button", paths=paths, settings=settings,
                            spawn=FakeSpawn())
    assert r2.started


def test_launcher_rejects_numpy_ints_and_nan(paths, settings):
    import numpy as np
    for pool in (np.int64(100), float("nan"), 100.0, True, "100", None):
        with pytest.raises(ValueError):
            start_pipeline_run("full", pool, 50, paths=paths, settings=settings, spawn=FakeSpawn())
    assert read_lock(paths) is None                             # no side effects


# ====================================================================== time handling


def test_midnight_rollover_mid_run_writes_markers_for_both_days(paths, settings, tmp_path, monkeypatch):
    assert try_acquire_lock(paths, RUN_ID, "2026-10-10", "full", "auto")
    script = _fake_child(tmp_path, FULL_LINES)
    monkeypatch.setattr(pj, "local_today", lambda: "2026-10-11")      # wrapper clock is past midnight
    assert pj.run_job(_job_args(paths, script, run_date="2026-10-10")) == 0
    assert has_done_marker(paths, "2026-10-10") and has_done_marker(paths, "2026-10-11")
    monkeypatch.setattr(launcher, "local_today", lambda: "2026-10-11")
    res = maybe_start_daily_run(paths, settings, spawn=FakeSpawn())
    assert not res.started and res.reason == "already_done"


def test_midnight_rollover_failed_full_run_counts_for_start_day_only(paths, settings, tmp_path, monkeypatch):
    assert try_acquire_lock(paths, RUN_ID, "2026-10-10", "full", "auto")
    script = _fake_child(tmp_path, FULL_LINES[:3], exit_code=1)
    monkeypatch.setattr(pj, "local_today", lambda: "2026-10-11")
    assert pj.run_job(_job_args(paths, script, run_date="2026-10-10")) == 1
    assert read_day_state(paths, "2026-10-10").full_failures == 1
    assert read_day_state(paths, "2026-10-11").full_failures == 0     # new day starts clean (by design)


def test_local_today_follows_local_timezone_including_dst_days(monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("tzset unavailable")
    old = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "America/New_York"
        time.tzset()
        assert local_today() == datetime.now().date().isoformat()
        class FixedDT(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 11, 1, 1, 30, 0) if tz is None else cls(2026, 11, 1, 6, 30, tzinfo=tz)
        monkeypatch.setattr(run_state, "datetime", FixedDT)
        assert local_today() == "2026-11-01"                         # ambiguous 01:30 on fall-back day
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


def test_cleanup_handles_dst_and_year_boundaries(paths):
    for d in ("2025-12-24", "2026-01-02", "2026-03-08", "2026-11-01"):
        run_state.write_done_markers(paths, [d])
    run_state.cleanup_old_files(paths, today="2026-01-05")
    assert not has_done_marker(paths, "2025-12-24")      # 12 days old: removed
    assert has_done_marker(paths, "2026-01-02")          # 3 days old: kept
    assert has_done_marker(paths, "2026-03-08") and has_done_marker(paths, "2026-11-01")   # future days kept


def test_heartbeat_slightly_in_future_with_live_pid_is_alive(paths, only_self_alive, make_status):
    _old_lock(paths, run_id="run1", age_min=1)
    write_status(paths, make_status(run_id="run1", pid=os.getpid(), heartbeat_at=_iso(_now() + timedelta(minutes=5))))
    assert is_run_active(paths)


def test_heartbeat_in_future_with_dead_pid_is_dead(paths, make_status):
    _old_lock(paths, run_id="run1", age_min=1)
    write_status(paths, make_status(run_id="run1", pid=2_000_000_000, heartbeat_at=_iso(_now() + timedelta(days=1))))
    assert not is_run_active(paths)


@pytest.mark.parametrize("state", ["done", "failed"])
def test_future_finished_at_terminal_status_is_dismissed(paths, make_status, state):
    write_status(paths, make_status(run_id="r", state=state, finished_at=_iso(_now() + timedelta(days=1)),
                                    error="x" if state == "failed" else None))
    assert get_run_view(paths).state == "idle"


def test_lock_created_in_the_far_future_is_not_alive_forever(paths):
    assert try_acquire_lock(paths, "future", local_today(), "full", "auto", _now() + timedelta(days=365))
    assert not is_run_active(paths)


def test_naive_and_aware_timestamps_are_both_accepted(paths, make_status):
    _old_lock(paths, run_id="r", age_min=0)
    write_status(paths, make_status(run_id="r", pid=os.getpid(), started_at="2026-10-11T10:00:00",
                                    heartbeat_at=_iso(_now()), step_started_at="garbage"))
    v = get_run_view(paths)
    assert v.state in ("running", "failed")
    from dashboard.components.pipeline_progress import _bar_html
    assert "step" in _bar_html(v) or v.state == "failed"


@pytest.mark.parametrize("idx,total", [(-5, 9), (0, 0), (3, 0), (999, 9), (-1, -1), (2, 1)])
def test_odd_step_numbers_render_without_error(paths, make_status, idx, total):
    from dashboard.components.pipeline_progress import _bar_html
    _old_lock(paths, run_id="r", age_min=0)
    write_status(paths, make_status(run_id="r", pid=os.getpid(), step_index=idx, step_total=total,
                                    heartbeat_at=_iso(_now())))
    v = get_run_view(paths)
    assert 0.0 <= v.progress <= 0.99
    html = _bar_html(v)
    assert "width:" in html and "-" not in html.split('style="width:')[1].split("%")[0]


def test_non_int_step_values_fall_back_to_zero(paths, make_status):
    _old_lock(paths, run_id="r", age_min=0)
    s = make_status(run_id="r", pid=os.getpid(), heartbeat_at=_iso(_now()))
    s["step_index"], s["step_total"] = "3", 9.5
    write_status(paths, s)
    v = get_run_view(paths)
    assert v.step_index == 0 and v.step_total == 0
    paths.status_file.write_text(json.dumps(s).replace('"step_index": "3"', '"step_index": NaN'), encoding="utf-8")
    assert get_run_view(paths).step_index == 0


def test_huge_step_index_does_not_raise(paths, make_status):
    _old_lock(paths, run_id="r", age_min=0)
    write_status(paths, make_status(run_id="r", pid=os.getpid(), step_index=10 ** 400, heartbeat_at=_iso(_now())))
    get_run_view(paths)


# ====================================================================== run_id / log tail


@pytest.mark.parametrize("rid", ["../x", "..", "a/b", "a\\b", "a.b", "x\x00", "", " ", "é", "１２３", "a b", "/etc/passwd"])
def test_read_log_tail_rejects_unsafe_run_ids(paths, rid):
    (paths.log_dir / "pipeline_job_x.log").write_text("secret\n", encoding="utf-8")
    assert read_log_tail(paths, rid) == []


def test_read_log_tail_trailing_newline_is_accepted_but_cannot_escape_log_dir(paths):
    (paths.log_dir / "pipeline_job_abc.log").write_text("real\n", encoding="utf-8")
    (paths.log_dir / "pipeline_job_abc\n.log").write_text("odd\n", encoding="utf-8")
    assert read_log_tail(paths, "abc\n") == ["odd"]                    # regex `$` quirk: reads a odd-named file
    target = paths.log_file("abc\n")
    assert target.resolve().parent == paths.log_dir.resolve()
    # and the str-join form used by callers that hand a str path
    assert os.path.dirname(os.path.join(str(paths.log_dir), f"pipeline_job_{'abc' + chr(10)}.log")) == str(paths.log_dir)
    # a newline followed by traversal characters is not accepted
    assert read_log_tail(paths, "abc\n../../x") == []
    assert read_log_tail(paths, "abc/../..\n") == []


def test_read_log_tail_very_long_run_id_is_harmless(paths):
    assert read_log_tail(paths, "a" * 5000) == []


def test_read_log_tail_non_str_run_id(paths):
    for rid in (None, 5, b"abc", ["a"]):
        assert read_log_tail(paths, rid) == []        # type: ignore[arg-type]


def test_read_log_tail_binary_and_missing_and_directory(paths):
    (paths.log_dir / "pipeline_job_bin.log").write_bytes(b"ok\n\xff\xfe\x00bad\nlast\n")
    out = read_log_tail(paths, "bin")
    assert out[-1] == "last" and "�" in out[1]
    assert read_log_tail(paths, "missing") == []
    (paths.log_dir / "pipeline_job_dir.log").mkdir()
    assert read_log_tail(paths, "dir") == []


def test_read_log_tail_large_log_cost(paths):
    """Settings page calls this on every rerun for the last run; it reads the WHOLE file."""
    line = ("x" * 99 + "\n")
    (paths.log_dir / "pipeline_job_big.log").write_text(line * 400_000, encoding="utf-8")   # ~40 MB
    t0 = time.perf_counter()
    out = read_log_tail(paths, "big")
    dt = time.perf_counter() - t0
    assert len(out) == 100
    assert dt < 5, f"read_log_tail took {dt:.2f}s on a 40 MB log"


# ====================================================================== phase1 artifact


def _p1(settings, entries, **over):
    payload = {"schema_version": 1, "created_at": _iso(_now()), "run_date": local_today(),
               "ticker_source": "sp500", "universe_size": len(entries), "include_indicators": True,
               "entries": entries}
    payload.update(over)
    p = phase1_result_path(settings.data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _backdate_raw(settings):
    for f in settings.raw_daily_dir.glob("*.parquet"):
        old = time.time() - 36000
        os.utime(f, (old, old))


def test_phase1_100k_entries_check_is_fast(settings):
    _backdate_raw(settings)
    _p1(settings, [{"ticker": f"T{i}", "rank": i + 1, "score": 1.0} for i in range(100_000)])
    t0 = time.perf_counter()
    r = check_phase1_reusable(settings, pool=100)
    dt = time.perf_counter() - t0
    assert r.usable and len(r.pool_tickers) == 100 and r.ranked_count == 100_000
    assert dt < 3, f"{dt:.2f}s"


def test_phase1_empty_entries_not_usable(settings):
    _backdate_raw(settings)
    _p1(settings, [])
    r = check_phase1_reusable(settings, pool=100)
    assert not r.usable and r.reason == "empty"


def test_phase1_pool_larger_than_ranked_returns_all(settings):
    _backdate_raw(settings)
    _p1(settings, [{"ticker": "AAA", "rank": 1, "score": 1}, {"ticker": "BBB", "rank": 2, "score": None}])
    r = check_phase1_reusable(settings, pool=500)
    assert r.usable and r.pool_tickers == ["AAA", "BBB"]


@pytest.mark.parametrize("pool", [0, -5, None])
def test_phase1_odd_pool_values(settings, pool):
    _backdate_raw(settings)
    _p1(settings, [{"ticker": f"T{i}", "rank": i} for i in range(10)])
    r = check_phase1_reusable(settings, pool=pool)
    assert r.usable
    if pool is None:
        assert r.pool_tickers is None
    elif pool == 0:
        assert r.pool_tickers == []
    else:
        assert len(r.pool_tickers) == 5          # entries[:-5] silently, a negative pool is not rejected


def test_phase1_nan_or_str_pool_never_raises(settings):
    _backdate_raw(settings)
    _p1(settings, [{"ticker": "AAA", "rank": 1}])
    for pool in (float("nan"), "5", 2.5):
        r = check_phase1_reusable(settings, pool=pool)     # type: ignore[arg-type]
        assert r.usable is False


def test_phase1_weird_tickers_are_passed_through_verbatim(settings):
    _backdate_raw(settings)
    weird = ["aapl", "BRK.B", " AAPL\n", "../x", "A\x00B", "日本", "A" * 5000]
    _p1(settings, [{"ticker": t, "rank": i + 1, "score": 1} for i, t in enumerate(weird)])
    r = check_phase1_reusable(settings, pool=100)
    assert r.usable and r.pool_tickers == weird         # no normalisation or validation of the saved tickers


@pytest.mark.parametrize("bad", [
    [{"ticker": 5}], [{"ticker": ""}], ["AAA"], [None], [{"rank": 1}],
])
def test_phase1_bad_entries_unreadable(settings, bad):
    _backdate_raw(settings)
    _p1(settings, bad)
    assert check_phase1_reusable(settings, pool=10).reason == "unreadable"


@pytest.mark.parametrize("text", ["", "{", "null", "[]", "﻿{}", "NaN"])
def test_phase1_corrupt_files_unreadable(settings, text):
    p = phase1_result_path(settings.data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    r = check_phase1_reusable(settings, pool=10)
    assert not r.usable


def test_phase1_bom_prefixed_valid_json_is_unreadable(settings):
    _backdate_raw(settings)
    p = _p1(settings, [{"ticker": "AAA", "rank": 1}])
    p.write_text("﻿" + p.read_text(encoding="utf-8"), encoding="utf-8")
    assert not check_phase1_reusable(settings, pool=10).usable


def test_phase1_created_at_variants(settings):
    _backdate_raw(settings)
    for ca, ok in ((_iso(_now()).replace("+00:00", "Z"), True), ("2026-10-11T10:00:00", None), ("nope", False),
                   (123456, False), (_iso(_now() + timedelta(days=30)), True)):
        _p1(settings, [{"ticker": "AAA", "rank": 1}], created_at=ca)
        r = check_phase1_reusable(settings, pool=10)
        if ok is not None:
            assert r.usable is ok, ca


def test_phase1_price_data_newer_than_artifact_invalidates(settings):
    _p1(settings, [{"ticker": "AAA", "rank": 1}], created_at=_iso(_now() - timedelta(hours=1)))
    f = next(settings.raw_daily_dir.glob("*.parquet"))
    os.utime(f, None)
    assert check_phase1_reusable(settings, pool=10).reason.startswith("price data updated")


def test_phase1_yesterdays_artifact_across_midnight_not_reused(settings):
    _backdate_raw(settings)
    _p1(settings, [{"ticker": "AAA", "rank": 1}])
    assert check_phase1_reusable(settings, pool=10, today="2099-01-01").reason.startswith("not from today")


# ====================================================================== wrapper process behaviour


def test_child_non_utf8_and_nul_output_is_survived(paths, tmp_path):
    assert try_acquire_lock(paths, RUN_ID, local_today(), "full", "auto")
    script = _fake_child(tmp_path, [], code="""
        sys.stdout.buffer.write(b"\\xff\\xfe\\x80 bad bytes\\x00 nul\\n"); sys.stdout.flush()
        print("[Phase 1] Fast-ranking 5", flush=True)
    """)
    assert pj.run_job(_job_args(paths, script)) == 0
    assert read_status(paths)["state"] == "done"
    assert "�" in paths.log_file(RUN_ID).read_text(encoding="utf-8")


def test_child_prints_10mb_without_newline_then_succeeds(paths, tmp_path):
    assert try_acquire_lock(paths, RUN_ID, local_today(), "full", "auto")
    script = _fake_child(tmp_path, [], code="""
        sys.stdout.write("x" * 10_000_000); sys.stdout.write("\\n"); sys.stdout.flush()
        print("[Phase 1] Fast-ranking 5", flush=True)
    """)
    t0 = time.time()
    assert pj.run_job(_job_args(paths, script)) == 0
    assert time.time() - t0 < 30
    assert read_status(paths)["step_index"] == len(pj.PIPELINE_STEPS_FULL) - 1


def test_failed_run_with_huge_last_line_keeps_status_small(paths, tmp_path):
    assert try_acquire_lock(paths, RUN_ID, local_today(), "full", "auto")
    script = _fake_child(tmp_path, [], code="""
        sys.stdout.write("E" * 10_000_000 + "\\n"); sys.stdout.flush()
    """, exit_code=3)
    assert pj.run_job(_job_args(paths, script)) == 1
    st = read_status(paths)
    assert st["state"] == "failed" and len(st["error"]) <= pj.ERROR_MAX_CHARS
    assert paths.status_file.stat().st_size < 5000


def test_child_closes_stdout_and_keeps_running_is_killed_at_time_limit(paths, tmp_path):
    assert try_acquire_lock(paths, RUN_ID, local_today(), "full", "auto")
    script = _fake_child(tmp_path, [], code="""
        os.close(1); os.close(2); time.sleep(40)
    """)
    t0 = time.time()
    rc = pj.run_job(_job_args(paths, script, extra=["--max-runtime-s", "2"]))
    assert rc == 1 and time.time() - t0 < 15
    assert "Timed out" in read_status(paths)["error"]
    assert read_lock(paths) is None


def test_child_ignoring_sigterm_is_killed_after_grace(paths, tmp_path, monkeypatch):
    monkeypatch.setattr(pj, "KILL_GRACE_S", 1)
    assert try_acquire_lock(paths, RUN_ID, local_today(), "full", "auto")
    script = _fake_child(tmp_path, [], code="""
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("[Phase 1] Fast-ranking 5", flush=True)
        time.sleep(60)
    """)
    t0 = time.time()
    rc = pj.run_job(_job_args(paths, script, extra=["--max-runtime-s", "1"]))
    assert rc == 1 and time.time() - t0 < 12
    assert read_status(paths)["state"] == "failed"
    assert read_lock(paths) is None


def test_grandchild_holding_stdout_does_not_defeat_time_limit(paths, tmp_path):
    assert try_acquire_lock(paths, RUN_ID, local_today(), "full", "auto")
    script = _fake_child(tmp_path, ["[Phase 1] Fast-ranking 5"], code="""
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(7)"])   # inherits stdout
        time.sleep(60)
    """)
    t0 = time.time()
    rc = pj.run_job(_job_args(paths, script, extra=["--max-runtime-s", "1"]))
    elapsed = time.time() - t0
    assert rc == 1
    assert elapsed < 4, f"wrapper returned after {elapsed:.1f}s although the limit was 1s"


def test_flood_of_100k_short_lines_finishes_reasonably(paths, tmp_path):
    assert try_acquire_lock(paths, RUN_ID, local_today(), "full", "auto")
    script = _fake_child(tmp_path, [], code="""
        sys.stdout.write("tick\\n" * 100_000); sys.stdout.flush()
    """)
    t0 = time.time()
    assert pj.run_job(_job_args(paths, script)) == 0
    assert time.time() - t0 < 30


def _spawn_wrapper(paths, script, run_id, extra=()):
    cmd = [sys.executable, str(ROOT / "cli" / "pipeline_job.py"), "--run-id", run_id,
           "--run-date", local_today(), "--mode", "full", "--pool", "100", "--top", "50",
           "--trigger", "auto", "--data-dir", str(paths.data_dir), "--pipeline-script", str(script),
           "--heartbeat-s", "0.5", *extra]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def _wait_for(pred, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.1)
    return None


def _kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(5)
    except Exception:
        pass


def _reap_child(pid):
    """Kill a (possibly orphaned) pipeline child and its process group; escalate TERM -> KILL; verify gone."""
    if not pid:
        return
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        pgid = None
    if pgid == os.getpgrp():
        pgid = None

    def _alive():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        try:  # zombie counts as gone
            with open(f"/proc/{pid}/stat") as f:
                return f.read().rsplit(")", 1)[1].split()[0] != "Z"
        except OSError:
            return False

    for sig, wait_s in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 5.0)):
        if not _alive():
            break
        for send in ((lambda: os.killpg(pgid, sig)) if pgid else None, lambda: os.kill(pid, sig)):
            if send is None:
                continue
            try:
                send()
            except (ProcessLookupError, PermissionError):
                pass
        end = time.time() + wait_s
        while time.time() < end and _alive():
            time.sleep(0.05)
    assert not _alive(), f"could not clean up pipeline child pid {pid}"


def test_wrapper_sigkill_mid_run_stale_lock_then_page_load(paths, settings, tmp_path):
    """Hard-killed wrapper: lock is stale, view shows failed, the next load counts one full failure and
    starts the Phase-1 fallback (not a second full run)."""
    assert try_acquire_lock(paths, RUN_ID, local_today(), "full", "auto")
    script = _fake_child(tmp_path, ["[Phase 1] Fast-ranking 5"], code="time.sleep(30)")
    proc = _spawn_wrapper(paths, script, RUN_ID)
    child = None
    try:
        st = _wait_for(lambda: (s := read_status(paths)) and s.get("child_pid") and s)
        assert st, "wrapper did not start"
        child = st["child_pid"]
        os.kill(st["pid"], signal.SIGKILL)
        proc.wait(5)
        assert _wait_for(lambda: not is_run_active(paths), 5)
        v = get_run_view(paths)
        assert v.state == "failed" and "stopped responding" in (v.error or "")
        spawn = FakeSpawn()
        res = maybe_start_daily_run(paths, settings, spawn=spawn)
        assert res.started and _mode_of(spawn.calls[0]) == "phase1"
        assert read_day_state(paths).full_failures == 1
    finally:
        _kill_group(proc)
        _reap_child(child)


@pytest.mark.xfail(strict=True, reason="BUG: SIGKILL of the wrapper orphans the pipeline child; it keeps "
                   "running (and later writes ranking_cache.parquet) while the stale lock is broken and a "
                   "fallback/new run starts (cli/pipeline_job.py has no parent-death handling)")
def test_wrapper_sigkill_must_not_leave_orphan_pipeline_child(paths, tmp_path):
    assert try_acquire_lock(paths, RUN_ID, local_today(), "full", "auto")
    script = _fake_child(tmp_path, ["[Phase 1] Fast-ranking 5"], code="time.sleep(30)")
    proc = _spawn_wrapper(paths, script, RUN_ID)
    child = None
    try:
        st = _wait_for(lambda: (s := read_status(paths)) and s.get("child_pid") and s)
        assert st
        child = st["child_pid"]
        os.kill(st["pid"], signal.SIGKILL)
        proc.wait(5)
        time.sleep(2)
        assert not run_state.pid_alive(child), "orphaned pipeline child still running after wrapper SIGKILL"
    finally:
        _kill_group(proc)
        _reap_child(child)


def test_two_wrappers_for_the_same_run_id_both_run_the_pipeline(paths, tmp_path):
    """No exclusivity between two wrappers that share a run id (not reachable via the launcher)."""
    assert try_acquire_lock(paths, RUN_ID, local_today(), "full", "auto")
    counter = tmp_path / "runs.txt"
    script = _fake_child(tmp_path, [], code=f"""
        open({str(counter)!r}, "a").write("x\\n"); time.sleep(1.5)
    """)
    a = _spawn_wrapper(paths, script, RUN_ID)
    b = _spawn_wrapper(paths, script, RUN_ID)
    try:
        a.wait(40)
        b.wait(40)
        assert counter.read_text().count("x") == 2
    finally:
        _kill_group(a)
        _kill_group(b)


def test_wrapper_refuses_to_start_without_its_lock(paths, tmp_path):
    script = _fake_child(tmp_path, FULL_LINES)
    assert pj.run_job(_job_args(paths, script)) == pj.EXIT_BAD
    assert try_acquire_lock(paths, "other", local_today(), "full", "auto")
    assert pj.run_job(_job_args(paths, script)) == pj.EXIT_BAD
    assert read_lock(paths).run_id == "other"


# ====================================================================== streamlit component


def test_bar_html_escapes_markup_in_error_and_label(paths):
    from dashboard.components.pipeline_progress import _bar_html
    evil = '<script>alert(1)</script><img src=x onerror=alert(2)>"\'& ' + "A" * 5000
    v = run_state.RunView("failed", "r", "full", "auto", False, evil, 1, 9, 0.5, evil, None, None,
                          _iso(_now()))
    h = _bar_html(v, fallback_pending=True)
    assert "<script" not in h and "<img" not in h and "onerror=alert" not in h.replace("&lt;img src=x onerror", "")
    assert len(h) < 4000                                 # error clipped to one 140-char line
    v2 = run_state.RunView("running", "r", "full", "auto", False, evil, 1, 9, 0.5, None, _iso(_now()),
                           _iso(_now()), None)
    h2 = _bar_html(v2)
    assert "<script" not in h2 and "<img" not in h2


def test_bar_html_error_with_blank_lines_does_not_break_html_block(paths):
    from dashboard.components.pipeline_progress import _bar_html
    v = run_state.RunView("failed", "r", "full", "auto", False, "x", 1, 9, 0.5,
                          "\n\n\n[click](javascript:alert(1))", None, None, _iso(_now()))
    assert "\n\n" not in _bar_html(v)


def test_bar_html_label_with_blank_line_keeps_single_html_block():
    """Labels come from the static step tables, but a tampered status file could carry newlines."""
    from dashboard.components.pipeline_progress import _bar_html
    v = run_state.RunView("running", "r", "full", "auto", False, "a\n\n[x](javascript:alert(1))", 1, 9, 0.3,
                          None, _iso(_now()), _iso(_now()), None)
    assert "\n\n" in _bar_html(v)       # documents: not neutralised (low risk, local file only)


def test_partial_status_json_is_idle_not_an_exception(paths, make_status):
    s = json.dumps(make_status(run_id="r"))
    for cut in (0, 1, len(s) // 2, len(s) - 1):
        paths.status_file.write_text(s[:cut], encoding="utf-8")
        assert get_run_view(paths).state == "idle"
        assert read_status(paths) is None


def test_status_read_while_being_rewritten_never_sees_partial_file(paths, make_status):
    """Atomic replace: hammer writer and reader on separate threads."""
    stop = threading.Event()
    bad: list = []
    base = make_status(run_id="r", label="L" * 2000)

    def writer():
        i = 0
        while not stop.is_set():
            write_status(paths, {**base, "step_index": i % 9})
            i += 1

    def reader():
        seen_none = 0
        while not stop.is_set():
            if read_status(paths) is None and paths.status_file.exists():
                seen_none += 1
        bad.append(seen_none)

    write_status(paths, base)
    ts = [threading.Thread(target=writer), threading.Thread(target=reader)]
    [t.start() for t in ts]
    time.sleep(1.5)
    stop.set()
    [t.join(5) for t in ts]
    assert bad == [0]


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_get_run_paths_on_read_only_data_dir_does_not_raise(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        get_run_paths(ro / "data")
    finally:
        ro.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_render_pipeline_progress_swallows_read_only_data_dir(tmp_path, monkeypatch):
    from config import settings as config_settings
    import dashboard.components.pipeline_progress as pp
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    ns = types.SimpleNamespace(data_dir=ro / "data", raw_daily_dir=ro / "data" / "raw",
                               ticker_source="sp500", auto_rank_enabled=True)
    monkeypatch.setattr(config_settings, "get_settings", lambda: ns)
    monkeypatch.setattr(launcher, "get_settings", lambda: ns)
    try:
        pp.render_pipeline_progress()            # must not raise into a page
    finally:
        ro.chmod(0o755)


def test_missing_data_dir_is_created_and_run_is_idle(tmp_path):
    p = get_run_paths(tmp_path / "a" / "b" / "c")
    assert get_run_view(p).state == "idle" and not is_run_active(p)


# ====================================================================== AUTO_RANK_ENABLED parsing


@pytest.fixture
def _settings_cls(monkeypatch):
    from config.settings import AlertSettings, AlpacaSettings, FredSettings, Settings
    for cls in (Settings, AlpacaSettings, FredSettings, AlertSettings):
        monkeypatch.setitem(cls.model_config, "env_file", None)
    monkeypatch.delenv("AUTO_RANK_ENABLED", raising=False)
    return Settings


@pytest.mark.parametrize("raw,expected", [
    (0, False), (1, True), (2, False), (1.0, False), (0.0, False), ("TRUE\n", True), ("  no ", False),
    ("enabled", False), ("None", False), ([], False),
])
def test_auto_rank_enabled_odd_values(_settings_cls, raw, expected):
    assert _settings_cls(auto_rank_enabled=raw).auto_rank_enabled is expected


def test_auto_rank_enabled_from_environment_variable(_settings_cls, monkeypatch):
    monkeypatch.setenv("AUTO_RANK_ENABLED", "FALSE")
    assert _settings_cls().auto_rank_enabled is False
    monkeypatch.setenv("AUTO_RANK_ENABLED", "")
    assert _settings_cls().auto_rank_enabled is True


# ====================================================================== ranker cache loader


def _write_cache(tmp_path, rows, drop=()):
    from src.ranking.ranker import _CACHE_FILENAME
    df = pd.DataFrame(rows).set_index("ticker")
    df = df.drop(columns=list(drop))
    p = tmp_path / _CACHE_FILENAME
    df.to_parquet(p)
    return types.SimpleNamespace(data_dir=tmp_path)


def _row(t, score, rank, **kw):
    return {"ticker": t, "composite_score": score, "rank": rank, "signals_available": 3,
            "updated_at": "2026-10-11T10:00:00", **kw}


def test_ranker_all_rows_failed_returns_empty_list_not_none(tmp_path):
    from src.ranking.ranker import load_ranking_cache
    s = _write_cache(tmp_path, [_row("A", float("-inf"), 0), _row("B", float("-inf"), 0)])
    assert load_ranking_cache(s) == []


def test_ranker_nan_score_rows_skipped(tmp_path):
    from src.ranking.ranker import load_ranking_cache
    s = _write_cache(tmp_path, [_row("A", float("nan"), 1), _row("B", 2.0, 2)])
    assert [e.ticker for e in load_ranking_cache(s)] == ["B"]


def test_ranker_float_and_string_rank_values(tmp_path):
    from src.ranking.ranker import load_ranking_cache
    s = _write_cache(tmp_path, [_row("A", 1.0, 2.0), _row("B", 2.0, 1.0)])
    assert [(e.ticker, e.rank) for e in load_ranking_cache(s)] == [("B", 1), ("A", 2)]
    s = _write_cache(tmp_path, [_row("A", 1.0, "2"), _row("B", 2.0, "1")])
    assert [(e.ticker, e.rank) for e in load_ranking_cache(s)] == [("B", 1), ("A", 2)]


def test_ranker_non_numeric_or_nan_rank_behaviour(tmp_path):
    """Record what happens: int() on NaN / 'x' raises ValueError out of load_ranking_cache."""
    from src.ranking.ranker import load_ranking_cache
    s = _write_cache(tmp_path, [_row("A", 1.0, float("nan")), _row("B", 2.0, 1.0)])
    try:
        out = load_ranking_cache(s)
        assert [e.ticker for e in out] == ["B"]
    except ValueError:
        pass


def test_ranker_nan_rank_row_is_skipped_not_raised(tmp_path):
    from src.ranking.ranker import load_ranking_cache
    s = _write_cache(tmp_path, [_row("A", 1.0, float("nan")), _row("B", 2.0, 1.0)])
    assert [e.ticker for e in load_ranking_cache(s)] == ["B"]


def test_ranker_single_valid_row_and_negative_rank(tmp_path):
    from src.ranking.ranker import load_ranking_cache
    s = _write_cache(tmp_path, [_row("A", float("-inf"), 0), _row("B", 1.5, 1), _row("C", 1.0, -1)])
    assert [e.ticker for e in load_ranking_cache(s)] == ["B"]


def test_ranker_legacy_cache_without_updated_at_or_signals_available(tmp_path):
    from src.ranking.ranker import load_ranking_cache
    s = _write_cache(tmp_path, [_row("A", 1.0, 1), _row("B", 0.5, 2)], drop=("updated_at", "signals_available"))
    out = load_ranking_cache(s)
    assert [e.ticker for e in out] == ["A", "B"]


def test_ranker_missing_file_none_and_corrupt_file_none(tmp_path):
    from src.ranking.ranker import _CACHE_FILENAME, load_ranking_cache
    s = types.SimpleNamespace(data_dir=tmp_path)
    assert load_ranking_cache(s) is None
    (tmp_path / _CACHE_FILENAME).write_bytes(b"not parquet")
    assert load_ranking_cache(s) is None


def test_ranker_positive_infinity_score_is_kept(tmp_path):
    from src.ranking.ranker import load_ranking_cache
    s = _write_cache(tmp_path, [_row("A", float("inf"), 1)])
    assert len(load_ranking_cache(s)) == 1       # documents: only -inf/NaN are treated as failures


def test_ranker_unsorted_rows_are_returned_in_rank_order(tmp_path):
    from src.ranking.ranker import load_ranking_cache
    s = _write_cache(tmp_path, [_row("C", 1, 3), _row("A", 3, 1), _row("B", 2, 2)])
    assert [e.ticker for e in load_ranking_cache(s)] == ["A", "B", "C"]


# ====================================================================== page 6 with read-only data dir


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_page6_survives_read_only_data_dir(tmp_path, monkeypatch):
    from config import settings as config_settings
    import dashboard.components.market_clock as market_clock
    import dashboard.components.pipeline_progress as progress
    data = tmp_path / "data"
    raw = data / "raw" / "daily"
    raw.mkdir(parents=True)
    (raw / "AAA.parquet").write_bytes(b"x")
    ro = data                                  # data/ is read-only and has no run_cache yet
    ns = types.SimpleNamespace(data_dir=data, raw_daily_dir=raw, ticker_source="sp500", auto_rank_enabled=False)
    monkeypatch.setattr(config_settings, "get_settings", lambda: ns)
    monkeypatch.setattr(market_clock, "render_market_clock", lambda: None)
    monkeypatch.setattr(progress, "render_pipeline_progress", lambda: None)
    fake_ranker = types.ModuleType("src.ranking.ranker")
    fake_ranker.top_picks = lambda *a, **k: []
    fake_ranker.load_ranking_cache = lambda settings=None: None
    fake_ranker.to_dataframe = lambda picks: pd.DataFrame()
    monkeypatch.setitem(sys.modules, "src.ranking.ranker", fake_ranker)
    monkeypatch.setattr(launcher, "reap_children", lambda: None)
    import streamlit as st
    st.cache_data.clear()
    ro.chmod(0o555)
    try:
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "dashboard" / "pages" / "6_stock_rankings.py"), default_timeout=90)
        at.run()
        assert not at.exception
    finally:
        ro.chmod(0o755)
        st.cache_data.clear()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
@pytest.mark.parametrize("entry", ["button", "daily"])
def test_read_only_data_dir_start_returns_spawn_error_not_raise(tmp_path, settings, entry):
    data_dir = settings.data_dir
    data_dir.chmod(0o555)
    try:
        ro_paths = get_run_paths(data_dir)       # mkdir failure is swallowed
        spawn = FakeSpawn()
        if entry == "button":
            res = start_pipeline_run("full", paths=ro_paths, settings=settings, spawn=spawn)
        else:
            res = maybe_start_daily_run(paths=ro_paths, settings=settings, spawn=spawn)
        assert res.started is False
        assert res.reason == "spawn_error"
    finally:
        data_dir.chmod(0o755)
