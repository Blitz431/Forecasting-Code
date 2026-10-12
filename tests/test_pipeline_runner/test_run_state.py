"""Tests for src/pipeline_runner/run_state.py (plan section 11, items 13-25)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

from src.pipeline_runner import run_state as rs
from src.pipeline_runner.run_state import (
    BREAKER_STALE_S,
    DONE_DISMISS_S,
    FAILED_DISMISS_S,
    LOG_RETENTION_COUNT,
    PID_REUSE_GUARD_S,
    SPAWN_GRACE_S,
    DayState,
    RunPaths,
    break_stale_lock,
    cleanup_old_files,
    get_run_paths,
    get_run_view,
    has_done_marker,
    is_run_active,
    local_today,
    lock_owner_alive,
    new_run_id,
    pid_alive,
    ranking_cache_version,
    read_day_state,
    read_json,
    read_lock,
    read_log_tail,
    read_status,
    record_fallback_started,
    record_full_failure,
    record_phase1_success,
    release_lock,
    try_acquire_lock,
    write_done_markers,
    write_json_atomic,
    write_status,
)
from tests.test_pipeline_runner.conftest import NOW, iso

TODAY = local_today()


def S(seconds: float):
    return timedelta(seconds=seconds)


# ------------------------------------------------------------------ paths / ids / json


def test_get_run_paths_creates_dirs(tmp_path):
    p = get_run_paths(tmp_path / "d")
    assert p.run_cache_dir == tmp_path / "d" / "run_cache" and p.run_cache_dir.is_dir()
    assert p.log_dir == tmp_path / "d" / "logs" and p.log_dir.is_dir()
    assert p.lock_file.name == "pipeline_run.lock"
    assert p.breaker_file.name == "pipeline_run.lock.break"
    assert p.status_file.name == "pipeline_status.json"
    assert p.done_marker("2026-10-11").name == ".done_rank_2026-10-11"
    assert p.day_state_file("2026-10-11").name == "auto_rank_2026-10-11.json"
    assert p.log_file("abc").name == "pipeline_job_abc.log"


def test_new_run_id_format():
    rid = new_run_id(NOW)
    assert re.match(r"^\d{8}T\d{6}Z-[0-9a-f]{6}$", rid)
    assert rid.startswith("20261011T150000Z-")
    assert re.match(r"^\d{8}T\d{6}Z-[0-9a-f]{6}$", new_run_id())
    assert new_run_id(NOW) != new_run_id(NOW)


def test_write_read_json_round_trip(tmp_path):
    p = tmp_path / "a.json"
    write_json_atomic(p, {"k": "v…", "n": 1})
    assert read_json(p) == {"k": "v…", "n": 1}
    assert "…" in p.read_text(encoding="utf-8")   # ensure_ascii=False


@pytest.mark.parametrize("content", ["", "   \n", "{bad", "[1, 2]", "42", '"str"', "null"])
def test_read_json_invalid_gives_none(tmp_path, content):
    p = tmp_path / "a.json"
    p.write_text(content, encoding="utf-8")
    assert read_json(p) is None


def test_read_json_missing(tmp_path):
    assert read_json(tmp_path / "nope.json") is None


def test_write_json_atomic_leaves_no_tmp(tmp_path):
    p = tmp_path / "x.json"

    def w(i):
        for j in range(20):
            write_json_atomic(p, {"i": i, "j": j})

    threads = [threading.Thread(target=w, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert [f.name for f in tmp_path.iterdir()] == ["x.json"]
    assert read_json(p) is not None


def test_write_json_atomic_cleans_tmp_on_failure(tmp_path):
    p = tmp_path / "x.json"
    with pytest.raises(TypeError):
        write_json_atomic(p, {"bad": object()})
    assert list(tmp_path.iterdir()) == []


def test_read_status_validation(paths, make_status, write_st):
    assert read_status(paths) is None
    write_st(make_status())
    assert read_status(paths)["run_id"] == "run1"
    write_st(make_status(schema_version=2))
    assert read_status(paths) is None
    write_st(make_status(state="weird"))
    assert read_status(paths) is None
    write_st(make_status(state=None))
    assert read_status(paths) is None
    paths.status_file.write_text("{oops", encoding="utf-8")
    assert read_status(paths) is None


# ------------------------------------------------------------------ lock


def test_lock_acquire_release(paths):
    assert read_lock(paths) is None
    assert try_acquire_lock(paths, "r1", TODAY, "full", "auto", NOW) is True
    assert try_acquire_lock(paths, "r2", TODAY, "full", "auto", NOW) is False
    lock = read_lock(paths)
    assert lock.run_id == "r1" and lock.mode == "full" and lock.trigger == "auto"
    assert lock.run_date == TODAY and lock.created_at == NOW
    assert release_lock(paths, "wrong") is False
    assert paths.lock_file.exists()
    assert release_lock(paths, "r1") is True
    assert not paths.lock_file.exists()
    assert release_lock(paths, "r1") is False   # already gone


def test_lock_file_contents(paths):
    try_acquire_lock(paths, "r1", TODAY, "phase1", "button", NOW)
    data = json.loads(paths.lock_file.read_text(encoding="utf-8"))
    assert data["launcher_pid"] == os.getpid()
    assert data["created_at"] == iso(NOW)


def test_lock_race_16_threads(paths):
    n = 16
    barrier = threading.Barrier(n)
    results = []
    rl = threading.Lock()

    def worker(i):
        barrier.wait()
        ok = try_acquire_lock(paths, f"run{i}", TODAY, "full", "auto", NOW)
        with rl:
            results.append(ok)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert results.count(True) == 1 and results.count(False) == n - 1


_PROC_SCRIPT = """
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from src.pipeline_runner.run_state import get_run_paths, try_acquire_lock
p = get_run_paths(Path(sys.argv[2]))
gate = Path(sys.argv[3])
deadline = time.time() + 60
while not gate.exists() and time.time() < deadline:
    time.sleep(0.005)
print("RESULT", try_acquire_lock(p, "proc" + sys.argv[4], "2026-10-11", "full", "auto"))
"""


def test_lock_race_4_processes(paths, tmp_path):
    """Four separate OS processes (fresh interpreters) race for the lock released by a start gate."""
    root = str(Path(__file__).resolve().parents[2])
    gate = tmp_path / "gate"
    procs = [subprocess.Popen([sys.executable, "-c", _PROC_SCRIPT, root, str(paths.data_dir), str(gate), str(i)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(4)]
    try:
        time.sleep(1.5)           # let the interpreters start and reach the gate
        gate.write_text("go")
        outs = [p.communicate(timeout=90) for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait(10)
    results = []
    for p, (out, err) in zip(procs, outs):
        assert p.returncode == 0, err
        results.append(out.split("RESULT")[1].strip())
    assert results.count("True") == 1 and results.count("False") == 3


def test_corrupt_lock_aged_by_mtime(paths):
    paths.lock_file.write_text("", encoding="utf-8")
    lock = read_lock(paths)
    assert lock.run_id is None
    assert abs(lock.created_at.timestamp() - paths.lock_file.stat().st_mtime) < 1
    # fresh by mtime: alive within grace
    assert lock_owner_alive(paths, lock) is True
    old = time.time() - SPAWN_GRACE_S - 30
    os.utime(paths.lock_file, (old, old))
    lock = read_lock(paths)
    assert lock_owner_alive(paths, lock) is False
    # garbage JSON and unparseable created_at also fall back to mtime
    paths.lock_file.write_text('{"run_id": "r", "created_at": "garbage"}', encoding="utf-8")
    os.utime(paths.lock_file, (old, old))
    lock = read_lock(paths)
    assert lock.run_id == "r" and abs(lock.created_at.timestamp() - old) < 1
    assert lock_owner_alive(paths, lock) is False


def test_corrupt_lock_can_be_broken(paths):
    paths.lock_file.write_text("{{", encoding="utf-8")
    old = time.time() - 3600
    os.utime(paths.lock_file, (old, old))
    assert break_stale_lock(paths) is True
    assert not paths.lock_file.exists()


# ------------------------------------------------------------------ liveness


def test_liveness_no_status_grace(paths, make_lock):
    make_lock(now=NOW)
    lock = read_lock(paths)
    assert lock_owner_alive(paths, lock, NOW + S(5)) is True
    assert lock_owner_alive(paths, lock, NOW + S(SPAWN_GRACE_S + 1)) is False


def test_liveness_status_other_run_id_uses_grace(paths, make_lock, make_status, write_st, only_self_alive):
    make_lock(run_id="mine", now=NOW)
    write_st(make_status(run_id="other", pid=os.getpid(), heartbeat_at=iso(NOW)))
    lock = read_lock(paths)
    assert lock_owner_alive(paths, lock, NOW + S(SPAWN_GRACE_S + 1)) is False
    assert lock_owner_alive(paths, lock, NOW + S(1)) is True


def test_liveness_running_long_silent_but_pid_alive(paths, make_lock, make_status, write_st):
    make_lock(run_id="run1", now=NOW - timedelta(hours=3))
    write_st(make_status(pid=os.getpid(), started_at=iso(NOW - timedelta(hours=3)),
                         heartbeat_at=iso(NOW - S(600))))
    assert lock_owner_alive(paths, read_lock(paths), NOW) is True


def test_liveness_heartbeat_older_than_guard_is_dead(paths, make_lock, make_status, write_st):
    make_lock(now=NOW - timedelta(hours=3))
    write_st(make_status(pid=os.getpid(), heartbeat_at=iso(NOW - S(PID_REUSE_GUARD_S + 5))))
    assert lock_owner_alive(paths, read_lock(paths), NOW) is False


def test_liveness_dead_pid_with_fresh_heartbeat_is_dead(paths, make_lock, make_status, write_st, monkeypatch):
    monkeypatch.setattr(rs, "pid_alive", lambda pid: False)
    make_lock(now=NOW)
    write_st(make_status(pid=424242, heartbeat_at=iso(NOW - S(1))))
    assert lock_owner_alive(paths, read_lock(paths), NOW) is False


def test_liveness_pid_none_uses_grace(paths, make_lock, make_status, write_st):
    make_lock(now=NOW)
    write_st(make_status(pid=None))
    lock = read_lock(paths)
    assert lock_owner_alive(paths, lock, NOW + S(10)) is True
    assert lock_owner_alive(paths, lock, NOW + S(SPAWN_GRACE_S + 1)) is False


@pytest.mark.parametrize("state", ["done", "failed"])
def test_liveness_terminal_status(paths, make_lock, make_status, write_st, monkeypatch, state):
    make_lock(now=NOW - S(500))
    write_st(make_status(state=state, pid=777, finished_at=iso(NOW - S(10))))
    lock = read_lock(paths)
    monkeypatch.setattr(rs, "pid_alive", lambda pid: False)
    assert lock_owner_alive(paths, lock, NOW) is True                     # within grace
    assert lock_owner_alive(paths, lock, NOW + S(SPAWN_GRACE_S + 1)) is False
    monkeypatch.setattr(rs, "pid_alive", lambda pid: True)
    assert lock_owner_alive(paths, lock, NOW + S(SPAWN_GRACE_S + 1)) is True  # pid still alive


def test_is_run_active(paths, make_lock, make_status, write_st):
    assert is_run_active(paths) is False
    make_lock(now=datetime.now(timezone.utc))
    assert is_run_active(paths) is True
    write_st(make_status(pid=os.getpid(), heartbeat_at=iso(datetime.now(timezone.utc))))
    assert is_run_active(paths) is True


# ------------------------------------------------------------------ pid_alive


def test_pid_alive_basics():
    assert pid_alive(os.getpid()) is True
    for bad in (None, 0, -3, True, "12", 1.5):
        assert pid_alive(bad) is False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")
def test_pid_alive_reaped_child_is_dead():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    pid = p.pid
    p.wait(30)
    assert pid_alive(pid) is False


def test_pid_alive_windows_never_uses_os_kill(monkeypatch):
    calls = []
    monkeypatch.setattr(rs.os, "kill", lambda *a: (_ for _ in ()).throw(AssertionError("os.kill used")))
    monkeypatch.setattr(rs, "_pid_alive_windows", lambda pid: calls.append(pid) or True)
    monkeypatch.setattr(rs.sys, "platform", "win32")
    assert pid_alive(4321) is True
    assert calls == [4321]


# ------------------------------------------------------------------ break_stale_lock


def _stale_full(paths, make_lock, make_status, write_st, monkeypatch, mode="full"):
    monkeypatch.setattr(rs, "pid_alive", lambda pid: False)
    make_lock(run_id="run1", mode=mode, now=NOW - timedelta(hours=1))
    write_st(make_status(run_id="run1", mode=mode, pid=999, heartbeat_at=iso(NOW - S(5))))


def test_break_stale_full_dead_pid(paths, make_lock, make_status, write_st, monkeypatch):
    _stale_full(paths, make_lock, make_status, write_st, monkeypatch)
    assert break_stale_lock(paths, NOW) is True
    assert not paths.lock_file.exists() and not paths.breaker_file.exists()
    s = read_status(paths)
    assert s["state"] == "failed" and s["finished_at"] == iso(NOW)
    assert "stopped responding" in s["error"]
    assert read_day_state(paths, TODAY).full_failures == 1


def test_break_stale_phase1_no_day_state_change(paths, make_lock, make_status, write_st, monkeypatch):
    _stale_full(paths, make_lock, make_status, write_st, monkeypatch, mode="phase1")
    assert break_stale_lock(paths, NOW) is True
    assert read_status(paths)["state"] == "failed"
    assert read_day_state(paths, TODAY) == DayState(TODAY)
    assert not paths.day_state_file(TODAY).exists()


def test_break_stale_no_status_removes_lock_without_failure(paths, make_lock):
    make_lock(now=NOW - timedelta(hours=1))
    assert break_stale_lock(paths, NOW) is True
    assert not paths.lock_file.exists()
    assert not paths.day_state_file(TODAY).exists()


def test_break_stale_status_other_run_id_untouched(paths, make_lock, make_status, write_st):
    make_lock(run_id="lockrun", now=NOW - timedelta(hours=1))
    write_st(make_status(run_id="otherrun", state="running", pid=os.getpid()))
    assert break_stale_lock(paths, NOW) is True
    s = read_status(paths)
    assert s["run_id"] == "otherrun" and s["state"] == "running"
    assert not paths.day_state_file(TODAY).exists()


def test_break_stale_terminal_status_not_rewritten(paths, make_lock, make_status, write_st, monkeypatch):
    monkeypatch.setattr(rs, "pid_alive", lambda pid: False)
    make_lock(now=NOW - timedelta(hours=1))
    write_st(make_status(state="done", finished_at=iso(NOW - S(500)), error=None))
    assert break_stale_lock(paths, NOW) is True
    assert read_status(paths)["state"] == "done"
    assert not paths.day_state_file(TODAY).exists()


def test_break_live_lock_changes_nothing(paths, make_lock, make_status, write_st):
    make_lock(now=NOW)
    write_st(make_status(pid=os.getpid(), heartbeat_at=iso(NOW)))
    before = paths.status_file.read_text(encoding="utf-8")
    assert break_stale_lock(paths, NOW) is False
    assert paths.lock_file.exists()
    assert paths.status_file.read_text(encoding="utf-8") == before
    assert not paths.day_state_file(TODAY).exists()


def test_break_no_lock_returns_false(paths):
    assert break_stale_lock(paths, NOW) is False


def test_break_fresh_breaker_blocks(paths, make_lock, make_status, write_st, monkeypatch):
    _stale_full(paths, make_lock, make_status, write_st, monkeypatch)
    paths.breaker_file.write_text("123", encoding="utf-8")
    assert break_stale_lock(paths, NOW) is False
    assert paths.lock_file.exists()
    assert read_status(paths)["state"] == "running"


def test_break_old_breaker_is_removed_and_proceeds(paths, make_lock, make_status, write_st, monkeypatch):
    _stale_full(paths, make_lock, make_status, write_st, monkeypatch)
    paths.breaker_file.write_text("123", encoding="utf-8")
    old = time.time() - BREAKER_STALE_S - 30
    os.utime(paths.breaker_file, (old, old))
    assert break_stale_lock(paths, NOW) is True
    assert not paths.lock_file.exists()
    assert read_day_state(paths, TODAY).full_failures == 1


def test_eight_threads_break_same_lock_counts_one_failure(paths, make_lock, make_status, write_st, monkeypatch):
    _stale_full(paths, make_lock, make_status, write_st, monkeypatch)
    barrier = threading.Barrier(8)
    results = []
    rl = threading.Lock()

    def w():
        barrier.wait()
        r = break_stale_lock(paths, NOW)
        with rl:
            results.append(r)

    ts = [threading.Thread(target=w) for _ in range(8)]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert results.count(True) == 1
    assert read_day_state(paths, TODAY).full_failures == 1


# ------------------------------------------------------------------ markers / day state


def test_done_markers(paths):
    assert has_done_marker(paths, "2026-10-11") is False
    write_done_markers(paths, ["2026-10-11", "2026-10-12", "2026-10-11"], NOW)
    assert has_done_marker(paths, "2026-10-11") and has_done_marker(paths, "2026-10-12")
    assert has_done_marker(paths, "2026-10-13") is False
    assert paths.done_marker("2026-10-11").read_text(encoding="utf-8") == iso(NOW) + "\n"
    names = sorted(p.name for p in paths.run_cache_dir.iterdir())
    assert names == [".done_rank_2026-10-11", ".done_rank_2026-10-12"]


def test_has_done_marker_default_day_is_local_today(paths):
    assert has_done_marker(paths) is False
    write_done_markers(paths, [local_today()])
    assert has_done_marker(paths) is True


def test_write_done_markers_never_raises(paths, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(type(paths.done_marker("x")), "write_text", boom)
    write_done_markers(paths, ["2026-10-11"])   # must not raise


def test_day_state_defaults_when_missing(paths):
    assert read_day_state(paths, "2026-10-11") == DayState("2026-10-11", 0, False, False)
    assert read_day_state(paths).day == local_today()


def test_day_state_recorders(paths):
    d = "2026-10-11"
    assert record_full_failure(paths, d).full_failures == 1
    assert record_full_failure(paths, d).full_failures == 2
    assert record_fallback_started(paths, d).fallback_started is True
    st = record_phase1_success(paths, d)
    assert st == DayState(d, 2, True, True)
    assert read_day_state(paths, d) == st
    assert read_day_state(paths, "2026-10-12") == DayState("2026-10-12")   # per-day isolation
    data = json.loads(paths.day_state_file(d).read_text(encoding="utf-8"))
    assert data["schema_version"] == 1 and data["day"] == d and "updated_at" in data


@pytest.mark.parametrize("payload", [
    {"schema_version": 1, "day": "d", "full_failures": -1},
    {"schema_version": 1, "day": "d", "full_failures": "2"},
    {"schema_version": 1, "day": "d", "full_failures": True},
    {"schema_version": 1, "day": "d", "full_failures": 1.5},
    {"schema_version": 1, "day": "d", "fallback_started": "yes"},
    {"schema_version": 1, "day": "d", "fallback_started": 1},
    {"schema_version": 1, "day": "d", "phase1_success": "true"},
    {"schema_version": 1, "day": "d", "phase1_success": None},
    {"schema_version": 7, "day": "d", "full_failures": 3},
    {"day": "d", "full_failures": 3},
])
def test_day_state_invalid_fields_give_defaults(paths, payload):
    d = "2026-10-11"
    paths.day_state_file(d).write_text(json.dumps(payload), encoding="utf-8")
    assert read_day_state(paths, d) == DayState(d)


@pytest.mark.parametrize("content", ["", "{bad", "[1]", "null"])
def test_day_state_corrupt_file_gives_defaults(paths, content):
    d = "2026-10-11"
    paths.day_state_file(d).write_text(content, encoding="utf-8")
    assert read_day_state(paths, d) == DayState(d)


def test_record_after_corrupt_file_starts_from_defaults(paths):
    d = "2026-10-11"
    paths.day_state_file(d).write_text("{bad", encoding="utf-8")
    assert record_full_failure(paths, d).full_failures == 1


# ------------------------------------------------------------------ cleanup


def test_cleanup_old_files(paths):
    today = "2026-10-11"
    for d in ("2026-10-01", "2026-10-03", "2026-10-04", "2026-10-10", "2026-10-11"):
        paths.done_marker(d).write_text("x", encoding="utf-8")
        paths.day_state_file(d).write_text("{}", encoding="utf-8")
    (paths.run_cache_dir / ".done_rank_garbage").write_text("x", encoding="utf-8")
    (paths.run_cache_dir / "auto_rank_nope.json").write_text("{}", encoding="utf-8")
    (paths.run_cache_dir / "phase1_result.json").write_text("{}", encoding="utf-8")
    base = time.time()
    for i in range(LOG_RETENTION_COUNT + 4):
        f = paths.log_file(f"r{i:02d}")
        f.write_text("x", encoding="utf-8")
        os.utime(f, (base + i, base + i))
    (paths.log_dir / "other.log").write_text("keep", encoding="utf-8")

    cleanup_old_files(paths, today)

    kept = {p.name for p in paths.run_cache_dir.iterdir()}
    assert ".done_rank_2026-10-01" not in kept and "auto_rank_2026-10-01.json" not in kept
    assert ".done_rank_2026-10-03" not in kept and "auto_rank_2026-10-03.json" not in kept
    for d in ("2026-10-04", "2026-10-10", "2026-10-11"):
        assert f".done_rank_{d}" in kept and f"auto_rank_{d}.json" in kept
    assert ".done_rank_garbage" in kept and "auto_rank_nope.json" in kept
    assert "phase1_result.json" in kept
    logs = sorted(p.name for p in paths.log_dir.glob("pipeline_job_*.log"))
    assert len(logs) == LOG_RETENTION_COUNT
    assert "pipeline_job_r00.log" not in logs and f"pipeline_job_r{LOG_RETENTION_COUNT + 3:02d}.log" in logs
    assert (paths.log_dir / "other.log").exists()


def test_cleanup_never_raises(tmp_path):
    ghost = RunPaths(tmp_path / "x", tmp_path / "x" / "rc", tmp_path / "l", tmp_path / "b",
                     tmp_path / "s", tmp_path / "x" / "logs")
    cleanup_old_files(ghost, "2026-10-11")
    cleanup_old_files(ghost, "not-a-date")


# ------------------------------------------------------------------ get_run_view


def _view(paths, now=NOW):
    return get_run_view(paths, now)


def test_view_idle_without_or_with_bad_status(paths, make_status, write_st):
    assert _view(paths).state == "idle"
    write_st(make_status(schema_version=5))
    assert _view(paths).state == "idle"


def test_view_running_alive(paths, make_lock, make_status, write_st):
    make_lock(run_id="run1", now=NOW)
    write_st(make_status(run_id="run1", pid=os.getpid(), step_index=4, step_total=9, heartbeat_at=iso(NOW),
                         label="Phase 2: x", trigger="auto", mode="full", phase1_reused=True))
    v = _view(paths)
    assert v.state == "running" and v.run_id == "run1" and v.mode == "full" and v.trigger == "auto"
    assert v.phase1_reused is True and v.label == "Phase 2: x"
    assert v.step_index == 4 and v.step_total == 9
    assert v.progress == pytest.approx(0.5)
    assert v.started_at == iso(NOW) and v.finished_at is None


def test_view_running_without_lock_is_idle(paths, make_status, write_st):
    write_st(make_status(pid=os.getpid()))
    assert _view(paths).state == "idle"


def test_view_running_other_lock_is_idle(paths, make_lock, make_status, write_st):
    make_lock(run_id="another", now=NOW)
    write_st(make_status(run_id="run1", pid=os.getpid()))
    assert _view(paths).state == "idle"


def test_view_running_dead_lock_is_failed_stopped_responding(paths, make_lock, make_status, write_st, monkeypatch):
    monkeypatch.setattr(rs, "pid_alive", lambda pid: False)
    make_lock(run_id="run1", now=NOW - timedelta(hours=1))
    write_st(make_status(run_id="run1", pid=999, heartbeat_at=iso(NOW - S(5)), step_index=2, step_total=9))
    v = _view(paths)
    assert v.state == "failed"
    assert v.label == "Run stopped responding"
    assert "stopped responding" in v.error
    assert v.progress < 1.0
    # view is read-only
    assert paths.lock_file.exists() and read_status(paths)["state"] == "running"


def test_view_done_window(paths, make_status, write_st):
    write_st(make_status(state="done", step_index=8, finished_at=iso(NOW - S(DONE_DISMISS_S - 1)), label="Ranking complete"))
    v = _view(paths)
    assert v.state == "done" and v.progress == 1.0
    write_st(make_status(state="done", finished_at=iso(NOW - S(DONE_DISMISS_S + 1))))
    assert _view(paths).state == "idle"


def test_view_failed_window(paths, make_status, write_st):
    write_st(make_status(state="failed", finished_at=iso(NOW - S(FAILED_DISMISS_S - 1)), error="boom"))
    v = _view(paths)
    assert v.state == "failed" and v.error == "boom"
    write_st(make_status(state="failed", finished_at=iso(NOW - S(FAILED_DISMISS_S + 1)), error="boom"))
    assert _view(paths).state == "idle"


def test_view_terminal_with_missing_finished_at_is_idle(paths, make_status, write_st):
    write_st(make_status(state="done", finished_at=None))
    assert _view(paths).state == "idle"


def test_view_junk_numeric_fields_default(paths, make_lock, make_status, write_st):
    make_lock(run_id="run1", now=NOW)
    write_st(make_status(pid=os.getpid(), step_index="3", step_total=None, label=None, error=5,
                        started_at=12, phase1_reused=None))
    v = _view(paths)
    assert v.state == "running"
    assert v.step_index == 0 and v.step_total == 0 and v.progress == 0.0
    assert v.label == "" and v.error is None and v.started_at is None and v.phase1_reused is False


# ------------------------------------------------------------------ log tail / cache version


def test_read_log_tail(paths):
    paths.log_file("run-1").write_text("\n".join(f"line{i}" for i in range(150)) + "\n", encoding="utf-8")
    tail = read_log_tail(paths, "run-1", 5)
    assert tail == [f"line{i}" for i in range(145, 150)]
    assert len(read_log_tail(paths, "run-1")) == 100
    assert read_log_tail(paths, "missing") == []
    assert read_log_tail(paths, "run-1", 0) == []


@pytest.mark.parametrize("bad", ["../secret", "..\\secret", "a/b", "a\\b", "", "x y", "run-1/../../etc",
                                 "run\x00", "..", ".", None, 5])
def test_read_log_tail_rejects_path_traversal(paths, bad):
    (paths.data_dir / "secret").write_text("TOP SECRET", encoding="utf-8")
    (paths.data_dir / "secret.log").write_text("TOP SECRET", encoding="utf-8")
    paths.log_file("legit").write_text("ok", encoding="utf-8")
    assert read_log_tail(paths, bad) == []


def test_ranking_cache_version(tmp_path):
    assert ranking_cache_version(tmp_path) == 0.0
    f = tmp_path / "ranking_cache.parquet"
    f.write_bytes(b"x")
    os.utime(f, (1234567.0, 1234567.0))
    assert ranking_cache_version(tmp_path) == 1234567.0
