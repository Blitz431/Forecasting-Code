"""Tests for src/pipeline_runner/launcher.py (plan section 11, items 26-39).

Spawning is always faked: no wrapper process is ever started.
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.pipeline_runner import launcher, run_state
from src.pipeline_runner.launcher import (
    DEFAULT_POOL,
    DEFAULT_TOP,
    build_wrapper_cmd,
    decide_auto_action,
    has_scraped_data,
    maybe_start_daily_run,
    reap_children,
    spawn_detached,
    start_pipeline_run,
)
from src.pipeline_runner.phase1_store import save_phase1_result
from src.pipeline_runner.run_state import (
    DayState,
    local_today,
    read_day_state,
    read_lock,
    read_status,
    record_full_failure,
    record_phase1_success,
    release_lock,
    try_acquire_lock,
    write_done_markers,
    write_status,
)
from tests.test_pipeline_runner.conftest import iso

TODAY = local_today()


@pytest.fixture(autouse=True)
def _deterministic(monkeypatch):
    """Only this process counts as alive; never keep real Popen objects around."""
    monkeypatch.setattr(run_state, "pid_alive", lambda pid: pid == os.getpid())
    monkeypatch.setattr(launcher, "_SPAWNED", [])


class FakeSpawn:
    def __init__(self, pid=4242, exc=None, hook=None):
        self.pid, self.exc, self.hook = pid, exc, hook
        self.calls: list[tuple[list[str], Path]] = []

    def __call__(self, argv, cwd):
        self.calls.append((list(argv), cwd))
        if self.hook:
            self.hook(argv)
        if self.exc:
            raise self.exc
        return self.pid

    @property
    def argv(self):
        return self.calls[-1][0]


def _arg(argv, name):
    return argv[argv.index(name) + 1]


# ------------------------------------------------------------------ decide_auto_action


def _expected(enabled, marker, alive, has_data, ff, fb, p1):
    if not enabled:
        return "disabled"
    if marker:
        return "already_done"
    if alive:
        return "running"
    if not has_data:
        return "no_data"
    if ff == 0:
        return "start_full"
    if not fb and not p1:
        return "start_fallback"
    return "buttons_only"


@pytest.mark.parametrize("enabled,marker,alive,has_data,ff,fb,p1", list(itertools.product(
    [True, False], [True, False], [True, False], [True, False], [0, 1, 2], [True, False], [True, False])))
def test_decide_auto_action_truth_table(enabled, marker, alive, has_data, ff, fb, p1):
    day = DayState("d", ff, fb, p1)
    assert decide_auto_action(enabled, marker, alive, day, has_data) == _expected(
        enabled, marker, alive, has_data, ff, fb, p1)


def test_decide_auto_action_named_rows():
    d0 = DayState("d")
    assert decide_auto_action(True, False, False, d0, True) == "start_full"
    assert decide_auto_action(True, False, False, DayState("d", 1), True) == "start_fallback"
    assert decide_auto_action(True, False, False, DayState("d", 1, True), True) == "buttons_only"
    assert decide_auto_action(True, False, False, DayState("d", 1, False, True), True) == "buttons_only"
    assert decide_auto_action(True, False, False, DayState("d", 2), True) == "start_fallback"


# ------------------------------------------------------------------ helpers


def test_has_scraped_data(tmp_path):
    assert has_scraped_data(tmp_path / "missing") is False
    d = tmp_path / "d"
    d.mkdir()
    assert has_scraped_data(d) is False
    (d / "x.txt").write_text("a")
    assert has_scraped_data(d) is False
    (d / "A.parquet").write_bytes(b"x")
    assert has_scraped_data(d) is True


def test_build_wrapper_cmd(tmp_path):
    cmd = build_wrapper_cmd("rid", "2026-10-11", "full", 100, 50, "auto", False, root_dir=tmp_path)
    assert cmd[0] == sys.executable
    assert cmd[1] == str(tmp_path / "cli" / "pipeline_job.py")
    assert cmd[2:] == ["--run-id", "rid", "--run-date", "2026-10-11", "--mode", "full",
                       "--pool", "100", "--top", "50", "--trigger", "auto"]
    assert all(isinstance(c, str) for c in cmd)
    cmd2 = build_wrapper_cmd("rid", "2026-10-11", "full", 100, 50, "button", True, root_dir=tmp_path)
    assert cmd2[-1] == "--reuse-phase1"


def test_reap_children_drops_finished_and_never_raises(monkeypatch):
    class P:
        def __init__(self, rc):
            self.rc = rc

        def poll(self):
            if self.rc == "boom":
                raise RuntimeError("x")
            return self.rc

    live, done = P(None), P(0)
    monkeypatch.setattr(launcher, "_SPAWNED", [live, done, P("boom")])
    reap_children()
    assert launcher._SPAWNED == [live]


# ------------------------------------------------------------------ maybe_start_daily_run


def test_fresh_day_starts_full(paths, settings):
    sp = FakeSpawn()
    res = maybe_start_daily_run(paths=paths, settings=settings, spawn=sp)
    assert res.started and res.reason == "started" and res.mode == "full"
    assert res.phase1_reused is False
    assert len(sp.calls) == 1
    argv = sp.argv
    assert _arg(argv, "--mode") == "full" and _arg(argv, "--pool") == "100" and _arg(argv, "--top") == "50"
    assert _arg(argv, "--trigger") == "auto" and _arg(argv, "--run-date") == TODAY
    assert "--reuse-phase1" not in argv
    assert argv[0] == sys.executable
    s = read_status(paths)
    assert s["pid"] == 4242 and s["state"] == "running" and s["run_id"] == res.run_id
    assert s["mode"] == "full" and s["trigger"] == "auto" and s["step_total"] == 9 and s["step_index"] == 0
    assert s["phase1_reused"] is False
    assert read_lock(paths).run_id == res.run_id
    assert s["log_path"] == f"data/logs/pipeline_job_{res.run_id}.log"


def test_done_marker_means_already_done(paths, settings):
    write_done_markers(paths, [TODAY])
    sp = FakeSpawn()
    res = maybe_start_daily_run(paths=paths, settings=settings, spawn=sp)
    assert (res.started, res.reason) == (False, "already_done")
    assert sp.calls == [] and read_lock(paths) is None


def test_disabled_does_not_spawn(paths, settings):
    settings.auto_rank_enabled = False
    sp = FakeSpawn()
    res = maybe_start_daily_run(paths=paths, settings=settings, spawn=sp)
    assert (res.started, res.reason) == (False, "disabled")
    assert sp.calls == [] and read_lock(paths) is None and read_status(paths) is None


def test_disabled_wins_over_marker_and_data(paths, settings):
    settings.auto_rank_enabled = False
    write_done_markers(paths, [TODAY])
    assert maybe_start_daily_run(paths=paths, settings=settings, spawn=FakeSpawn()).reason == "disabled"


def test_no_data(paths, settings):
    (settings.raw_daily_dir / "AAA.parquet").unlink()
    sp = FakeSpawn()
    res = maybe_start_daily_run(paths=paths, settings=settings, spawn=sp)
    assert (res.started, res.reason) == (False, "no_data")
    assert sp.calls == [] and read_lock(paths) is None


def test_get_settings_raising_returns_disabled(paths, monkeypatch):
    def boom():
        raise RuntimeError("bad .env")
    monkeypatch.setattr(launcher, "get_settings", boom)
    sp = FakeSpawn()
    res = maybe_start_daily_run(paths=paths, settings=None, spawn=sp)
    assert (res.started, res.reason) == (False, "disabled")
    assert sp.calls == []


@pytest.mark.parametrize("exc", [ValueError("x"), OSError("y"), KeyError("z")])
def test_get_settings_any_exception_does_not_raise(paths, monkeypatch, exc):
    def boom():
        raise exc
    monkeypatch.setattr(launcher, "get_settings", boom)
    assert maybe_start_daily_run(paths=paths, spawn=FakeSpawn()).reason == "disabled"


def test_auto_rank_disabled_from_get_settings(paths, settings, monkeypatch):
    settings.auto_rank_enabled = False
    monkeypatch.setattr(launcher, "get_settings", lambda: settings)
    sp = FakeSpawn()
    res = maybe_start_daily_run(paths=paths, spawn=sp)
    assert res.reason == "disabled" and sp.calls == []


def test_fallback_after_one_full_failure(paths, settings):
    record_full_failure(paths, TODAY)
    sp = FakeSpawn()
    res = maybe_start_daily_run(paths=paths, settings=settings, spawn=sp)
    assert res.started and res.mode == "phase1"
    assert _arg(sp.argv, "--trigger") == "auto_fallback" and _arg(sp.argv, "--mode") == "phase1"
    assert read_status(paths)["trigger"] == "auto_fallback"
    assert read_status(paths)["step_total"] == 5
    day = read_day_state(paths, TODAY)
    assert day.fallback_started is True and day.full_failures == 1
    # lock released by the (fake) wrapper, then a second call is buttons only
    assert release_lock(paths, res.run_id)
    sp2 = FakeSpawn()
    res2 = maybe_start_daily_run(paths=paths, settings=settings, spawn=sp2)
    assert (res2.started, res2.reason) == (False, "buttons_only")
    assert sp2.calls == []


def test_full_failure_with_phase1_success_is_buttons_only(paths, settings):
    record_full_failure(paths, TODAY)
    record_phase1_success(paths, TODAY)
    sp = FakeSpawn()
    res = maybe_start_daily_run(paths=paths, settings=settings, spawn=sp)
    assert (res.started, res.reason) == (False, "buttons_only")
    assert sp.calls == []
    assert read_day_state(paths, TODAY).fallback_started is False


def test_live_lock_means_running(paths, settings, make_lock, make_status, write_st):
    now = datetime.now(timezone.utc)
    make_lock(run_id="live1", now=now)
    write_st(make_status(run_id="live1", pid=os.getpid(), heartbeat_at=iso(now)))
    sp = FakeSpawn()
    res = maybe_start_daily_run(paths=paths, settings=settings, spawn=sp)
    assert (res.started, res.reason, res.run_id) == (False, "running", "live1")
    assert sp.calls == [] and read_lock(paths).run_id == "live1"
    assert read_day_state(paths, TODAY) == DayState(TODAY)


def test_stale_full_lock_is_broken_then_fallback_starts(paths, settings, make_lock, make_status, write_st):
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    make_lock(run_id="dead1", mode="full", now=old)
    write_st(make_status(run_id="dead1", mode="full", pid=424242, heartbeat_at=iso(old)))
    sp = FakeSpawn()
    res = maybe_start_daily_run(paths=paths, settings=settings, spawn=sp)
    day = read_day_state(paths, TODAY)
    assert day.full_failures == 1 and day.fallback_started is True
    assert res.started and res.mode == "phase1"
    assert _arg(sp.argv, "--trigger") == "auto_fallback"
    assert read_lock(paths).run_id == res.run_id != "dead1"


def test_revalidation_under_lock_blocks_duplicate_fallback(paths, settings, monkeypatch):
    record_full_failure(paths, TODAY)
    real = launcher.try_acquire_lock

    def racing(p, *a, **k):
        ok = real(p, *a, **k)
        if ok:   # another tab claims the fallback right after we grab the lock
            launcher.record_fallback_started(p, TODAY)
        return ok

    monkeypatch.setattr(launcher, "try_acquire_lock", racing)
    sp = FakeSpawn()
    res = maybe_start_daily_run(paths=paths, settings=settings, spawn=sp)
    assert res.started is False and res.reason == "buttons_only"
    assert sp.calls == []
    assert read_lock(paths) is None


def test_revalidation_under_lock_blocks_when_marker_appears(paths, settings, monkeypatch):
    real = launcher.try_acquire_lock

    def racing(p, *a, **k):
        ok = real(p, *a, **k)
        if ok:
            write_done_markers(p, [TODAY])
        return ok

    monkeypatch.setattr(launcher, "try_acquire_lock", racing)
    sp = FakeSpawn()
    res = maybe_start_daily_run(paths=paths, settings=settings, spawn=sp)
    assert (res.started, res.reason) == (False, "already_done")
    assert sp.calls == [] and read_lock(paths) is None


def test_revalidation_failure_moved_on_maps_to_running(paths, settings, monkeypatch):
    real = launcher.try_acquire_lock

    def racing(p, *a, **k):
        ok = real(p, *a, **k)
        if ok:   # a full run already failed in another tab: start_full is no longer right
            record_full_failure(p, TODAY)
        return ok

    monkeypatch.setattr(launcher, "try_acquire_lock", racing)
    res = maybe_start_daily_run(paths=paths, settings=settings, spawn=FakeSpawn())
    assert res.started is False and res.reason == "running"
    assert read_lock(paths) is None


# ------------------------------------------------------------------ start_pipeline_run (button)


def _artifact(settings, day=None):
    entries = [types.SimpleNamespace(ticker=t, rank=i + 1, composite_score=0.5 - i * 0.01)
               for i, t in enumerate(["A", "B", "C"])]
    save_phase1_result(entries, settings, 3, now=datetime.now(timezone.utc) - timedelta(hours=1), day=day)


def test_button_full_reuses_usable_artifact(paths, settings):
    _artifact(settings)
    sp = FakeSpawn()
    res = start_pipeline_run("full", 100, 50, trigger="button", paths=paths, settings=settings, spawn=sp)
    assert res.started and res.phase1_reused is True
    assert "--reuse-phase1" in sp.argv and _arg(sp.argv, "--trigger") == "button"
    s = read_status(paths)
    assert s["phase1_reused"] is True and s["step_total"] == 8
    assert "ML only" in res.message or "reusing" in res.message.lower()


def test_button_full_with_stale_artifact_is_normal_run(paths, settings):
    _artifact(settings, day="2000-01-01")
    sp = FakeSpawn()
    res = start_pipeline_run("full", 100, 50, trigger="button", paths=paths, settings=settings, spawn=sp)
    assert res.started and res.phase1_reused is False
    assert "--reuse-phase1" not in sp.argv
    s = read_status(paths)
    assert s["phase1_reused"] is False and s["step_total"] == 9


def test_phase1_button_never_reuses(paths, settings):
    _artifact(settings)
    sp = FakeSpawn()
    res = start_pipeline_run("phase1", 100, 50, trigger="button", paths=paths, settings=settings, spawn=sp)
    assert res.started and res.phase1_reused is False
    assert "--reuse-phase1" not in sp.argv
    assert read_status(paths)["step_total"] == 5
    assert res.message == "Fast rank started in the background."


def test_button_ignores_marker_and_failure_cap(paths, settings):
    write_done_markers(paths, [TODAY])
    record_full_failure(paths, TODAY)
    record_full_failure(paths, TODAY)
    sp = FakeSpawn()
    res = start_pipeline_run("full", 100, 50, trigger="button", paths=paths, settings=settings, spawn=sp)
    assert res.started and len(sp.calls) == 1


def test_button_refused_on_live_lock(paths, settings, make_lock, make_status, write_st):
    now = datetime.now(timezone.utc)
    make_lock(run_id="live1", now=now)
    write_st(make_status(run_id="live1", pid=os.getpid(), heartbeat_at=iso(now)))
    sp = FakeSpawn()
    res = start_pipeline_run("phase1", 100, 50, trigger="button", paths=paths, settings=settings, spawn=sp)
    assert (res.started, res.reason) == (False, "running")
    assert res.message == "A ranking run is already in progress."
    assert sp.calls == [] and read_lock(paths).run_id == "live1"


def test_button_refused_on_no_data(paths, settings):
    (settings.raw_daily_dir / "AAA.parquet").unlink()
    sp = FakeSpawn()
    res = start_pipeline_run("full", 100, 50, trigger="button", paths=paths, settings=settings, spawn=sp)
    assert (res.started, res.reason) == (False, "no_data")
    assert sp.calls == [] and read_lock(paths) is None


def test_button_breaks_stale_lock_and_starts(paths, settings, make_lock, make_status, write_st):
    old = datetime.now(timezone.utc) - timedelta(hours=1)
    make_lock(run_id="dead1", mode="phase1", now=old)
    write_st(make_status(run_id="dead1", mode="phase1", pid=424242, heartbeat_at=iso(old)))
    res = start_pipeline_run("phase1", 100, 50, trigger="button", paths=paths, settings=settings,
                             spawn=FakeSpawn())
    assert res.started and read_lock(paths).run_id == res.run_id


def test_defaults_are_app_defaults():
    assert (DEFAULT_POOL, DEFAULT_TOP) == (100, 50)


# ------------------------------------------------------------------ spawn errors / pid fill


def test_spawn_error_full(paths, settings):
    sp = FakeSpawn(exc=OSError("no exec"))
    res = start_pipeline_run("full", 100, 50, trigger="button", paths=paths, settings=settings, spawn=sp)
    assert (res.started, res.reason) == (False, "spawn_error")
    assert "no exec" in res.message
    assert read_lock(paths) is None
    s = read_status(paths)
    assert s["state"] == "failed" and "Could not start background run" in s["error"]
    assert s["finished_at"]
    assert read_day_state(paths, TODAY).full_failures == 1


def test_spawn_error_fallback_keeps_claim_and_does_not_count_failure(paths, settings):
    record_full_failure(paths, TODAY)
    sp = FakeSpawn(exc=OSError("no exec"))
    res = maybe_start_daily_run(paths=paths, settings=settings, spawn=sp)
    assert (res.started, res.reason) == (False, "spawn_error")
    day = read_day_state(paths, TODAY)
    assert day.fallback_started is True and day.full_failures == 1
    assert read_lock(paths) is None and read_status(paths)["state"] == "failed"
    # nothing more happens automatically
    assert maybe_start_daily_run(paths=paths, settings=settings, spawn=FakeSpawn()).reason == "buttons_only"


def test_spawn_error_phase1_button_no_day_state(paths, settings):
    res = start_pipeline_run("phase1", 100, 50, trigger="button", paths=paths, settings=settings,
                             spawn=FakeSpawn(exc=OSError("x")))
    assert res.reason == "spawn_error"
    assert not paths.day_state_file(TODAY).exists()


def test_pid_fill_does_not_overwrite_wrapper_record(paths, settings):
    def wrapper_wrote(argv):
        rid = _arg(argv, "--run-id")
        s = read_status(paths)
        write_status(paths, {**s, "pid": 999, "run_id": rid})

    sp = FakeSpawn(pid=4242, hook=wrapper_wrote)
    res = start_pipeline_run("phase1", 100, 50, trigger="button", paths=paths, settings=settings, spawn=sp)
    assert res.started
    assert read_status(paths)["pid"] == 999


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize("kw", [
    {"mode": "bogus"}, {"trigger": "bogus"},
    {"pool": 9}, {"pool": 504}, {"pool": True}, {"pool": "100"}, {"pool": 100.0},
    {"top": 4}, {"top": 201}, {"top": False}, {"top": "50"},
])
def test_invalid_arguments_raise_before_side_effects(tmp_path, kw):
    data = tmp_path / "fresh" / "data"
    st = types.SimpleNamespace(data_dir=data, raw_daily_dir=data / "raw" / "daily",
                               ticker_source="sp500", auto_rank_enabled=True)
    args = {"mode": "full", "pool": 100, "top": 50, "trigger": "button"}
    args.update(kw)
    sp = FakeSpawn()
    with pytest.raises(ValueError):
        start_pipeline_run(args["mode"], args["pool"], args["top"], trigger=args["trigger"],
                           settings=st, spawn=sp)
    assert sp.calls == []
    assert not (tmp_path / "fresh").exists()


@pytest.mark.parametrize("pool,top", [(10, 5), (503, 200)])
def test_boundary_values_accepted(paths, settings, pool, top):
    res = start_pipeline_run("phase1", pool, top, trigger="button", paths=paths, settings=settings,
                             spawn=FakeSpawn())
    assert res.started
    assert read_status(paths)["pool"] == pool and read_status(paths)["top"] == top


# ------------------------------------------------------------------ spawn_detached flags


class _FakePopen:
    calls: list = []
    fail_first_with = None

    def __init__(self, argv, **kwargs):
        type(self).calls.append((argv, kwargs))
        if type(self).fail_first_with is not None and len(type(self).calls) == 1:
            raise type(self).fail_first_with
        self.pid = 31337

    def poll(self):
        return 0


@pytest.fixture
def fake_popen(monkeypatch):
    _FakePopen.calls = []
    _FakePopen.fail_first_with = None
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    return _FakePopen


def test_spawn_detached_posix(fake_popen, monkeypatch, tmp_path):
    monkeypatch.setattr(launcher.sys, "platform", "linux")
    argv = [sys.executable, "-c", "pass"]
    pid = spawn_detached(argv, tmp_path)
    assert pid == 31337
    (got_argv, kw), = fake_popen.calls
    assert got_argv == argv and got_argv[0] == sys.executable
    assert kw["start_new_session"] is True
    assert kw["stdin"] == subprocess.DEVNULL and kw["stdout"] == subprocess.DEVNULL
    assert kw["stderr"] == subprocess.DEVNULL
    assert kw["close_fds"] is True and kw["cwd"] == str(tmp_path)
    assert not kw.get("shell")
    assert "creationflags" not in kw
    assert kw["env"]["PYTHONIOENCODING"] == "utf-8" and kw["env"]["PYTHONUTF8"] == "1"
    assert len(launcher._SPAWNED) == 1


def _win_consts(monkeypatch):
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000, raising=False)
    monkeypatch.setattr(subprocess, "DETACHED_PROCESS", 0x8, raising=False)
    monkeypatch.setattr(launcher.sys, "platform", "win32")


def test_spawn_detached_windows_flags(fake_popen, monkeypatch, tmp_path):
    _win_consts(monkeypatch)
    spawn_detached([sys.executable, "x.py"], tmp_path)
    (argv, kw), = fake_popen.calls
    flags = kw["creationflags"]
    assert flags == 0x200 | 0x08000000 | 0x01000000
    assert not flags & 0x8                      # never DETACHED_PROCESS
    assert "start_new_session" not in kw
    assert not kw.get("shell") and argv[0] == sys.executable


def test_spawn_detached_windows_retries_without_breakaway_on_access_denied(fake_popen, monkeypatch, tmp_path):
    _win_consts(monkeypatch)
    err = OSError("denied")
    err.winerror = 5
    fake_popen.fail_first_with = err
    pid = spawn_detached([sys.executable, "x.py"], tmp_path)
    assert pid == 31337
    assert len(fake_popen.calls) == 2
    first, second = fake_popen.calls[0][1]["creationflags"], fake_popen.calls[1][1]["creationflags"]
    assert first & 0x01000000 and not second & 0x01000000
    assert second == 0x200 | 0x08000000 and not second & 0x8


def test_spawn_detached_windows_other_oserror_is_raised(fake_popen, monkeypatch, tmp_path):
    _win_consts(monkeypatch)
    err = OSError("other")
    err.winerror = 2
    fake_popen.fail_first_with = err
    with pytest.raises(OSError):
        spawn_detached([sys.executable, "x.py"], tmp_path)
    assert len(fake_popen.calls) == 1
