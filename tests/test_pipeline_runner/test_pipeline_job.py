"""Tests for cli/pipeline_job.py (plan section 11, items 40-50).

The wrapper runs in-process (run_job) and spawns a tiny fake child script instead of the real
cli/pipeline.py. Everything lives under tmp_path.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import textwrap
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from cli import pipeline_job as pj
from src.pipeline_runner.steps import PHASE1_FALLBACK_MARKER, PHASE1_REUSE_MARKER
from src.pipeline_runner.run_state import (
    get_run_paths,
    has_done_marker,
    local_today,
    read_day_state,
    read_lock,
    read_status,
    try_acquire_lock,
)

RUN_ID = "20261011T150000Z-abc123"
RUN_DATE = "2026-10-11"

L_FAST = "[Phase 1] Fast-ranking 500 tickers (no ML) ..."
L_P1DONE = "[Phase 1] Done in 12s -- pool: top 100 candidates"
L_TRAIN = "[Phase 2] Training ML models on 100 candidates ..."
L_MLDONE = "[Phase 2] ML training done in 40s"
L_RERANK = "[Phase 2] Re-ranking top 100 with ML signal ..."
L_RERANKDONE = "[Phase 2] Re-rank done in 3s"
L_OUT = "[Output] Collecting price targets …"
L_CACHE = "[Cache] Ranking cache saved -> x"
L_TABLE = "  AutoStockAnalyzer -- Two-Phase Pipeline Results"
L_P1ONLY = "[Phase 1 Only] Collecting price targets ..."
L_REUSE = f"[Phase 1] {PHASE1_REUSE_MARKER} from 2026-10-11T15:00:00+00:00 (498 ranked) -- skipping Phase 1, pool: top 100 candidates"
L_FALLBACK = f"[Phase 1] {PHASE1_FALLBACK_MARKER}: price data updated after Phase 1 ran -- running Phase 1"

FULL_LINES = [L_FAST, L_P1DONE, L_TRAIN, L_MLDONE, L_RERANK, L_RERANKDONE, L_OUT, L_CACHE, L_TABLE]
PHASE1_LINES = [L_FAST, L_P1DONE, L_P1ONLY, L_CACHE, L_TABLE]
REUSE_LINES = [L_REUSE, L_TRAIN, L_MLDONE, L_RERANK, L_RERANKDONE, L_OUT, L_CACHE, L_TABLE]


@pytest.fixture(autouse=True)
def _isolate_process_state(monkeypatch):
    """The wrapper lowers niceness and installs a SIGTERM handler; keep the test process untouched."""
    monkeypatch.setattr(pj.os, "nice", lambda n: 0, raising=False)
    old = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, old)


@pytest.fixture
def env(tmp_path):
    data_dir = tmp_path / "data"
    paths = get_run_paths(data_dir)
    assert try_acquire_lock(paths, RUN_ID, RUN_DATE, "full", "auto")
    return type("Env", (), {"paths": paths, "data_dir": data_dir, "tmp": tmp_path})


def make_fake(tmp_path: Path, steps, exit_code=0, name="fake_pipeline.py"):
    """Write a fake pipeline script.

    steps: list of ("print", text) | ("sleep", seconds) | ("code", python source)
    It records its argv to <tmp>/child_argv.json.
    """
    body = [
        "import json, sys, time, os",
        f"open({str(tmp_path / 'child_argv.json')!r}, 'w').write(json.dumps(sys.argv[1:]))",
    ]
    for kind, val in steps:
        if kind == "print":
            body.append(f"print({val!r}, flush=True)")
        elif kind == "sleep":
            body.append(f"time.sleep({val!r})")
        else:
            body.append(textwrap.dedent(val))
    body.append(f"sys.exit({exit_code})")
    script = tmp_path / name
    script.write_text("\n".join(body) + "\n", encoding="utf-8")
    return script


def prints(lines):
    return [("print", ln) for ln in lines]


def job_args(env, script, mode="full", reuse=False, run_id=RUN_ID, trigger="auto", extra=None):
    argv = ["--run-id", run_id, "--run-date", RUN_DATE, "--mode", mode, "--pool", "100", "--top", "50",
            "--trigger", trigger, "--data-dir", str(env.data_dir), "--pipeline-script", str(script),
            "--heartbeat-s", "0.5"]
    if reuse:
        argv.append("--reuse-phase1")
    argv += extra or []
    return pj.parse_args(argv)


def child_argv(env):
    return json.loads((env.tmp / "child_argv.json").read_text(encoding="utf-8"))


def dt(s):
    return datetime.fromisoformat(s)


class Poller(threading.Thread):
    """Samples the status file while the job runs."""

    def __init__(self, paths):
        super().__init__(daemon=True)
        self.paths, self.samples, self.stop = paths, [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            s = read_status(self.paths)
            if s:
                self.samples.append(s)
            time.sleep(0.05)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.join(5)


# ------------------------------------------------------------------ pure helpers


def test_pipeline_argv_shapes(tmp_path):
    s = tmp_path / "p.py"
    base = [sys.executable, "-u", str(s), "--pool", "100", "--top", "50"]
    assert pj.pipeline_argv("phase1", 100, 50, s, False) == base + ["--phase1-only"]
    assert pj.pipeline_argv("phase1", 100, 50, s, True) == base + ["--phase1-only"]
    assert pj.pipeline_argv("full", 100, 50, s, False) == base + ["--no-deep"]
    assert pj.pipeline_argv("full", 100, 50, s, True) == base + ["--no-deep", "--skip-phase1"]


@pytest.mark.parametrize("argv", [
    [],
    ["--run-id", "x", "--run-date", "not-a-date", "--mode", "full", "--pool", "1", "--top", "1", "--trigger", "auto"],
    ["--run-id", "x", "--run-date", "2026-10-11", "--mode", "weird", "--pool", "1", "--top", "1", "--trigger", "auto"],
    ["--run-id", "x", "--run-date", "2026-10-11", "--mode", "full", "--pool", "1", "--top", "1", "--trigger", "nope"],
    ["--run-id", "x", "--run-date", "2026-10-11", "--mode", "phase1", "--pool", "1", "--top", "1",
     "--trigger", "auto", "--reuse-phase1"],
])
def test_parse_args_rejects_bad_input(argv):
    with pytest.raises(SystemExit) as e:
        pj.parse_args(argv)
    assert e.value.code == 2


# ------------------------------------------------------------------ success paths


def test_full_success(env):
    script = make_fake(env.tmp, prints(FULL_LINES[:-1]) + [("sleep", 1.2), ("print", L_TABLE)])
    rc = pj.run_job(job_args(env, script))
    assert rc == 0
    s = read_status(env.paths)
    assert s["state"] == "done" and s["step_index"] == 8 and s["step_total"] == 9
    assert s["label"] == "Ranking complete" and s["returncode"] == 0 and s["error"] is None
    assert s["pid"] == os.getpid() and isinstance(s["child_pid"], int) and s["child_pid"] > 0
    assert s["finished_at"]
    assert s["phase1_reused"] is False
    assert dt(s["step_started_at"]) > dt(s["started_at"])     # step_started_at moved on between steps
    assert has_done_marker(env.paths, RUN_DATE) and has_done_marker(env.paths, local_today())
    assert read_lock(env.paths) is None
    log = env.paths.log_file(RUN_ID).read_text(encoding="utf-8")
    assert "…" in log and L_TABLE.strip() in log and RUN_ID in log.splitlines()[0]
    assert "--no-deep" in child_argv(env) and "--skip-phase1" not in child_argv(env)


def test_phase1_success(env):
    # re-acquire the lock as a phase1 run
    env.paths.lock_file.unlink()
    assert try_acquire_lock(env.paths, RUN_ID, RUN_DATE, "phase1", "button")
    script = make_fake(env.tmp, prints(PHASE1_LINES))
    rc = pj.run_job(job_args(env, script, mode="phase1", trigger="button"))
    assert rc == 0
    s = read_status(env.paths)
    assert s["state"] == "done" and s["step_total"] == 5 and s["step_index"] == 4
    assert not has_done_marker(env.paths, RUN_DATE) and not has_done_marker(env.paths, local_today())
    assert read_day_state(env.paths, RUN_DATE).phase1_success is True
    assert read_day_state(env.paths, RUN_DATE).full_failures == 0
    assert read_lock(env.paths) is None
    assert "--phase1-only" in child_argv(env) and "--no-deep" not in child_argv(env)


def test_full_reuse_phase2_only(env):
    script = make_fake(env.tmp, prints(REUSE_LINES))
    rc = pj.run_job(job_args(env, script, reuse=True))
    assert rc == 0
    s = read_status(env.paths)
    assert s["state"] == "done" and s["step_total"] == 8 and s["step_index"] == 7
    assert s["phase1_reused"] is True
    assert "--skip-phase1" in child_argv(env) and "--no-deep" in child_argv(env)
    assert has_done_marker(env.paths, RUN_DATE)


def test_reuse_fallback_switches_to_nine_steps(env):
    steps = [("print", L_FALLBACK), ("sleep", 1.5)] + prints([L_FAST, L_P1DONE] + FULL_LINES[2:])
    script = make_fake(env.tmp, steps)
    with Poller(env.paths) as poll:
        rc = pj.run_job(job_args(env, script, reuse=True))
    assert rc == 0
    s = read_status(env.paths)
    assert s["state"] == "done" and s["step_total"] == 9 and s["step_index"] == 8
    assert s["phase1_reused"] is False
    # while running, the bar switched to the 9 step table at index 0 (before Phase 1 progressed)
    switched = [x for x in poll.samples
                if x["state"] == "running" and x["step_total"] == 9 and x["phase1_reused"] is False]
    assert switched, "never observed the 9 step table"
    assert switched[0]["step_index"] == 0
    assert has_done_marker(env.paths, RUN_DATE)


# ------------------------------------------------------------------ failure paths


def test_full_failure(env):
    script = make_fake(env.tmp, prints([L_FAST, L_P1DONE, "Traceback boom: kaput"]), exit_code=1)
    rc = pj.run_job(job_args(env, script))
    assert rc == 1
    s = read_status(env.paths)
    assert s["state"] == "failed" and s["returncode"] == 1 and s["finished_at"]
    assert "kaput" in s["error"] and "exited with code 1" in s["error"]
    assert len(s["error"]) <= 1000
    assert read_day_state(env.paths, RUN_DATE).full_failures == 1
    assert not has_done_marker(env.paths, RUN_DATE) and not has_done_marker(env.paths, local_today())
    assert read_lock(env.paths) is None


def test_failure_error_is_capped_and_keeps_tail(env):
    lines = [f"noise line {i} " + "x" * 80 for i in range(60)] + ["LAST-LINE-MARKER"]
    script = make_fake(env.tmp, prints(lines), exit_code=3)
    pj.run_job(job_args(env, script))
    err = read_status(env.paths)["error"]
    assert len(err) <= 1000 and "LAST-LINE-MARKER" in err and "noise line 0 " not in err


def test_phase1_failure_changes_no_day_state(env):
    env.paths.lock_file.unlink()
    assert try_acquire_lock(env.paths, RUN_ID, RUN_DATE, "phase1", "button")
    script = make_fake(env.tmp, prints([L_FAST, "oops"]), exit_code=2)
    rc = pj.run_job(job_args(env, script, mode="phase1", trigger="button"))
    assert rc == 1
    assert read_status(env.paths)["state"] == "failed"
    assert not env.paths.day_state_file(RUN_DATE).exists()
    assert not has_done_marker(env.paths, RUN_DATE)
    assert read_lock(env.paths) is None


def test_timeout(env):
    script = make_fake(env.tmp, prints([L_FAST]) + [("sleep", 60)])
    t0 = time.monotonic()
    rc = pj.run_job(job_args(env, script, extra=["--max-runtime-s", "2"]))
    assert time.monotonic() - t0 < 30
    assert rc == 1
    s = read_status(env.paths)
    assert s["state"] == "failed" and "Timed out after 2 s" in s["error"]
    assert read_day_state(env.paths, RUN_DATE).full_failures == 1
    assert not has_done_marker(env.paths, RUN_DATE)
    assert read_lock(env.paths) is None


def test_wrapper_exception_best_effort_failure(env, monkeypatch):
    script = make_fake(env.tmp, prints([L_FAST]))
    monkeypatch.setattr(pj._Job, "_spawn_child", lambda self: (_ for _ in ()).throw(RuntimeError("no spawn")))
    rc = pj.run_job(job_args(env, script))
    assert rc == 1
    s = read_status(env.paths)
    assert s["state"] == "failed" and "Wrapper error: RuntimeError: no spawn" in s["error"]
    assert read_lock(env.paths) is None


# ------------------------------------------------------------------ ordering


def test_marker_then_status_then_lock_release(env, monkeypatch):
    events = []
    real_release, real_markers, real_write = pj.release_lock, pj.write_done_markers, pj.write_status

    def markers(paths, days, *a, **k):
        events.append("marker")
        return real_markers(paths, days, *a, **k)

    def write(paths, status):
        real_write(paths, status)
        if status.get("state") == "done":
            events.append("status_done")

    def release(paths, run_id):
        # at this point the marker and the terminal status must already be on disk
        assert has_done_marker(paths, RUN_DATE)
        assert read_status(paths)["state"] == "done"
        events.append("release")
        return real_release(paths, run_id)

    monkeypatch.setattr(pj, "write_done_markers", markers)
    monkeypatch.setattr(pj, "write_status", write)
    monkeypatch.setattr(pj, "release_lock", release)
    script = make_fake(env.tmp, prints(FULL_LINES))
    assert pj.run_job(job_args(env, script)) == 0
    assert events == ["marker", "status_done", "release"]


def test_failure_day_state_then_status_then_lock_release(env, monkeypatch):
    events = []
    real_release, real_fail, real_write = pj.release_lock, pj.record_full_failure, pj.write_status

    def fail(paths, day):
        events.append("day_state")
        return real_fail(paths, day)

    def write(paths, status):
        real_write(paths, status)
        if status.get("state") == "failed":
            events.append("status_failed")

    def release(paths, run_id):
        assert read_day_state(paths, RUN_DATE).full_failures == 1
        assert read_status(paths)["state"] == "failed"
        events.append("release")
        return real_release(paths, run_id)

    monkeypatch.setattr(pj, "record_full_failure", fail)
    monkeypatch.setattr(pj, "write_status", write)
    monkeypatch.setattr(pj, "release_lock", release)
    script = make_fake(env.tmp, prints([L_FAST]), exit_code=1)
    assert pj.run_job(job_args(env, script)) == 1
    assert events == ["day_state", "status_failed", "release"]


# ------------------------------------------------------------------ ownership


def test_not_owner_exits_2_and_writes_nothing(env):
    script = make_fake(env.tmp, prints(FULL_LINES))
    rc = pj.run_job(job_args(env, script, run_id="someone-else"))
    assert rc == 2
    assert read_status(env.paths) is None
    assert read_lock(env.paths).run_id == RUN_ID          # untouched
    assert not env.paths.log_file("someone-else").exists()
    assert not (env.tmp / "child_argv.json").exists()      # child never spawned
    assert not env.paths.day_state_file(RUN_DATE).exists()


def test_no_lock_exits_2(env):
    env.paths.lock_file.unlink()
    script = make_fake(env.tmp, prints(FULL_LINES))
    assert pj.run_job(job_args(env, script)) == 2
    assert read_status(env.paths) is None


def test_reuse_with_phase1_exits_2(env):
    script = make_fake(env.tmp, prints(PHASE1_LINES))
    args = job_args(env, script, mode="phase1")
    args.reuse_phase1 = True    # parse_args refuses this combination, run_job must too
    assert pj.run_job(args) == 2
    assert read_status(env.paths) is None
    assert not (env.tmp / "child_argv.json").exists()


def test_superseded_exits_3_terminates_child_writes_nothing(env):
    other = {"run_id": "other-run", "created_at": "2026-10-11T15:00:05+00:00", "run_date": RUN_DATE,
             "mode": "full", "trigger": "button", "launcher_pid": 1}
    lock_path = str(env.paths.lock_file)
    code = f"""
        import json, os
        os.unlink({lock_path!r})
        open({lock_path!r}, 'w').write(json.dumps({other!r}))
    """
    script = make_fake(env.tmp, [("print", L_FAST), ("code", code), ("sleep", 60)])
    t0 = time.monotonic()
    rc = pj.run_job(job_args(env, script))
    assert time.monotonic() - t0 < 30                      # child was terminated, not waited out
    assert rc == 3
    assert read_lock(env.paths).run_id == "other-run"       # lock untouched
    s = read_status(env.paths)
    assert s["state"] == "running"                          # no terminal status written
    assert not has_done_marker(env.paths, RUN_DATE) and not has_done_marker(env.paths, local_today())
    assert not env.paths.day_state_file(RUN_DATE).exists()


# ------------------------------------------------------------------ heartbeat / midnight


def test_heartbeat_advances_while_child_is_silent(env):
    script = make_fake(env.tmp, prints([L_FAST]) + [("sleep", 3.5)])
    with Poller(env.paths) as poll:
        rc = pj.run_job(job_args(env, script))
    assert rc == 0
    beats = {s["heartbeat_at"] for s in poll.samples if s["state"] == "running"}
    assert len(beats) >= 3, f"heartbeat did not advance: {sorted(beats)}"


def test_midnight_full_run_writes_two_markers(env, monkeypatch):
    monkeypatch.setattr(pj, "local_today", lambda: "2026-10-12")
    script = make_fake(env.tmp, prints(FULL_LINES))
    assert pj.run_job(job_args(env, script)) == 0
    assert has_done_marker(env.paths, "2026-10-11") and has_done_marker(env.paths, "2026-10-12")


def test_midnight_phase1_records_both_days(env, monkeypatch):
    env.paths.lock_file.unlink()
    assert try_acquire_lock(env.paths, RUN_ID, RUN_DATE, "phase1", "button")
    monkeypatch.setattr(pj, "local_today", lambda: "2026-10-12")
    script = make_fake(env.tmp, prints(PHASE1_LINES))
    assert pj.run_job(job_args(env, script, mode="phase1", trigger="button")) == 0
    assert read_day_state(env.paths, "2026-10-11").phase1_success is True
    assert read_day_state(env.paths, "2026-10-12").phase1_success is True


def test_started_at_kept_from_prespawn_record(env, make_status, write_st):
    pre = make_status(run_id=RUN_ID, started_at="2026-10-11T14:59:00+00:00", pid=None)
    write_st(pre)
    script = make_fake(env.tmp, prints(FULL_LINES))
    assert pj.run_job(job_args(env, script)) == 0
    assert read_status(env.paths)["started_at"] == "2026-10-11T14:59:00+00:00"
