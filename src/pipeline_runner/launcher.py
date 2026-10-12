"""
Purpose: Start the ranking pipeline as a detached background process, automatically once per day or on a button press.

Connections:
  - src/pipeline_runner/run_state.py: lock, status file, done marker, per-day auto-state, liveness
  - src/pipeline_runner/phase1_store.py: check_phase1_reusable decides whether a full run skips Phase 1
  - src/pipeline_runner/steps.py: step tables used for the pre-spawn status record
  - cli/pipeline_job.py: the detached wrapper that this module spawns
  - config/settings.py: data_dir, raw_daily_dir, ticker_source, auto_rank_enabled
  - dashboard/components/pipeline_progress.py and dashboard/app.py: call maybe_start_daily_run / start_pipeline_run

In:  settings, run paths, requested mode / pool / top / trigger
Out: StartResult; a lock file, a pre-spawn status record and (on success) a detached cli/pipeline_job.py process
"""

import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from config.settings import ROOT_DIR, get_settings
from src.pipeline_runner.phase1_store import check_phase1_reusable
from src.pipeline_runner.run_state import (
    DayState,
    RunPaths,
    RunTrigger,
    break_stale_lock,
    get_run_paths,
    has_done_marker,
    local_today,
    lock_owner_alive,
    new_run_id,
    read_day_state,
    read_lock,
    read_status,
    record_fallback_started,
    record_full_failure,
    release_lock,
    try_acquire_lock,
    utc_now_iso,
    write_status,
)
from src.pipeline_runner.steps import RunMode, steps_for_run

logger = logging.getLogger("pipeline_runner.launcher")

DEFAULT_POOL: int = 100     # app.py number_input default
DEFAULT_TOP: int = 50       # app.py number_input default

AutoAction = Literal["disabled", "already_done", "running", "no_data",
                     "start_full", "start_fallback", "buttons_only"]

StartReason = Literal["started", "already_done", "running", "no_data",
                      "spawn_error", "disabled", "buttons_only"]

SpawnFn = Callable[[list[str], Path], int]   # (argv, cwd) -> pid

_VALID_MODES = ("phase1", "full")
_VALID_TRIGGERS = ("auto", "auto_fallback", "button")
_MSG_RUNNING = "A ranking run is already in progress."
_MSG_NO_DATA = "No scraped price data found. Run the scraper first."

# Popen objects of spawned wrappers, polled by reap_children() so POSIX zombies do not look alive.
_SPAWNED: list[subprocess.Popen] = []
_SPAWNED_LOCK = threading.Lock()


@dataclass(frozen=True)
class StartResult:
    started: bool
    reason: StartReason
    run_id: str | None
    mode: RunMode | None
    phase1_reused: bool
    message: str          # short, user-facing


# --------------------------------------------------------------------------- decision (pure)


def decide_auto_action(enabled: bool, has_marker: bool, lock_alive: bool,
                       day: DayState, has_data: bool) -> AutoAction:
    """Decide what the automatic daily run should do. Pure; evaluated in plan order."""
    if not enabled:
        return "disabled"
    if has_marker or day.full_success:
        return "already_done"
    if lock_alive:
        return "running"
    if not has_data:
        return "no_data"
    if day.full_failures == 0:
        return "start_full"
    if not day.fallback_started and not day.phase1_success:
        return "start_fallback"
    return "buttons_only"


# --------------------------------------------------------------------------- helpers


def has_scraped_data(raw_daily_dir: Path | None = None) -> bool:
    """True if the raw daily directory exists and holds at least one parquet file."""
    if raw_daily_dir is None:
        raw_daily_dir = get_settings().raw_daily_dir
    raw_daily_dir = Path(raw_daily_dir)
    try:
        return raw_daily_dir.is_dir() and next(iter(raw_daily_dir.glob("*.parquet")), None) is not None
    except OSError:
        return False


def build_wrapper_cmd(run_id: str, run_date: str, mode: RunMode, pool: int, top: int,
                      trigger: RunTrigger, reuse_phase1: bool,
                      root_dir: Path | None = None) -> list[str]:
    """Build the argv for cli/pipeline_job.py. All strings, no shell."""
    root = Path(root_dir) if root_dir is not None else ROOT_DIR
    cmd = [
        sys.executable, str(root / "cli" / "pipeline_job.py"),
        "--run-id", run_id, "--run-date", run_date,
        "--mode", mode, "--pool", str(pool), "--top", str(top),
        "--trigger", trigger,
    ]
    if reuse_phase1:
        cmd.append("--reuse-phase1")
    return cmd


def spawn_detached(argv: list[str], cwd: Path) -> int:
    """Spawn argv fully detached from this process and return its pid.

    POSIX: new session. Windows: new process group, hidden console, job breakaway
    (retried without breakaway if the job forbids it). Never DETACHED_PROCESS and never a shell.
    """
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    kwargs: dict[str, Any] = dict(
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        env=env,
    )
    if sys.platform == "win32":
        new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        breakaway = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
        try:
            proc = subprocess.Popen(argv, creationflags=new_group | no_window | breakaway, **kwargs)
        except OSError as exc:
            if getattr(exc, "winerror", None) != 5:
                raise
            proc = subprocess.Popen(argv, creationflags=new_group | no_window, **kwargs)
    else:
        proc = subprocess.Popen(argv, start_new_session=True, **kwargs)
    with _SPAWNED_LOCK:
        _SPAWNED.append(proc)
    return proc.pid


def reap_children() -> None:
    """Poll every spawned wrapper and drop the finished ones. Never raises."""
    try:
        with _SPAWNED_LOCK:
            alive = []
            for proc in _SPAWNED:
                try:
                    if proc.poll() is None:
                        alive.append(proc)
                except Exception:
                    pass
            _SPAWNED[:] = alive
    except Exception:
        pass


def _validate(mode: Any, pool: Any, top: Any, trigger: Any) -> None:
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
    if trigger not in _VALID_TRIGGERS:
        raise ValueError(f"trigger must be one of {_VALID_TRIGGERS}, got {trigger!r}")
    if isinstance(pool, bool) or not isinstance(pool, int) or not 10 <= pool <= 503:
        raise ValueError(f"pool must be an int in 10..503, got {pool!r}")
    if isinstance(top, bool) or not isinstance(top, int) or not 5 <= top <= 200:
        raise ValueError(f"top must be an int in 5..200, got {top!r}")


def _is_prespawn_record(s: dict | None, run_id: str) -> bool:
    """True if s is still the launcher's pre-spawn status: no pid, step 0, running."""
    return bool(s) and (s.get("run_id") == run_id and s.get("pid") is None
                        and s.get("state") == "running" and s.get("step_index") == 0)


def _no_start(reason: StartReason, message: str, run_id: str | None = None) -> StartResult:
    return StartResult(False, reason, run_id, None, False, message)


# --------------------------------------------------------------------------- start


def start_pipeline_run(mode: RunMode, pool: int = DEFAULT_POOL, top: int = DEFAULT_TOP,
                       trigger: RunTrigger = "button",
                       paths: RunPaths | None = None, settings: Any | None = None,
                       spawn: SpawnFn | None = None, now: datetime | None = None,
                       _expected_auto_action: AutoAction | None = None) -> StartResult:
    """Start a background pipeline run (plan 6.5).

    Buttons call this directly: they ignore the done marker and the day-state cap and are
    refused only while a run is live or when there is no scraped data.

    Raises:
        ValueError: On a bad mode, trigger, pool or top, before any side effect.
    """
    _validate(mode, pool, top, trigger)
    reap_children()
    if settings is None:
        settings = get_settings()
    if paths is None:
        paths = get_run_paths(settings.data_dir)
    today = local_today()

    lock = read_lock(paths)
    if lock is not None:
        if lock_owner_alive(paths, lock, now):
            return _no_start("running", _MSG_RUNNING, lock.run_id)
        break_stale_lock(paths, now)

    if not has_scraped_data(settings.raw_daily_dir):
        return _no_start("no_data", _MSG_NO_DATA)

    reuse = mode == "full" and bool(check_phase1_reusable(settings, pool=pool).usable)
    run_id = new_run_id(now)
    try:
        acquired = try_acquire_lock(paths, run_id, today, mode, trigger, now)
    except OSError as exc:   # e.g. read-only or missing data dir; no lock exists to release
        err = f"Could not start background run: {exc}"
        logger.warning(err)
        return StartResult(False, "spawn_error", run_id, mode, reuse, err[:300])
    if not acquired:
        cur = read_lock(paths)
        return _no_start("running", _MSG_RUNNING, cur.run_id if cur else None)

    # Lock held from here. Every exit path must release it if no wrapper was spawned.
    spawned = False
    try:
        if _expected_auto_action is not None:
            actual = decide_auto_action(
                bool(settings.auto_rank_enabled), has_done_marker(paths, today), False,
                read_day_state(paths, today), True)
            if actual != _expected_auto_action:
                release_lock(paths, run_id)
                if actual in ("already_done", "buttons_only", "disabled"):
                    return _no_start(actual, "Automatic ranking is not needed right now.", run_id)
                return _no_start("running", _MSG_RUNNING, run_id)

        if trigger == "auto_fallback":
            record_fallback_started(paths, today)   # claimed even if the spawn fails

        started_iso = utc_now_iso() if now is None else now.astimezone(timezone.utc).isoformat(timespec="seconds")
        steps = steps_for_run(mode, reuse)
        status: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "state": "running",
            "mode": mode,
            "trigger": trigger,
            "phase1_reused": reuse,
            "pool": pool,
            "top": top,
            "pid": None,
            "child_pid": None,
            "step_index": 0,
            "step_total": len(steps),
            "label": steps[0][1],
            "run_date": today,
            "started_at": started_iso,
            "step_started_at": started_iso,
            "heartbeat_at": started_iso,
            "finished_at": None,
            "returncode": None,
            "error": None,
            "log_path": f"data/logs/pipeline_job_{run_id}.log",
        }
        try:
            write_status(paths, status)
            argv = build_wrapper_cmd(run_id, today, mode, pool, top, trigger, reuse)
            pid = (spawn or spawn_detached)(argv, ROOT_DIR)
            spawned = True
            s = read_status(paths)
            if _is_prespawn_record(s, run_id):
                # Re-read right before writing and only fill while the record is still the
                # launcher's own; never overwrite anything the wrapper has written since.
                s = read_status(paths)
                if _is_prespawn_record(s, run_id):
                    write_status(paths, {**s, "pid": pid})
        except Exception as exc:
            if spawned:
                # The wrapper is running; only the pid fill failed. Leave the lock to the wrapper.
                logger.warning("pid fill failed for run %s: %s", run_id, exc)
            else:
                err = f"Could not start background run: {exc}"
                logger.warning(err)
                try:
                    write_status(paths, {**status, "state": "failed", "label": "Failed to start",
                                         "error": err[:1000], "finished_at": utc_now_iso()})
                except Exception:
                    pass
                if mode == "full":
                    record_full_failure(paths, today)
                release_lock(paths, run_id)
                return StartResult(False, "spawn_error", run_id, mode, reuse, err[:300])
    except BaseException:
        if not spawned:
            release_lock(paths, run_id)
        raise

    if mode == "full":
        msg = ("Full analysis started (reusing today's Phase 1, running ML only)." if reuse
               else "Full analysis started in the background.")
    else:
        msg = "Fast rank started in the background."
    return StartResult(True, "started", run_id, mode, reuse, msg)


# --------------------------------------------------------------------------- automatic daily run

_ACTION_TO_REASON: dict[str, StartReason] = {
    "disabled": "disabled",
    "already_done": "already_done",
    "running": "running",
    "no_data": "no_data",
    "buttons_only": "buttons_only",
}


def maybe_start_daily_run(paths: RunPaths | None = None, settings: Any | None = None,
                          spawn: SpawnFn | None = None, now: datetime | None = None) -> StartResult:
    """Start the automatic daily run if one is due (plan 6.6)."""
    reap_children()
    if settings is None:
        try:
            settings = get_settings()
        except Exception as exc:   # e.g. an unusable .env value
            logger.warning("Settings unavailable, automatic ranking disabled: %s", exc)
            return _no_start("disabled", "Automatic ranking is disabled.")
    if not bool(getattr(settings, "auto_rank_enabled", True)):
        return _no_start("disabled", "Automatic ranking is disabled.")

    if paths is None:
        paths = get_run_paths(settings.data_dir)
    today = local_today()
    lock = read_lock(paths)
    lock_alive = bool(lock) and lock_owner_alive(paths, lock, now)
    if lock is not None and not lock_alive:
        break_stale_lock(paths, now)
        lock_alive = False
    action = decide_auto_action(
        bool(settings.auto_rank_enabled), has_done_marker(paths, today), lock_alive,
        read_day_state(paths, today), has_scraped_data(settings.raw_daily_dir))

    if action == "start_full":
        return start_pipeline_run("full", DEFAULT_POOL, DEFAULT_TOP, trigger="auto",
                                  paths=paths, settings=settings, spawn=spawn, now=now,
                                  _expected_auto_action="start_full")
    if action == "start_fallback":
        return start_pipeline_run("phase1", DEFAULT_POOL, DEFAULT_TOP, trigger="auto_fallback",
                                  paths=paths, settings=settings, spawn=spawn, now=now,
                                  _expected_auto_action="start_fallback")
    run_id = lock.run_id if (action == "running" and lock is not None) else None
    msg = _MSG_RUNNING if action == "running" else (
        _MSG_NO_DATA if action == "no_data" else "Nothing to start automatically.")
    return _no_start(_ACTION_TO_REASON[action], msg, run_id)
