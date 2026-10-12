"""
Purpose: On-disk state for the background ranking pipeline run: paths, atomic JSON IO,
the run lock, stale-lock breaking, the daily done marker, the per-day auto-state,
PID liveness and the derived view the dashboard renders. No Streamlit, no spawning.

Connections:
  - src/pipeline_runner/steps.py: RunMode / RunState aliases and progress_fraction
  - src/pipeline_runner/phase1_store.py: uses write_json_atomic
  - src/pipeline_runner/launcher.py: acquires the lock, writes the pre-spawn status
  - cli/pipeline_job.py: wrapper writes status, heartbeat, markers, day state, releases lock
  - dashboard/components/pipeline_progress.py: reads the view via get_run_view
  - config/settings.py: get_settings().data_dir is the default data directory

In:  a RunPaths (from get_run_paths), run ids, local day keys, UTC datetimes
Out: status / lock / marker / day-state files under <data_dir>/run_cache, RunView for the UI
"""

import json
import logging
import os
import re
import secrets
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Literal

from src.pipeline_runner.steps import RunMode, RunState, progress_fraction

logger = logging.getLogger("pipeline_runner.run_state")

RunTrigger = Literal["auto", "auto_fallback", "button"]
ViewState = Literal["idle", "running", "done", "failed"]

STATUS_SCHEMA_VERSION = 1
DAY_STATE_SCHEMA_VERSION = 1
HEARTBEAT_INTERVAL_S = 5       # wrapper writes heartbeat at least this often
PID_REUSE_GUARD_S = 900        # SECONDARY guard only: pid looks alive but no heartbeat for 15 min
FUTURE_TOLERANCE_S = 60        # a timestamp further ahead than this is treated as unknown
SPAWN_GRACE_S = 60             # lock/run with no pid recorded yet is trusted this long
BREAKER_STALE_S = 30           # lock-break mutex older than this is removed
DONE_DISMISS_S = 6             # bar shows "done" this long after finished_at
FAILED_DISMISS_S = 30          # bar shows "failed" this long after finished_at
MARKER_RETENTION_DAYS = 7
LOG_RETENTION_COUNT = 14

_VALID_STATES = ("running", "done", "failed")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_STALE_ERROR = "Run stopped responding (process exited or was killed without reporting). See log."


# --------------------------------------------------------------------------- paths


@dataclass(frozen=True)
class RunPaths:
    data_dir: Path
    run_cache_dir: Path      # <data_dir>/run_cache
    lock_file: Path          # <run_cache_dir>/pipeline_run.lock
    breaker_file: Path       # <run_cache_dir>/pipeline_run.lock.break
    status_file: Path        # <run_cache_dir>/pipeline_status.json
    log_dir: Path            # <data_dir>/logs

    def done_marker(self, day: str) -> Path:
        return self.run_cache_dir / f".done_rank_{day}"

    def day_state_file(self, day: str) -> Path:
        return self.run_cache_dir / f"auto_rank_{day}.json"

    def log_file(self, run_id: str) -> Path:
        return self.log_dir / f"pipeline_job_{run_id}.log"


def get_run_paths(data_dir: Path | None = None) -> RunPaths:
    """Build RunPaths and create run_cache and logs directories.

    Args:
        data_dir: Data directory; None uses get_settings().data_dir.

    Returns:
        RunPaths for that data directory.
    """
    if data_dir is None:
        from config.settings import get_settings
        data_dir = get_settings().data_dir
    data_dir = Path(data_dir)
    run_cache = data_dir / "run_cache"
    paths = RunPaths(
        data_dir=data_dir,
        run_cache_dir=run_cache,
        lock_file=run_cache / "pipeline_run.lock",
        breaker_file=run_cache / "pipeline_run.lock.break",
        status_file=run_cache / "pipeline_status.json",
        log_dir=data_dir / "logs",
    )
    for d in (paths.run_cache_dir, paths.log_dir):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as exc:   # e.g. read-only data dir: readers degrade, writers fail on their own
            logger.warning("Could not create %s: %s", d, exc)
    return paths


# --------------------------------------------------------------------------- time / ids


def _utc(now: datetime | None) -> datetime:
    return now if now is not None else datetime.now(timezone.utc)


def local_today() -> str:
    """Local calendar date as YYYY-MM-DD."""
    return datetime.now().date().isoformat()


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with offset, seconds precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id(now: datetime | None = None) -> str:
    """Filename-safe run id, e.g. 20261011T150203Z-a1b2c3."""
    return f"{_utc(now).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"


def _parse_iso(value: object) -> datetime | None:
    """Parse an ISO string to an aware UTC datetime; None when invalid."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _age_s(now: datetime, then: datetime | None) -> float:
    """Seconds between then and now; infinite when then is unknown."""
    if then is None:
        return float("inf")
    return (now - then).total_seconds()


def _age_s_safe(now: datetime, then: datetime | None) -> float:
    """Like _age_s, but a timestamp more than FUTURE_TOLERANCE_S ahead counts as unknown (inf)
    and a slightly-future one counts as age 0."""
    age = _age_s(now, then)
    if age < -FUTURE_TOLERANCE_S:
        return float("inf")
    return max(age, 0.0)


# --------------------------------------------------------------------------- atomic JSON


def write_json_atomic(path: Path, data: dict) -> None:
    """Write data as UTF-8 JSON via a temp file and os.replace.

    Retries 5 times, 50 ms apart, on PermissionError (Windows reader holds the file).
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        attempt = 0
        while True:
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                attempt += 1
                if attempt > 5:
                    raise
                time.sleep(0.05)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def read_json(path: Path) -> dict | None:
    """Read a JSON object; None if missing, empty, invalid or not an object. Never raises."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        if not text.strip():
            return None
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


# --------------------------------------------------------------------------- status


def read_status(paths: RunPaths) -> dict | None:
    """Read the status file; None unless schema_version and state are valid."""
    s = read_json(paths.status_file)
    if s is None:
        return None
    if s.get("schema_version") != STATUS_SCHEMA_VERSION:
        return None
    if s.get("state") not in _VALID_STATES:
        return None
    return s


def write_status(paths: RunPaths, status: dict) -> None:
    """Atomically write the status file."""
    write_json_atomic(paths.status_file, status)


# --------------------------------------------------------------------------- pid liveness


def pid_alive(pid: int | None) -> bool:
    """True if a process with this pid exists. Never terminates the process."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if sys.platform == "win32":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _pid_alive_windows(pid: int) -> bool:
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return kernel32.GetLastError() == 5  # ACCESS_DENIED means it exists
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


# --------------------------------------------------------------------------- lock


@dataclass
class LockInfo:
    run_id: str | None
    created_at: datetime     # from JSON; if missing/unparseable -> file mtime (UTC)
    run_date: str | None
    mode: str | None
    trigger: str | None


def read_lock(paths: RunPaths) -> LockInfo | None:
    """Read the lock file. None only if the file does not exist."""
    try:
        mtime = datetime.fromtimestamp(paths.lock_file.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None
    data = read_json(paths.lock_file) or {}

    def _str(key: str) -> str | None:
        v = data.get(key)
        return v if isinstance(v, str) and v else None

    return LockInfo(
        run_id=_str("run_id"),
        created_at=_parse_iso(data.get("created_at")) or mtime,
        run_date=_str("run_date"),
        mode=_str("mode"),
        trigger=_str("trigger"),
    )


def try_acquire_lock(paths: RunPaths, run_id: str, run_date: str, mode: RunMode,
                     trigger: RunTrigger, now: datetime | None = None) -> bool:
    """Atomically create the lock file. False if it already exists."""
    paths.run_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(paths.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    payload = {
        "run_id": run_id,
        "created_at": _utc(now).astimezone(timezone.utc).isoformat(timespec="seconds"),
        "run_date": run_date,
        "mode": mode,
        "trigger": trigger,
        "launcher_pid": os.getpid(),
    }
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        try:
            paths.lock_file.unlink()
        except OSError:
            pass
        raise
    return True


def release_lock(paths: RunPaths, run_id: str) -> bool:
    """Remove the lock only if it belongs to run_id. True if removed."""
    lock = read_lock(paths)
    if lock is None or lock.run_id != run_id:
        return False
    try:
        paths.lock_file.unlink()
    except FileNotFoundError:
        return False
    return True


def lock_owner_alive(paths: RunPaths, lock: LockInfo, now: datetime | None = None) -> bool:
    """Decide whether the process holding the lock is alive (plan 6.3, PID first)."""
    now = _utc(now)
    s = read_status(paths)
    if s is not None and lock.run_id is not None and s.get("run_id") == lock.run_id:
        pid = s.get("pid")
        if s["state"] in ("done", "failed"):
            finished = _parse_iso(s.get("finished_at"))
            return pid_alive(pid) or _age_s_safe(now, finished) < SPAWN_GRACE_S
        # running
        if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
            hb = _parse_iso(s.get("heartbeat_at")) or _parse_iso(s.get("started_at"))
            return pid_alive(pid) and _age_s(now, hb) < PID_REUSE_GUARD_S
        return _age_s_safe(now, lock.created_at) < SPAWN_GRACE_S
    return _age_s_safe(now, lock.created_at) < SPAWN_GRACE_S


def is_run_active(paths: RunPaths, now: datetime | None = None) -> bool:
    """True if a lock exists and its owner is alive. Read-only."""
    lock = read_lock(paths)
    return lock is not None and lock_owner_alive(paths, lock, now)


def _acquire_breaker(paths: RunPaths) -> bool:
    """Take the lock-break mutex; remove a stale one once. True if held."""
    for _ in range(2):
        try:
            fd = os.open(paths.breaker_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                age = time.time() - paths.breaker_file.stat().st_mtime
            except FileNotFoundError:
                continue  # released meanwhile; retry
            except OSError:
                return False
            if age > BREAKER_STALE_S:
                try:
                    paths.breaker_file.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    return False
                continue
            return False
        try:
            os.write(fd, str(os.getpid()).encode("ascii"))
        finally:
            os.close(fd)
        return True
    return False


def break_stale_lock(paths: RunPaths, now: datetime | None = None) -> bool:
    """Remove a dead run's lock, mark its status failed, count a full failure (plan 6.4).

    Returns True only if this call broke the lock.
    """
    now = _utc(now)
    lock = read_lock(paths)
    if lock is None or lock_owner_alive(paths, lock, now):
        return False
    if not _acquire_breaker(paths):
        return False
    try:
        orig_id = lock.run_id
        lock = read_lock(paths)
        if lock is None or lock.run_id != orig_id or lock_owner_alive(paths, lock, now):
            return False
        s = read_status(paths)
        if (s is not None and lock.run_id is not None
                and s.get("run_id") == lock.run_id and s["state"] == "running"):
            write_status(paths, {
                **s,
                "state": "failed",
                "finished_at": now.isoformat(timespec="seconds"),
                "error": _STALE_ERROR,
            })
            if (lock.mode or s.get("mode")) == "full":
                record_full_failure(paths, lock.run_date or local_today())
        try:
            paths.lock_file.unlink()
        except FileNotFoundError:
            pass
        return True
    finally:
        try:
            paths.breaker_file.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------- done marker


def has_done_marker(paths: RunPaths, day: str | None = None) -> bool:
    """True if a full run succeeded for this local day."""
    return paths.done_marker(day or local_today()).exists()


def write_done_markers(paths: RunPaths, days: Iterable[str], now: datetime | None = None) -> None:
    """Write one done marker per distinct day. Never raises."""
    stamp = _utc(now).astimezone(timezone.utc).isoformat(timespec="seconds") + "\n"
    for day in sorted(set(days)):
        try:
            paths.run_cache_dir.mkdir(parents=True, exist_ok=True)
            paths.done_marker(day).write_text(stamp, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write done marker for %s: %s", day, exc)


# --------------------------------------------------------------------------- day state


@dataclass(frozen=True)
class DayState:
    day: str
    full_failures: int = 0          # failed FULL runs today (any trigger, incl. stale-broken)
    fallback_started: bool = False  # the automatic Phase-1 fallback was claimed today
    phase1_success: bool = False    # any Phase-1 run (any trigger) succeeded today
    full_success: bool = False      # a FULL run succeeded today (backs up the done marker)


def read_day_state(paths: RunPaths, day: str | None = None) -> DayState:
    """Read the per-day auto-state; defaults when missing or invalid. Never raises."""
    day = day or local_today()
    try:
        d = read_json(paths.day_state_file(day))
        if d is None or d.get("schema_version") != DAY_STATE_SCHEMA_VERSION:
            return DayState(day)
        ff = d.get("full_failures", 0)
        fb = d.get("fallback_started", False)
        p1 = d.get("phase1_success", False)
        fs = d.get("full_success", False)   # absent in older files = False
        if isinstance(ff, bool) or not isinstance(ff, int) or ff < 0:
            return DayState(day)
        if not isinstance(fb, bool) or not isinstance(p1, bool) or not isinstance(fs, bool):
            return DayState(day)
        return DayState(day, ff, fb, p1, fs)
    except Exception:
        return DayState(day)


def _write_day_state(paths: RunPaths, state: DayState) -> None:
    payload = {
        "schema_version": DAY_STATE_SCHEMA_VERSION,
        "day": state.day,
        "full_failures": state.full_failures,
        "fallback_started": state.fallback_started,
        "phase1_success": state.phase1_success,
        "full_success": state.full_success,
        "updated_at": utc_now_iso(),
    }
    paths.run_cache_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths.day_state_file(state.day), payload)


def record_full_failure(paths: RunPaths, day: str) -> DayState:
    """full_failures += 1 (caller holds the run lock or the breaker mutex)."""
    cur = read_day_state(paths, day)
    new = DayState(day, cur.full_failures + 1, cur.fallback_started, cur.phase1_success, cur.full_success)
    _write_day_state(paths, new)
    return new


def record_fallback_started(paths: RunPaths, day: str) -> DayState:
    """Mark the automatic Phase-1 fallback as claimed for the day."""
    cur = read_day_state(paths, day)
    new = DayState(day, cur.full_failures, True, cur.phase1_success, cur.full_success)
    _write_day_state(paths, new)
    return new


def record_full_success(paths: RunPaths, day: str) -> DayState:
    """Mark a full-run success for the day (second success signal besides the done marker)."""
    cur = read_day_state(paths, day)
    new = DayState(day, cur.full_failures, cur.fallback_started, cur.phase1_success, True)
    _write_day_state(paths, new)
    return new


def record_phase1_success(paths: RunPaths, day: str) -> DayState:
    """Mark a Phase-1 success for the day."""
    cur = read_day_state(paths, day)
    new = DayState(day, cur.full_failures, cur.fallback_started, True, cur.full_success)
    _write_day_state(paths, new)
    return new


# --------------------------------------------------------------------------- cleanup


def cleanup_old_files(paths: RunPaths, today: str | None = None) -> None:
    """Delete old per-day markers and surplus logs. Never raises."""
    try:
        today_d = date.fromisoformat(today or local_today())
    except ValueError:
        return
    cutoff = today_d - timedelta(days=MARKER_RETENTION_DAYS)

    def _old(day_text: str) -> bool:
        try:
            return date.fromisoformat(day_text) < cutoff
        except ValueError:
            return False

    try:
        for f in paths.run_cache_dir.iterdir():
            name = f.name
            day_text = None
            if name.startswith(".done_rank_"):
                day_text = name[len(".done_rank_"):]
            elif name.startswith("auto_rank_") and name.endswith(".json"):
                day_text = name[len("auto_rank_"):-len(".json")]
            if day_text is not None and _old(day_text):
                try:
                    f.unlink()
                except OSError:
                    pass
    except Exception as exc:
        logger.warning("Marker cleanup failed: %s", exc)

    try:
        logs = []
        for f in paths.log_dir.glob("pipeline_job_*.log"):
            try:
                logs.append((f.stat().st_mtime, f))
            except OSError:
                pass
        logs.sort(key=lambda t: t[0], reverse=True)
        for _, f in logs[LOG_RETENTION_COUNT:]:
            try:
                f.unlink()
            except OSError:
                pass
    except Exception as exc:
        logger.warning("Log cleanup failed: %s", exc)


# --------------------------------------------------------------------------- UI view


@dataclass(frozen=True)
class RunView:
    state: ViewState
    run_id: str | None
    mode: str | None
    trigger: str | None
    phase1_reused: bool
    label: str
    step_index: int
    step_total: int
    progress: float
    error: str | None
    started_at: str | None
    step_started_at: str | None
    finished_at: str | None


def _idle_view() -> RunView:
    return RunView("idle", None, None, None, False, "", 0, 0, 0.0, None, None, None, None)


def _int(v: object, default: int = 0) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) else default


def _opt_str(v: object) -> str | None:
    return v if isinstance(v, str) else None


def get_run_view(paths: RunPaths, now: datetime | None = None) -> RunView:
    """Derive what the progress bar should show (plan 7.2). Read-only."""
    now = _utc(now)
    s = read_status(paths)
    if s is None:
        return _idle_view()

    stored: RunState = s["state"]
    view_state: ViewState
    error = _opt_str(s.get("error"))
    label = _opt_str(s.get("label")) or ""
    finished_at = _opt_str(s.get("finished_at"))

    if stored == "running":
        lock = read_lock(paths)
        if lock is None or lock.run_id != s.get("run_id"):
            return _idle_view()
        if lock_owner_alive(paths, lock, now):
            view_state = "running"
        else:
            view_state = "failed"
            label = "Run stopped responding"
            error = _STALE_ERROR
    elif stored == "done":
        age = _age_s_safe(now, _parse_iso(finished_at))
        if age >= DONE_DISMISS_S:
            return _idle_view()
        view_state = "done"
    else:
        age = _age_s_safe(now, _parse_iso(finished_at))
        if age >= FAILED_DISMISS_S:
            return _idle_view()
        view_state = "failed"

    step_index = _int(s.get("step_index"))
    step_total = _int(s.get("step_total"))
    # progress_fraction treats "done" as 1.0; a failed view shows its last position.
    prog_state: RunState = "done" if view_state == "done" else "running"
    return RunView(
        state=view_state,
        run_id=_opt_str(s.get("run_id")),
        mode=_opt_str(s.get("mode")),
        trigger=_opt_str(s.get("trigger")),
        phase1_reused=bool(s.get("phase1_reused", False)),
        label=label,
        step_index=step_index,
        step_total=step_total,
        progress=progress_fraction(step_index, step_total, prog_state),
        error=error,
        started_at=_opt_str(s.get("started_at")),
        step_started_at=_opt_str(s.get("step_started_at")),
        finished_at=finished_at,
    )


def ranking_cache_version(data_dir: Path | None = None) -> float:
    """st_mtime of <data_dir>/ranking_cache.parquet, or 0.0 when absent."""
    try:
        if data_dir is None:
            from config.settings import get_settings
            data_dir = get_settings().data_dir
        return float((Path(data_dir) / "ranking_cache.parquet").stat().st_mtime)
    except Exception:
        return 0.0


def read_log_tail(paths: RunPaths, run_id: str, max_lines: int = 100) -> list[str]:
    """Last max_lines lines of a run's log; [] if missing or run_id is unsafe."""
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id) or max_lines <= 0:
        return []
    try:
        text = paths.log_file(run_id).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-max_lines:]
