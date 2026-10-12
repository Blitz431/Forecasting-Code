"""Detached wrapper that runs cli/pipeline.py and reports progress to the status file.

The launcher (src/pipeline_runner/launcher.py) starts this script fully detached. It owns the
run lock for its run id. It runs cli/pipeline.py as a child, turns the child's stdout markers into
step progress, keeps a heartbeat, enforces a hard time limit, and on exit writes the day state or
done marker first, the terminal status second and releases the lock last.

Connections:
  - src/pipeline_runner/launcher.py: spawns this script with --run-id/--run-date/--mode/--pool/--top/--trigger
  - src/pipeline_runner/run_state.py: status, lock, done marker, day state
  - src/pipeline_runner/steps.py: step tables, advance_step, PHASE1_FALLBACK_MARKER
  - cli/pipeline.py: the child process (--phase1-only / --no-deep / --skip-phase1)

In:  command line arguments (see parse_args)
Out: status file, heartbeat, log file, done marker / day state; exit code
     0 ok, 1 failed, 2 not owner or bad arguments, 3 superseded

Usage
-----
    python cli/pipeline_job.py --run-id ID --run-date 2026-10-11 --mode full --pool 100 --top 50 --trigger auto
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline_runner.run_state import (  # noqa: E402
    HEARTBEAT_INTERVAL_S,
    RunPaths,
    cleanup_old_files,
    get_run_paths,
    local_today,
    read_lock,
    read_status,
    record_full_failure,
    record_full_success,
    record_phase1_success,
    release_lock,
    utc_now_iso,
    write_done_markers,
    write_status,
)
from src.pipeline_runner.steps import (  # noqa: E402
    PHASE1_FALLBACK_MARKER,
    PIPELINE_STEPS_FULL,
    RunMode,
    advance_step,
    steps_for_run,
)
from src.utils.logging import setup_logger  # noqa: E402

logger = setup_logger("cli.pipeline_job")

MAX_RUNTIME_PHASE1_S = 1800
MAX_RUNTIME_FULL_S = 14400   # 4 h hard limit
KILL_GRACE_S = 10
ERROR_MAX_CHARS = 1000
TAIL_LINES = 20

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BAD = 2
EXIT_SUPERSEDED = 3


# --------------------------------------------------------------------------- args


def _run_date(value: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a YYYY-MM-DD date: {value!r}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the wrapper command line. Exits with code 2 on invalid input."""
    p = argparse.ArgumentParser(description="Background wrapper for cli/pipeline.py")
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-date", required=True, type=_run_date, metavar="YYYY-MM-DD")
    p.add_argument("--mode", required=True, choices=["phase1", "full"])
    p.add_argument("--pool", required=True, type=int)
    p.add_argument("--top", required=True, type=int)
    p.add_argument("--trigger", required=True, choices=["auto", "auto_fallback", "button"])
    p.add_argument("--reuse-phase1", action="store_true",
                   help="Full mode only: pass --skip-phase1 to the pipeline.")
    # test hooks
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--pipeline-script", type=Path, default=None)
    p.add_argument("--max-runtime-s", type=int, default=None)
    p.add_argument("--heartbeat-s", type=float, default=None)
    args = p.parse_args(argv)
    if args.reuse_phase1 and args.mode != "full":
        p.error("--reuse-phase1 is only valid with --mode full")
    return args


def pipeline_argv(mode: RunMode, pool: int, top: int, script: Path, reuse_phase1: bool) -> list[str]:
    """Build the argv for the pipeline child. No shell; unbuffered stdout."""
    cmd = [sys.executable, "-u", str(script), "--pool", str(pool), "--top", str(top)]
    if mode == "phase1":
        cmd.append("--phase1-only")
    else:
        cmd.append("--no-deep")
        if reuse_phase1:
            cmd.append("--skip-phase1")
    return cmd


# --------------------------------------------------------------------------- job


class _Job:
    def __init__(self, args: argparse.Namespace, paths: RunPaths) -> None:
        self.args = args
        self.paths = paths
        self.mode: RunMode = args.mode
        self.run_id: str = args.run_id
        self.lock = threading.Lock()          # guards status writes and the flags below
        self.stop_hb = threading.Event()
        self.proc: subprocess.Popen | None = None
        self.reader: threading.Thread | None = None
        self.stop_lock = threading.Lock()
        self.superseded = False
        self.timed_out = False
        self.terminated_by_signal = False
        self.finished = False
        self.reused = bool(args.reuse_phase1)
        self.steps = steps_for_run(self.mode, self.reused)
        self.status: dict = {}
        self.tail: deque[str] = deque(maxlen=TAIL_LINES)
        self.log_fh = None
        self.t0 = time.monotonic()
        self.max_runtime_s = (args.max_runtime_s if args.max_runtime_s is not None
                              else (MAX_RUNTIME_FULL_S if self.mode == "full" else MAX_RUNTIME_PHASE1_S))
        self.heartbeat_s = args.heartbeat_s if args.heartbeat_s is not None else HEARTBEAT_INTERVAL_S

    # ---- ownership / status ----

    def _owns_lock(self) -> bool:
        lock = read_lock(self.paths)
        return lock is not None and lock.run_id == self.run_id

    def _write_status_if_owner(self) -> bool:
        """Write the status unless the lock moved on. Caller holds self.lock."""
        if self.superseded:
            return False
        if not self._owns_lock():
            self.superseded = True
            threading.Thread(target=self._stop_child, daemon=True).start()
            return False
        write_status(self.paths, self.status)
        return True

    def _touch(self, **changes) -> bool:
        with self.lock:
            if self.finished:
                return False
            self.status.update(changes)
            self.status["heartbeat_at"] = utc_now_iso()
            return self._write_status_if_owner()

    def _stop_child(self) -> None:
        proc = self.proc
        if proc is None:
            return
        with self.stop_lock:
            if proc.poll() is not None:
                return
            try:
                self._signal_tree(proc, force=False)
                try:
                    proc.wait(timeout=KILL_GRACE_S)
                except subprocess.TimeoutExpired:
                    self._signal_tree(proc, force=True)
            except OSError:
                pass
            # Pipe holders (grandchildren) may survive; make sure the reader loop is released.
            self._drain_reader(proc)

    @staticmethod
    def _signal_tree(proc: subprocess.Popen, force: bool) -> None:
        """Terminate (or kill) the child and its whole process group / tree. Best effort."""
        if sys.platform == "win32":
            if force:
                try:
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=15, check=False)
                except (OSError, subprocess.SubprocessError):
                    pass
            try:
                proc.kill() if force else proc.terminate()
            except OSError:
                pass
            return
        try:
            os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        except OSError:
            try:
                proc.kill() if force else proc.terminate()
            except OSError:
                pass

    def _drain_reader(self, proc: subprocess.Popen) -> None:
        """After a stop, give the reader a moment to drain. If a pipe holder survives, the
        daemon reader is abandoned (closing a pipe another thread is reading can block)."""
        reader = self.reader
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=2)

    def _heartbeat_loop(self) -> None:
        while not self.stop_hb.wait(self.heartbeat_s):
            if not self.timed_out and time.monotonic() - self.t0 > self.max_runtime_s:
                self.timed_out = True
                self._stop_child()
            self._touch()

    # ---- log / child ----

    def _log(self, text: str) -> None:
        if self.log_fh is None:
            return
        try:
            self.log_fh.write(text)
            self.log_fh.flush()
        except (OSError, ValueError):
            pass

    def _spawn_child(self) -> subprocess.Popen:
        script = self.args.pipeline_script or (ROOT / "cli" / "pipeline.py")
        cmd = pipeline_argv(self.mode, self.args.pool, self.args.top, Path(script), self.reused)
        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = (getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
                                       | getattr(subprocess, "CREATE_NO_WINDOW", 0)
                                       | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        else:
            kwargs["start_new_session"] = True   # own process group so the whole tree can be signalled
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
        return subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            cwd=str(ROOT), env=env, **kwargs)

    def _read_output(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self._handle_line(line)
        except (OSError, ValueError):
            pass   # pipe closed under us after a stop

    def _handle_line(self, line: str) -> None:
        self._log(line if line.endswith("\n") else line + "\n")
        text = line.rstrip("\r\n")
        self.tail.append(text)
        now = utc_now_iso()
        if self.reused and PHASE1_FALLBACK_MARKER in text:
            self.reused = False
            self.steps = PIPELINE_STEPS_FULL
            self._touch(phase1_reused=False, step_index=0, step_total=len(self.steps),
                        label=self.steps[0][1], step_started_at=now)
            return
        cur = int(self.status["step_index"])
        new = advance_step(text, self.steps, cur)
        if new != cur:
            self._touch(step_index=new, label=self.steps[new][1], step_started_at=now)

    # ---- finish ----

    def _error_text(self, reason: str) -> str:
        tail = "\n".join(self.tail)
        if not tail:
            return reason[:ERROR_MAX_CHARS]
        budget = ERROR_MAX_CHARS - len(reason) - 1
        if budget <= 0:
            return reason[:ERROR_MAX_CHARS]
        return reason + "\n" + tail[-budget:]

    def _days(self) -> set[str]:
        return {self.args.run_date, local_today()}

    def _finish(self, ok: bool, reason: str | None, returncode: int | None) -> int:
        """Order is mandatory: day state / marker, then status, then the lock."""
        self.stop_hb.set()
        with self.lock:
            self.finished = True
            if self.superseded or not self._owns_lock():
                self.superseded = True
                return EXIT_SUPERSEDED
        try:
            if ok:
                if self.mode == "full":
                    write_done_markers(self.paths, self._days())   # swallows OSError
                    for day in sorted(self._days()):               # backup success signal
                        record_full_success(self.paths, day)
                else:
                    for day in sorted(self._days()):
                        record_phase1_success(self.paths, day)
            elif self.mode == "full":
                record_full_failure(self.paths, self.args.run_date)
        except Exception as exc:  # status and lock release must still happen
            logger.warning("Could not record run outcome: %s", exc)
        now = utc_now_iso()
        try:
            with self.lock:
                if ok:
                    self.status.update(
                        state="done", step_index=len(self.steps) - 1, step_total=len(self.steps),
                        label="Ranking complete", finished_at=now, heartbeat_at=now,
                        returncode=returncode, error=None)
                else:
                    self.status.update(
                        state="failed", label="Run failed", finished_at=now, heartbeat_at=now,
                        returncode=returncode, error=self._error_text(reason or "Run failed"))
                write_status(self.paths, self.status)
        except Exception as exc:
            logger.warning("Could not write final status: %s", exc)
        release_lock(self.paths, self.run_id)
        return EXIT_OK if ok else EXIT_FAILED

    # ---- main ----

    def run(self) -> int:
        args, paths = self.args, self.paths
        cleanup_old_files(paths)
        if hasattr(os, "nice"):
            try:
                os.nice(10)
            except OSError:
                pass

        log_path = paths.log_file(self.run_id)
        try:
            self.log_fh = open(log_path, "a", encoding="utf-8")
        except OSError:
            self.log_fh = None
        self._log(f"# run_id={self.run_id} mode={self.mode} pool={args.pool} top={args.top} "
                  f"trigger={args.trigger} reuse_phase1={self.reused} "
                  f"started={datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")

        now = utc_now_iso()
        prev = read_status(paths)
        started_at = (prev.get("started_at") if prev and prev.get("run_id") == self.run_id else None) or now
        self.status = {
            "schema_version": 1,
            "run_id": self.run_id,
            "state": "running",
            "mode": self.mode,
            "trigger": args.trigger,
            "phase1_reused": self.reused,
            "pool": args.pool,
            "top": args.top,
            "pid": os.getpid(),
            "child_pid": None,
            "step_index": 0,
            "step_total": len(self.steps),
            "label": self.steps[0][1],
            "run_date": args.run_date,
            "started_at": started_at,
            "step_started_at": now,
            "heartbeat_at": now,
            "finished_at": None,
            "returncode": None,
            "error": None,
            "log_path": f"data/logs/pipeline_job_{self.run_id}.log",
        }
        with self.lock:
            if not self._write_status_if_owner():
                return EXIT_SUPERSEDED

        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

        if threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGTERM"):
            def _on_term(signum, frame):  # noqa: ARG001
                self.terminated_by_signal = True
                threading.Thread(target=self._stop_child, daemon=True).start()
            try:
                signal.signal(signal.SIGTERM, _on_term)
            except (ValueError, OSError):
                pass

        proc = self._spawn_child()
        self.proc = proc
        self._touch(child_pid=proc.pid)
        if self.superseded or self.terminated_by_signal:
            self._stop_child()

        assert proc.stdout is not None
        self.reader = threading.Thread(target=self._read_output, args=(proc,), daemon=True)
        self.reader.start()
        # Wait for the child itself; the reader may outlive it if a grandchild holds the pipe.
        returncode = proc.wait()
        self.reader.join(timeout=2)   # a surviving pipe holder must not block the wrapper

        if self.superseded:
            self.stop_hb.set()
            return EXIT_SUPERSEDED
        if self.timed_out:
            return self._finish(False, f"Timed out after {self.max_runtime_s} s", returncode)
        if self.terminated_by_signal:
            return self._finish(False, "Wrapper terminated by signal", returncode)
        if returncode != 0:
            return self._finish(False, f"Pipeline exited with code {returncode}", returncode)
        return self._finish(True, None, returncode)

    def fail_unexpected(self, exc: BaseException) -> int:
        """Best-effort failure path for an unexpected wrapper exception."""
        self._stop_child()
        if not self.status:
            return EXIT_FAILED
        return self._finish(False, f"Wrapper error: {type(exc).__name__}: {exc}", None)

    def close(self) -> None:
        self.stop_hb.set()
        if self.log_fh is not None:
            try:
                self.log_fh.close()
            except OSError:
                pass


def run_job(args: argparse.Namespace) -> int:
    """Run one background pipeline job. 0 ok, 1 failed, 2 not owner or bad args, 3 superseded."""
    if args.reuse_phase1 and args.mode != "full":
        return EXIT_BAD
    paths = get_run_paths(args.data_dir)
    lock = read_lock(paths)
    if lock is None or lock.run_id != args.run_id:
        return EXIT_BAD
    job = _Job(args, paths)
    try:
        return job.run()
    except Exception as exc:
        logger.exception("Wrapper error")
        try:
            return job.fail_unexpected(exc)
        except Exception:
            return EXIT_FAILED
    finally:
        job.close()


def main() -> None:
    sys.exit(run_job(parse_args()))


if __name__ == "__main__":
    main()
