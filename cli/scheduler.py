"""Daily cycle automation runner.

Usage
-----
    python cli/scheduler.py --start
        Enter the APScheduler loop and block until Ctrl-C.

    python cli/scheduler.py --run-now
        Execute the full pipeline immediately, ignoring the clock.
        Useful for manual testing without waiting for scheduled times.

    python cli/scheduler.py --status
        Print the next scheduled run time for each job.

Pipeline (all times UTC)
------------------------
    06:00  Step 1 — Incremental scrape     cli/scrape.py
    06:15  Step 2 — Indicators             cli/indicators.py --all
    06:30  Step 3 — Forecasts + ML + News  cli/forecast.py --all
                                           cli/ml.py --all --predict
                                           cli/news.py --premarket
    07:00  Step 4 — Morning report         cli/report.py --morning
                  + Alert dispatch         cli/report.py --alerts
    16:00  Step 5 — EOD snapshot           cli/trade.py --mode paper --dry-run

Each step runs as a subprocess.  A failure in one step does NOT abort the rest.
All stdout/stderr is streamed to the terminal and appended to
data/logs/scheduler_YYYY-MM-DD.log.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Project root so subprocess commands resolve regardless of working directory
ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / "data" / "logs"


# ---------------------------------------------------------------------------#
# Log helpers
# ---------------------------------------------------------------------------#

def _daily_log_path() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"scheduler_{date.today().isoformat()}.log"


def _log(msg: str, log_file: Path) -> None:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    try:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------#
# Subprocess runner
# ---------------------------------------------------------------------------#

def _run_step(label: str, cmd: list[str], log_file: Path) -> bool:
    """Run a subprocess and tee output to log_file.

    Returns True on success (exit 0).  Exceptions are caught and logged so
    a broken step never aborts downstream steps in the pipeline.
    """
    _log(f">>> START  {label}", log_file)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            encoding="utf-8",
            errors="replace",
        )
        with log_file.open("a", encoding="utf-8") as lf:
            for line in proc.stdout:  # type: ignore[union-attr]
                print(line, end="", flush=True)
                lf.write(line)
        rc = proc.wait()
    except Exception as exc:
        _log(f"!!! ERROR  {label}: {exc}", log_file)
        return False

    if rc == 0:
        _log(f"<<< OK     {label} (exit 0)", log_file)
    else:
        _log(f"<<< FAILED {label} (exit {rc}) — continuing pipeline", log_file)
    return rc == 0


# ---------------------------------------------------------------------------#
# Job functions
# ---------------------------------------------------------------------------#

def job_scrape() -> None:
    """6:00 AM — incremental price / macro / dividend scrape."""
    log = _daily_log_path()
    _log("=== JOB: scrape ===", log)
    _run_step("scrape", [sys.executable, str(ROOT / "cli" / "scrape.py")], log)


def job_indicators() -> None:
    """6:15 AM — technical indicators for all stored tickers."""
    log = _daily_log_path()
    _log("=== JOB: indicators ===", log)
    _run_step(
        "indicators",
        [sys.executable, str(ROOT / "cli" / "indicators.py"), "--all"],
        log,
    )


def job_analysis() -> None:
    """6:30 AM — forecasts + ML predictions + premarket news.

    Three subprocesses run sequentially; a failure in any one is logged but
    the remaining two still execute.
    """
    log = _daily_log_path()
    _log("=== JOB: analysis ===", log)
    _run_step(
        "forecast",
        [sys.executable, str(ROOT / "cli" / "forecast.py"), "--all"],
        log,
    )
    _run_step(
        "ml-predict",
        [sys.executable, str(ROOT / "cli" / "ml.py"), "--all", "--predict"],
        log,
    )
    _run_step(
        "news",
        [sys.executable, str(ROOT / "cli" / "news.py"), "--premarket"],
        log,
    )


def job_report() -> None:
    """7:00 AM — generate morning PDF/Excel report then dispatch alerts."""
    log = _daily_log_path()
    _log("=== JOB: report ===", log)
    _run_step(
        "report-morning",
        [sys.executable, str(ROOT / "cli" / "report.py"), "--morning"],
        log,
    )
    _run_step(
        "report-alerts",
        [sys.executable, str(ROOT / "cli" / "report.py"), "--alerts"],
        log,
    )


def job_eod() -> None:
    """4:00 PM — EOD portfolio snapshot.

    Uses --dry-run so the portfolio tracker saves the snapshot without
    placing any new orders.
    """
    log = _daily_log_path()
    _log("=== JOB: eod-snapshot ===", log)
    _run_step(
        "eod-snapshot",
        [sys.executable, str(ROOT / "cli" / "trade.py"), "--mode", "paper", "--dry-run"],
        log,
    )


# ---------------------------------------------------------------------------#
# Job registry — (name, cron kwargs, function)
# ---------------------------------------------------------------------------#

_JOB_REGISTRY: list[tuple[str, dict, object]] = [
    ("scrape",     {"hour": 6,  "minute": 0},  job_scrape),
    ("indicators", {"hour": 6,  "minute": 15}, job_indicators),
    ("analysis",   {"hour": 6,  "minute": 30}, job_analysis),
    ("report",     {"hour": 7,  "minute": 0},  job_report),
    ("eod",        {"hour": 16, "minute": 0},  job_eod),
]


# ---------------------------------------------------------------------------#
# Sub-commands
# ---------------------------------------------------------------------------#

def cmd_start() -> None:
    """Enter the APScheduler blocking loop."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone="UTC")

    print("\n[scheduler] Registering jobs:")
    for name, cron_kwargs, fn in _JOB_REGISTRY:
        scheduler.add_job(fn, CronTrigger(timezone="UTC", **cron_kwargs), id=name)
        print(f"  {name:<14} @ {cron_kwargs['hour']:02d}:{cron_kwargs['minute']:02d} UTC")

    log = _daily_log_path()
    _log("[scheduler] APScheduler loop started", log)
    print(f"\n[scheduler] Running. Log: {log}\nPress Ctrl-C to stop.\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        _log("[scheduler] Stopped by user", log)
        print("\n[scheduler] Stopped.")


def cmd_run_now() -> None:
    """Execute the full pipeline immediately, ignoring the clock."""
    log = _daily_log_path()
    _log("[scheduler] --run-now: executing full pipeline immediately", log)
    print(f"\n[scheduler] Running full pipeline now.\nLog: {log}\n")

    for name, _, fn in _JOB_REGISTRY:
        try:
            fn()  # type: ignore[operator]
        except Exception as exc:
            _log(f"!!! UNCAUGHT in job '{name}': {exc}", log)

    _log("[scheduler] --run-now complete", log)
    print("\n[scheduler] Full pipeline complete.")


def cmd_status() -> None:
    """Print next scheduled run times for each job."""
    from apscheduler.triggers.cron import CronTrigger

    now = datetime.now(tz=timezone.utc)
    print(f"\nScheduler status — {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    print(f"  {'Job':<14}  {'Time (UTC)':<12}  Next Scheduled Run")
    print("  " + "-" * 58)
    for name, cron_kwargs, _ in _JOB_REGISTRY:
        trigger = CronTrigger(timezone="UTC", **cron_kwargs)
        next_fire = trigger.get_next_fire_time(None, now)
        cron_str = f"{cron_kwargs['hour']:02d}:{cron_kwargs['minute']:02d}"
        next_str = next_fire.strftime("%Y-%m-%d %H:%M UTC") if next_fire else "N/A"
        print(f"  {name:<14}  {cron_str:<12}  {next_str}")
    print()


# ---------------------------------------------------------------------------#
# Entry point
# ---------------------------------------------------------------------------#

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoStockAnalyzer — daily cycle scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--start",
        action="store_true",
        help="Enter APScheduler loop and block (Ctrl-C to stop)",
    )
    group.add_argument(
        "--run-now",
        action="store_true",
        dest="run_now",
        help="Execute the full pipeline immediately (ignoring the clock)",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Print next scheduled run time for each job",
    )

    args = parser.parse_args()

    if args.start:
        cmd_start()
    elif args.run_now:
        cmd_run_now()
    elif args.status:
        cmd_status()


if __name__ == "__main__":
    main()
