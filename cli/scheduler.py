from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Project root so subprocess commands resolve regardless of working directory
ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / "data" / "logs"

# Set by main() when --force is passed; bypasses the "already ran today" guard.
_FORCE_RUN = False

"""
Purpose: Daily automation loop — APScheduler cron jobs that run the full pipeline (scrape → indicators → forecast/ML/news → report → EOD snapshot).

Connections:
  - cli/scrape.py: job_scrape() at 06:00 UTC
  - cli/indicators.py: job_indicators() at 06:15 UTC
  - cli/forecast.py, cli/ml.py, cli/news.py: job_analysis() at 06:30 UTC
  - cli/report.py: job_report() at 07:00 UTC (morning report + alerts)
  - cli/trade.py: job_eod() at 16:00 UTC (EOD snapshot)

In:  system clock (APScheduler triggers); no data inputs directly
Out: data/logs/scheduler_YYYY-MM-DD.log; launches all pipeline subprocesses
"""


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
# Daily "already ran today" guard
# ---------------------------------------------------------------------------#

def _utc_today() -> str:
    """UTC date string — matches the cron trigger timezone, not local time."""
    return datetime.now(tz=timezone.utc).date().isoformat()


def _marker_path(job_name: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f".done_{job_name}_{_utc_today()}"


def _already_done_today(job_name: str) -> bool:
    """True if this daily job already completed successfully today (UTC)."""
    if _FORCE_RUN:
        return False
    return _marker_path(job_name).exists()


def _mark_done_today(job_name: str) -> None:
    """Write the completion marker for today (UTC). Never raises."""
    try:
        _marker_path(job_name).write_text(
            datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") + "\n",
            encoding="utf-8",
        )
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
    if _already_done_today("scrape"):
        _log("=== JOB: scrape — already ran today, skipping ===", log)
        return
    _log("=== JOB: scrape ===", log)
    ok = _run_step("scrape", [sys.executable, str(ROOT / "cli" / "scrape.py")], log)
    if ok:
        _mark_done_today("scrape")


def job_indicators() -> None:
    """6:15 AM — technical indicators for all stored tickers."""
    log = _daily_log_path()
    if _already_done_today("indicators"):
        _log("=== JOB: indicators — already ran today, skipping ===", log)
        return
    _log("=== JOB: indicators ===", log)
    ok = _run_step(
        "indicators",
        [sys.executable, str(ROOT / "cli" / "indicators.py"), "--all"],
        log,
    )
    if ok:
        _mark_done_today("indicators")


def job_analysis() -> None:
    """6:30 AM — forecasts + ML predictions + premarket news.

    Three subprocesses run sequentially; a failure in any one is logged but
    the remaining two still execute.
    """
    log = _daily_log_path()
    if _already_done_today("analysis"):
        _log("=== JOB: analysis — already ran today, skipping ===", log)
        return
    _log("=== JOB: analysis ===", log)
    ok_forecast = _run_step(
        "forecast",
        [sys.executable, str(ROOT / "cli" / "forecast.py"), "--all"],
        log,
    )
    ok_ml = _run_step(
        "ml-predict",
        [sys.executable, str(ROOT / "cli" / "ml.py"), "--all", "--predict"],
        log,
    )
    ok_news = _run_step(
        "news",
        [sys.executable, str(ROOT / "cli" / "news.py"), "--premarket"],
        log,
    )
    if ok_forecast and ok_ml and ok_news:
        _mark_done_today("analysis")


def job_report() -> None:
    """7:00 AM — generate morning PDF/Excel report then dispatch alerts."""
    log = _daily_log_path()
    if _already_done_today("report"):
        _log("=== JOB: report — already ran today, skipping ===", log)
        return
    _log("=== JOB: report ===", log)
    ok_morning = _run_step(
        "report-morning",
        [sys.executable, str(ROOT / "cli" / "report.py"), "--morning"],
        log,
    )
    ok_alerts = _run_step(
        "report-alerts",
        [sys.executable, str(ROOT / "cli" / "report.py"), "--alerts"],
        log,
    )
    if ok_morning and ok_alerts:
        _mark_done_today("report")


def job_reconcile_stops() -> None:
    """Every 5 minutes — ensure every held Alpaca position has an active trailing stop.

    Does not place any new buy/sell trades; only manages stops on positions already held.
    """
    log = _daily_log_path()
    _log("=== JOB: reconcile-stops ===", log)
    try:
        sys.path.insert(0, str(ROOT))
        from config.settings import get_settings
        from src.trading.alpaca_client import AlpacaClient
        from src.trading.stop_manager import reconcile_trailing_stops

        settings = get_settings()
        client = AlpacaClient(settings)
        if not client.connected:
            _log("reconcile-stops: Alpaca not connected — skipping", log)
            return
        attached = reconcile_trailing_stops(client, settings)
        _log(f"reconcile-stops: attached {len(attached)} new stop(s): {attached}", log)
    except Exception as exc:
        _log(f"!!! ERROR  reconcile-stops: {exc}", log)


def job_live_quotes() -> None:
    """Every 5 minutes — refresh live Alpaca quotes for held + watchlisted tickers."""
    log = _daily_log_path()
    _log("=== JOB: live-quotes ===", log)
    try:
        sys.path.insert(0, str(ROOT))
        from src.scraper.live_quotes import refresh_quote_cache
        quotes = refresh_quote_cache()
        _log(f"live-quotes: refreshed {len(quotes)} tickers", log)
    except Exception as exc:
        _log(f"!!! ERROR  live-quotes: {exc}", log)


def job_eod() -> None:
    """4:00 PM — EOD portfolio snapshot.

    Uses --dry-run so the portfolio tracker saves the snapshot without
    placing any new orders.
    """
    log = _daily_log_path()
    if _already_done_today("eod"):
        _log("=== JOB: eod-snapshot — already ran today, skipping ===", log)
        return
    _log("=== JOB: eod-snapshot ===", log)
    ok = _run_step(
        "eod-snapshot",
        [sys.executable, str(ROOT / "cli" / "trade.py"), "--mode", "paper", "--dry-run"],
        log,
    )
    if ok:
        _mark_done_today("eod")


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

# Interval-based jobs (run continuously on a fixed cadence, not at a daily clock time)
_INTERVAL_JOB_REGISTRY: list[tuple[str, dict, object]] = [
    ("live-quotes", {"minutes": 5}, job_live_quotes),
    ("reconcile-stops", {"minutes": 5}, job_reconcile_stops),
]


# ---------------------------------------------------------------------------#
# Sub-commands
# ---------------------------------------------------------------------------#

def cmd_start() -> None:
    """Enter the APScheduler blocking loop."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler = BlockingScheduler(timezone="UTC")

    print("\n[scheduler] Registering jobs:")
    for name, cron_kwargs, fn in _JOB_REGISTRY:
        scheduler.add_job(fn, CronTrigger(timezone="UTC", **cron_kwargs), id=name)
        print(f"  {name:<14} @ {cron_kwargs['hour']:02d}:{cron_kwargs['minute']:02d} UTC")

    for name, interval_kwargs, fn in _INTERVAL_JOB_REGISTRY:
        scheduler.add_job(fn, IntervalTrigger(**interval_kwargs), id=name)
        mins = interval_kwargs.get("minutes", 0)
        print(f"  {name:<14} every {mins} min")

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
    print("  (Jobs already completed today were skipped. Use --force to re-run.)")


def cmd_status() -> None:
    """Print next scheduled run times for each job."""
    from apscheduler.triggers.cron import CronTrigger

    now = datetime.now(tz=timezone.utc)
    print(f"\nScheduler status — {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    print(f"  {'Job':<14}  {'Time (UTC)':<12}  {'Today':<12}  Next Scheduled Run")
    print("  " + "-" * 74)
    for name, cron_kwargs, _ in _JOB_REGISTRY:
        trigger = CronTrigger(timezone="UTC", **cron_kwargs)
        next_fire = trigger.get_next_fire_time(None, now)
        cron_str = f"{cron_kwargs['hour']:02d}:{cron_kwargs['minute']:02d}"
        next_str = next_fire.strftime("%Y-%m-%d %H:%M UTC") if next_fire else "N/A"
        today_str = "done today" if _marker_path(name).exists() else "pending"
        print(f"  {name:<14}  {cron_str:<12}  {today_str:<12}  {next_str}")
    for name, interval_kwargs, _ in _INTERVAL_JOB_REGISTRY:
        mins = interval_kwargs.get("minutes", 0)
        print(f"  {name:<14}  every {mins} min   (only while --start is running)")
    print()


def cmd_live_quotes_once() -> None:
    """Run job_live_quotes() a single time, for manual testing."""
    print("\n[scheduler] Running live-quotes refresh once...\n")
    job_live_quotes()
    print("\n[scheduler] Live-quotes refresh complete.")


def cmd_reconcile_stops_once() -> None:
    """Run job_reconcile_stops() a single time, for manual testing."""
    print("\n[scheduler] Running trailing-stop reconciliation once...\n")
    job_reconcile_stops()
    print("\n[scheduler] Trailing-stop reconciliation complete.")


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
    group.add_argument(
        "--live-quotes-once",
        action="store_true",
        dest="live_quotes_once",
        help="Run the live-quotes refresh job a single time (for testing)",
    )
    group.add_argument(
        "--reconcile-stops-once",
        action="store_true",
        dest="reconcile_stops_once",
        help="Run the trailing-stop reconciliation job a single time (for testing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the 'already ran today' markers and re-run daily jobs anyway",
    )

    args = parser.parse_args()

    global _FORCE_RUN
    _FORCE_RUN = args.force

    if args.start:
        cmd_start()
    elif args.run_now:
        cmd_run_now()
    elif args.status:
        cmd_status()
    elif args.live_quotes_once:
        cmd_live_quotes_once()
    elif args.reconcile_stops_once:
        cmd_reconcile_stops_once()


if __name__ == "__main__":
    main()
