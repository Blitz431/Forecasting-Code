"""
Purpose: Shows a thin fixed-position bar while the ranking pipeline runs in the background
         (step x/N, percent, elapsed time, a CSS pulse so ML training does not look frozen),
         then a short done/failed state. On each page load it also triggers the automatic
         daily run if one is due. When idle it renders nothing and does not poll.

Connections:
  - src/pipeline_runner/launcher.py: maybe_start_daily_run
  - src/pipeline_runner/run_state.py: get_run_view, read_status, read_day_state
  - Called at the top of every page in dashboard/app.py and dashboard/pages/*.py

In:  status/lock files under data/run_cache (written by the background wrapper)
Out: an HTML bar (no return value); clears st.cache_data once per successful run
"""

from __future__ import annotations

import html
import threading
from datetime import datetime, timezone

import streamlit as st

from src.pipeline_runner.launcher import maybe_start_daily_run
from src.pipeline_runner.run_state import (
    RunView,
    get_run_paths,
    get_run_view,
    read_day_state,
    read_status,
)
from src.utils.logging import setup_logger

POLL_INTERVAL = "2s"
_STEP_SECONDS_SHOWN_AFTER = 60
_ERROR_MAX_CHARS = 140

logger = setup_logger("dashboard.pipeline_progress")

_CLEARED_RUN_IDS: set[str] = set()
_CLEAR_LOCK = threading.Lock()


# --------------------------------------------------------------------------- pure helpers


def _fmt_elapsed(seconds: float) -> str:
    """Format seconds as "45s", "7m 12s" or "1h 03m"."""
    try:
        total = max(int(seconds), 0)
    except (TypeError, ValueError, OverflowError):
        total = 0
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m {total % 60}s"
    return f"{total // 3600}h {(total % 3600) // 60:02d}m"


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _since(value: str | None, now: datetime) -> float | None:
    dt = _parse_dt(value)
    if dt is None:
        return None
    return max((now - dt).total_seconds(), 0.0)


def _prefix(view: RunView) -> str:
    if view.trigger == "auto":
        base = "Daily full analysis"
    elif view.trigger == "auto_fallback":
        base = "Fallback fast rank (full analysis failed)"
    elif view.mode == "phase1":
        base = "Fast rank"
    else:
        base = "Full analysis"
    if view.mode == "full" and view.phase1_reused:
        base += " (ML only, reusing today's Phase 1)"
    return base


def _first_line(text: str | None, limit: int = _ERROR_MAX_CHARS) -> str:
    if not text:
        return "unknown error"
    for line in str(text).splitlines():
        line = line.strip()
        if line:
            return line[:limit]
    return "unknown error"


_CSS = """
.asa-plbar{position:fixed;top:3.75rem;left:50%;transform:translateX(-50%);
width:min(560px,92vw);z-index:999991;pointer-events:none;border-radius:6px;
padding:6px 12px;font-size:0.85rem;line-height:1.3;box-sizing:border-box;
background:rgba(20,24,33,0.92);color:#f1f3f5;border:1px solid rgba(255,255,255,0.18);
border-left:4px solid #4dabf7;box-shadow:0 2px 8px rgba(0,0,0,0.35);}
.asa-plbar.asa-done{border-left-color:#40c057;}
.asa-plbar.asa-failed{border-left-color:#fa5252;}
.asa-plbar .asa-sub{color:#adb5bd;font-size:0.75rem;margin-top:2px;}
.asa-plbar .asa-track{height:5px;margin-top:5px;border-radius:3px;
background:rgba(255,255,255,0.18);overflow:hidden;}
.asa-plbar .asa-fill{height:100%;border-radius:3px;background:#4dabf7;}
.asa-plbar.asa-done .asa-fill{background:#40c057;}
.asa-plbar.asa-failed .asa-fill{background:#fa5252;}
.asa-plbar.asa-running .asa-fill{background-image:linear-gradient(45deg,
rgba(255,255,255,0.28) 25%,transparent 25%,transparent 50%,rgba(255,255,255,0.28) 50%,
rgba(255,255,255,0.28) 75%,transparent 75%);background-size:16px 16px;
animation:asa-stripe 0.8s linear infinite,asa-pulse 1.6s ease-in-out infinite;}
@keyframes asa-stripe{from{background-position:0 0;}to{background-position:16px 0;}}
@keyframes asa-pulse{0%,100%{opacity:1;}50%{opacity:0.6;}}
"""


def _bar_html(view: RunView, now: datetime | None = None, fallback_pending: bool = False) -> str:
    """Build the bar HTML. Pure; every dynamic string is escaped with html.escape."""
    now = now or datetime.now(timezone.utc)
    prefix = _prefix(view)
    pct = int(round(min(max(view.progress or 0.0, 0.0), 1.0) * 100))
    sub = ""

    if view.state == "done":
        cls = "asa-done"
        text = "Ranking updated - refreshing data"
    elif view.state == "failed":
        cls = "asa-failed"
        text = f"{prefix} failed: {_first_line(view.error)}. See Settings page."
        if fallback_pending and view.trigger == "auto" and view.mode == "full":
            text += " A fast rank will run instead."
    else:
        cls = "asa-running"
        total = max(view.step_total, 1)
        step = min(max(view.step_index, 0) + 1, total)
        elapsed = _since(view.started_at, now)
        text = f"{prefix}: {view.label} - step {step}/{total} - {pct}%"
        if elapsed is not None:
            text += f" - {_fmt_elapsed(elapsed)} elapsed"
        step_elapsed = _since(view.step_started_at, now)
        if step_elapsed is not None and step_elapsed >= _STEP_SECONDS_SHOWN_AFTER:
            text += f" (this step {_fmt_elapsed(step_elapsed)})"
        if view.trigger == "auto":
            sub = '<div class="asa-sub">You can keep using the dashboard.</div>'

    return (
        f"<style>{_CSS}</style>"
        f'<div class="asa-plbar {cls}">'
        f"<div>{html.escape(text)}</div>"
        f'<div class="asa-track"><div class="asa-fill" style="width:{pct}%"></div></div>'
        f"{sub}</div>"
    )


# --------------------------------------------------------------------------- streamlit parts


def _maybe_clear_caches_for(run_id: str) -> None:
    """Clear st.cache_data once per server process per run_id."""
    if not run_id:
        return
    with _CLEAR_LOCK:
        if run_id in _CLEARED_RUN_IDS:
            return
        _CLEARED_RUN_IDS.add(run_id)
        try:
            st.cache_data.clear()
        except Exception as exc:
            logger.warning("cache clear failed: %s", exc)


@st.fragment(run_every=POLL_INTERVAL)
def _progress_fragment() -> None:
    try:
        paths = get_run_paths()
        view = get_run_view(paths)
        if view.state != "idle" and view.run_id:
            st.session_state["_plbar_seen_run_id"] = view.run_id
        if view.state == "done" and view.run_id:
            _maybe_clear_caches_for(view.run_id)
        if view.state == "idle":
            seen = st.session_state.get("_plbar_seen_run_id")
            if seen and st.session_state.get("_plbar_rerun_done_for") != seen:
                st.session_state["_plbar_rerun_done_for"] = seen
                needs_rerun = True
            else:
                needs_rerun = False
        else:
            needs_rerun = False
            fallback_pending = False
            if view.state == "failed" and view.trigger == "auto" and view.mode == "full":
                try:
                    fallback_pending = not read_day_state(paths).fallback_started
                except Exception:
                    fallback_pending = False
            st.markdown(_bar_html(view, fallback_pending=fallback_pending), unsafe_allow_html=True)
    except Exception as exc:
        logger.warning("progress fragment failed: %s", exc)
        return
    if needs_rerun:
        st.rerun(scope="app")


def render_pipeline_progress() -> None:
    """Auto-start hook plus the floating progress bar. Never raises into a page."""
    try:
        try:
            res = maybe_start_daily_run()
            st.session_state["_plbar_last_start_reason"] = res.reason
        except Exception as exc:
            logger.warning("maybe_start_daily_run failed: %s", exc)
        paths = get_run_paths()
        status = read_status(paths)
        if status and status.get("state") == "done":
            run_id = status.get("run_id")
            if isinstance(run_id, str) and run_id not in _CLEARED_RUN_IDS:
                _maybe_clear_caches_for(run_id)
        view = get_run_view(paths)
        if view.state == "idle":
            return
        _progress_fragment()
    except Exception as exc:
        logger.warning("render_pipeline_progress failed: %s", exc)
