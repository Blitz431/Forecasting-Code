"""Tests for dashboard/components/pipeline_progress.py (plan section 11, items 58-60 plus an AppTest smoke)."""

from __future__ import annotations

import types
from datetime import datetime, timedelta, timezone

import pytest

from dashboard.components import pipeline_progress as pp
from src.pipeline_runner.run_state import DayState, RunView

NOW = datetime(2026, 10, 11, 15, 0, 0, tzinfo=timezone.utc)


def iso(dt):
    return dt.isoformat(timespec="seconds")


def view(**kw) -> RunView:
    base = dict(state="running", run_id="r1", mode="full", trigger="auto", phase1_reused=False,
                label="Phase 2: Training", step_index=2, step_total=9, progress=0.25, error=None,
                started_at=iso(NOW - timedelta(seconds=432)), step_started_at=iso(NOW - timedelta(seconds=5)),
                finished_at=None)
    base.update(kw)
    return RunView(**base)


# ------------------------------------------------------------------ _fmt_elapsed


@pytest.mark.parametrize("seconds,expected", [
    (0, "0s"), (45, "45s"), (59.9, "59s"), (60, "1m 0s"), (432, "7m 12s"), (3599, "59m 59s"),
    (3600, "1h 00m"), (3780, "1h 03m"), (7325, "2h 02m"),
    (-5, "0s"), (None, "0s"), (float("nan"), "0s"), (float("inf"), "0s"), ("x", "0s"),
])
def test_fmt_elapsed(seconds, expected):
    assert pp._fmt_elapsed(seconds) == expected


# ------------------------------------------------------------------ _bar_html


def test_bar_html_escapes_script_in_label():
    out = pp._bar_html(view(label="<script>alert(1)</script>"), NOW)
    assert "<script" not in out and "</script" not in out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out


def test_bar_html_escapes_error_markup():
    out = pp._bar_html(view(state="failed", error="<script>x()</script> and <b>bold</b>"), NOW)
    assert "<script" not in out and "<b>" not in out and "</b>" not in out
    assert "&lt;b&gt;bold&lt;/b&gt;" in out


def test_bar_html_escapes_quotes_and_ampersands():
    out = pp._bar_html(view(label='"a" & \'b\''), NOW)
    assert "&quot;a&quot; &amp; &#x27;b&#x27;" in out


def test_running_full_text():
    out = pp._bar_html(view(), NOW)
    assert "Daily full analysis: Phase 2: Training - step 3/9 - 25%" in out
    assert "7m 12s elapsed" in out
    assert 'class="asa-plbar asa-running"' in out
    assert "width:25%" in out
    assert "You can keep using the dashboard." in out


def test_reused_text_uses_eight_steps_and_ml_only():
    out = pp._bar_html(view(trigger="button", phase1_reused=True, step_index=1, step_total=8), NOW)
    assert "step 2/8" in out and "ML only" in out
    assert out.count("Full analysis (ML only, reusing today&#x27;s Phase 1)") == 1
    assert "You can keep using" not in out     # auto only


def test_phase1_reused_flag_ignored_for_phase1_mode():
    out = pp._bar_html(view(trigger="button", mode="phase1", phase1_reused=True, step_total=5, step_index=0), NOW)
    assert "ML only" not in out and "Fast rank:" in out


def test_auto_fallback_prefix():
    out = pp._bar_html(view(trigger="auto_fallback", mode="phase1", step_total=5, step_index=1), NOW)
    assert "Fallback fast rank (full analysis failed): " in out
    assert "step 2/5" in out
    assert "You can keep using" not in out


def test_button_prefixes():
    assert "Fast rank: " in pp._bar_html(view(trigger="button", mode="phase1"), NOW)
    assert "Full analysis: " in pp._bar_html(view(trigger="button", mode="full"), NOW)


def test_this_step_suffix_threshold():
    just_under = pp._bar_html(view(step_started_at=iso(NOW - timedelta(seconds=59))), NOW)
    at_limit = pp._bar_html(view(step_started_at=iso(NOW - timedelta(seconds=60))), NOW)
    long = pp._bar_html(view(step_started_at=iso(NOW - timedelta(seconds=311))), NOW)
    assert "this step" not in just_under
    assert "(this step 1m 0s)" in at_limit
    assert "(this step 5m 11s)" in long


@pytest.mark.parametrize("started,step_started", [
    ("garbage", ""), (None, None), ("", "not-a-date"), ("2026-13-45T99:00:00", "0000"),
])
def test_junk_timestamps_do_not_crash_and_hide_elapsed(started, step_started):
    out = pp._bar_html(view(started_at=started, step_started_at=step_started), NOW)
    assert "elapsed" not in out and "this step" not in out
    assert "step 3/9" in out


def test_naive_timestamp_treated_as_utc():
    out = pp._bar_html(view(started_at="2026-10-11T14:52:48"), NOW)
    assert "7m 12s elapsed" in out


def test_future_started_at_clamped_to_zero():
    out = pp._bar_html(view(started_at=iso(NOW + timedelta(minutes=5))), NOW)
    assert "0s elapsed" in out


def test_step_clamped_to_total():
    out = pp._bar_html(view(step_index=50, step_total=9), NOW)
    assert "step 9/9" in out
    out0 = pp._bar_html(view(step_index=0, step_total=0), NOW)
    assert "step 1/1" in out0


def test_done_text_and_class():
    out = pp._bar_html(view(state="done", progress=1.0), NOW)
    assert "Ranking updated - refreshing data" in out
    assert "asa-done" in out and "width:100%" in out


def test_failed_text_first_error_line_only_and_truncated():
    out = pp._bar_html(view(state="failed", error="first problem\nsecond line\nthird"), NOW)
    assert "Daily full analysis failed: first problem. See Settings page." in out
    assert "second line" not in out
    out2 = pp._bar_html(view(state="failed", error="x" * 500), NOW)
    assert "x" * 140 in out2 and "x" * 141 not in out2
    assert "asa-failed" in out2


@pytest.mark.parametrize("err", [None, "", "   \n  "])
def test_failed_without_error_text(err):
    assert "failed: unknown error." in pp._bar_html(view(state="failed", error=err), NOW)


def test_fallback_pending_text():
    pend = pp._bar_html(view(state="failed", error="boom"), NOW, fallback_pending=True)
    assert "A fast rank will run instead." in pend
    no = pp._bar_html(view(state="failed", error="boom"), NOW, fallback_pending=False)
    assert "will run instead" not in no
    btn = pp._bar_html(view(state="failed", trigger="button", error="boom"), NOW, fallback_pending=True)
    assert "will run instead" not in btn
    p1 = pp._bar_html(view(state="failed", mode="phase1", error="boom"), NOW, fallback_pending=True)
    assert "will run instead" not in p1


def test_progress_out_of_range_is_clamped():
    assert "width:100%" in pp._bar_html(view(progress=7.0), NOW)
    assert "width:0%" in pp._bar_html(view(progress=-3.0), NOW)
    assert "width:0%" in pp._bar_html(view(progress=None), NOW)


def test_bar_has_no_emoji_or_script_tag():
    out = pp._bar_html(view(), NOW)
    assert "<script" not in out
    assert all(ord(c) < 0x2000 or c in "…" for c in out)


# ------------------------------------------------------------------ _maybe_clear_caches_for


def test_cache_clear_exactly_once_per_run_id(monkeypatch):
    calls = []
    monkeypatch.setattr(pp, "_CLEARED_RUN_IDS", set())
    monkeypatch.setattr(pp.st, "cache_data", types.SimpleNamespace(clear=lambda: calls.append(1)))
    pp._maybe_clear_caches_for("run-a")
    pp._maybe_clear_caches_for("run-a")
    assert len(calls) == 1
    pp._maybe_clear_caches_for("run-b")
    assert len(calls) == 2
    pp._maybe_clear_caches_for("")
    assert len(calls) == 2


def test_cache_clear_failure_is_swallowed_and_not_retried(monkeypatch):
    def boom():
        raise RuntimeError("nope")
    monkeypatch.setattr(pp, "_CLEARED_RUN_IDS", set())
    monkeypatch.setattr(pp.st, "cache_data", types.SimpleNamespace(clear=boom))
    pp._maybe_clear_caches_for("run-x")     # must not raise
    assert "run-x" in pp._CLEARED_RUN_IDS


def test_cache_clear_thread_safe_once(monkeypatch):
    import threading
    calls = []
    monkeypatch.setattr(pp, "_CLEARED_RUN_IDS", set())
    monkeypatch.setattr(pp.st, "cache_data", types.SimpleNamespace(clear=lambda: calls.append(1)))
    barrier = threading.Barrier(8)

    def w():
        barrier.wait()
        pp._maybe_clear_caches_for("same")

    ts = [threading.Thread(target=w) for _ in range(8)]
    [t.start() for t in ts]
    [t.join(10) for t in ts]
    assert len(calls) == 1


# ------------------------------------------------------------------ AppTest smoke


def _app():
    from dashboard.components.pipeline_progress import render_pipeline_progress
    render_pipeline_progress()


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Stub everything render_pipeline_progress touches so no disk state or process is involved."""
    state = types.SimpleNamespace(view=None, started=[], cleared=[], day=DayState("d", 0, False, False),
                                  start_exc=None)

    def fake_start():
        state.started.append(1)
        if state.start_exc:
            raise state.start_exc
        return types.SimpleNamespace(reason="disabled")

    monkeypatch.setattr(pp, "maybe_start_daily_run", fake_start)
    monkeypatch.setattr(pp, "get_run_paths", lambda: object())
    monkeypatch.setattr(pp, "read_status", lambda paths: None)
    monkeypatch.setattr(pp, "get_run_view", lambda paths: state.view)
    monkeypatch.setattr(pp, "read_day_state", lambda paths: state.day)
    monkeypatch.setattr(pp, "_maybe_clear_caches_for", lambda rid: state.cleared.append(rid))
    return state


def _run_app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_app, default_timeout=30)
    at.run()
    return at


def _bars(at):
    return [m.value for m in at.markdown if "asa-plbar" in m.value]


def test_apptest_idle_renders_nothing(stubbed):
    stubbed.view = view(state="idle", run_id=None)
    at = _run_app()
    assert not at.exception
    assert _bars(at) == []
    assert stubbed.started == [1]                    # auto-start hook ran once


def test_apptest_running_renders_bar(stubbed):
    stubbed.view = view(started_at=iso(datetime.now(timezone.utc) - timedelta(seconds=30)),
                        step_started_at=iso(datetime.now(timezone.utc)))
    at = _run_app()
    assert not at.exception
    bars = _bars(at)
    assert len(bars) == 1
    assert "step 3/9" in bars[0] and "Daily full analysis" in bars[0]
    assert at.session_state["_plbar_seen_run_id"] == "r1"
    assert at.session_state["_plbar_last_start_reason"] == "disabled"


def test_apptest_done_renders_bar_and_clears_cache(stubbed):
    stubbed.view = view(state="done", progress=1.0, finished_at=iso(datetime.now(timezone.utc)))
    at = _run_app()
    assert not at.exception
    assert any("Ranking updated" in b for b in _bars(at))
    assert stubbed.cleared == ["r1"]


def test_apptest_failed_auto_full_mentions_fallback(stubbed):
    stubbed.view = view(state="failed", error="kaput\nmore", finished_at=iso(datetime.now(timezone.utc)))
    at = _run_app()
    assert not at.exception
    (bar,) = _bars(at)
    assert "failed: kaput" in bar and "A fast rank will run instead." in bar
    stubbed.day = DayState("d", 1, True, False)
    at2 = _run_app()
    (bar2,) = _bars(at2)
    assert "will run instead" not in bar2


def test_apptest_never_raises_into_page(stubbed):
    stubbed.view = view(state="idle", run_id=None)
    stubbed.start_exc = RuntimeError("launcher exploded")
    at = _run_app()
    assert not at.exception
    assert stubbed.started == [1]
