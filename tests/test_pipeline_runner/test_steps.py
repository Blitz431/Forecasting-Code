"""Tests for src/pipeline_runner/steps.py (plan section 11, items 1-8)."""

from __future__ import annotations

import pytest

from src.pipeline_runner.steps import (
    PHASE1_FALLBACK_MARKER,
    PHASE1_REUSE_MARKER,
    PIPELINE_STEPS_FULL,
    PIPELINE_STEPS_PHASE1,
    PIPELINE_STEPS_PHASE2_ONLY,
    advance_step,
    progress_fraction,
    steps_for_run,
)

PHASE1_LINES = [
    "[Phase 1] Fast-ranking 500 tickers (no ML) ...",
    "[Phase 1] Done in 12s -- pool: top 100 candidates",
    "[Phase 1 Only] Collecting price targets ...",
    "[Cache] Ranking cache saved -> x",
    "  AutoStockAnalyzer -- Two-Phase Pipeline Results",
]

FULL_LINES = [
    "[Phase 1] Fast-ranking 500 tickers (no ML) ...",
    "[Phase 1] Done in 12s -- pool: top 100 candidates",
    "[Phase 2] Training ML models on 100 candidates ...",
    "[Phase 2] ML training done in 400s",
    "[Phase 2] Re-ranking top 100 with ML signal ...",
    "[Phase 2] Re-rank done in 9s",
    "[Output] Collecting price targets …",
    "[Cache] Ranking cache saved -> x",
    "  AutoStockAnalyzer -- Two-Phase Pipeline Results",
]

PHASE2_ONLY_LINES = [
    f"[Phase 1] {PHASE1_REUSE_MARKER} from 2026-10-11T15:00:00+00:00 (498 ranked) -- skipping Phase 1, pool: top 100 candidates",
    "[Phase 2] Training ML models on 100 candidates ...",
    "[Phase 2] ML training done in 400s",
    "[Phase 2] Re-ranking top 100 with ML signal ...",
    "[Phase 2] Re-rank done in 9s",
    "[Output] Collecting price targets …",
    "[Cache] Ranking cache saved -> x",
    "  AutoStockAnalyzer -- Two-Phase Pipeline Results",
]


def _walk(lines, steps, start=0):
    idx = start
    seen = []
    for ln in lines:
        idx = advance_step(ln, steps, idx)
        seen.append(idx)
    return seen


def test_advance_step_matches_in_order():
    steps = PIPELINE_STEPS_FULL
    assert advance_step("[Phase 1] Done in 3s", steps, 0) == 1
    assert advance_step("[Phase 2] Training ML models on 5", steps, 1) == 2


def test_advance_step_never_goes_backwards():
    steps = PIPELINE_STEPS_FULL
    assert advance_step("[Phase 1] Fast-ranking x", steps, 5) == 5
    assert advance_step("[Phase 1] Done in 3s", steps, 1) == 1  # same index is not "later"


def test_advance_step_can_skip_ahead():
    assert advance_step("[Phase 2] Re-ranking top 10", PIPELINE_STEPS_FULL, 0) == 4


def test_advance_step_non_match_returns_current():
    assert advance_step("nothing interesting", PIPELINE_STEPS_FULL, 3) == 3
    assert advance_step("", PIPELINE_STEPS_FULL, 0) == 0


def test_advance_step_is_case_sensitive():
    assert advance_step("[phase 1] done", PIPELINE_STEPS_FULL, 0) == 0


def test_phase1_real_lines_reach_last_index():
    seen = _walk(PHASE1_LINES, PIPELINE_STEPS_PHASE1)
    assert seen == [0, 1, 2, 3, 4]


def test_full_sequence_including_ellipsis_reaches_index_8():
    seen = _walk(FULL_LINES, PIPELINE_STEPS_FULL)
    assert seen[-1] == 8
    assert seen == [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_phase2_only_sequence_reaches_index_7():
    seen = _walk(PHASE2_ONLY_LINES, PIPELINE_STEPS_PHASE2_ONLY)
    assert seen == [0, 1, 2, 3, 4, 5, 6, 7]


def test_phase2_only_markers_match_full_tail():
    assert len(PIPELINE_STEPS_PHASE2_ONLY) == 8
    assert [m for m, _ in PIPELINE_STEPS_PHASE2_ONLY[1:]] == [m for m, _ in PIPELINE_STEPS_FULL[2:]]
    assert PIPELINE_STEPS_PHASE2_ONLY[0][0] == [PHASE1_REUSE_MARKER]


@pytest.mark.parametrize("mode,reused,expected", [
    ("phase1", False, PIPELINE_STEPS_PHASE1),
    ("phase1", True, PIPELINE_STEPS_PHASE1),    # reuse flag ignored for phase1
    ("full", False, PIPELINE_STEPS_FULL),
    ("full", True, PIPELINE_STEPS_PHASE2_ONLY),
])
def test_steps_for_run(mode, reused, expected):
    assert steps_for_run(mode, reused) is expected


def test_steps_for_run_default_not_reused():
    assert steps_for_run("full") is PIPELINE_STEPS_FULL


@pytest.mark.parametrize("bad", ["", "FULL", "phase2", None])
def test_steps_for_run_bad_mode(bad):
    with pytest.raises(ValueError):
        steps_for_run(bad)


def test_progress_fraction_edges():
    assert progress_fraction(8, 9, "done") == 1.0
    assert progress_fraction(0, 9, "done") == 1.0
    assert progress_fraction(3, 1, "running") == 0.0
    assert progress_fraction(3, 0, "running") == 0.0
    assert progress_fraction(0, 9, "running") == 0.0
    assert progress_fraction(4, 9, "running") == pytest.approx(0.5)
    assert progress_fraction(8, 9, "running") == pytest.approx(0.99)   # capped until done
    assert progress_fraction(99, 9, "running") == pytest.approx(0.99)
    assert progress_fraction(-5, 9, "running") == 0.0
    assert 0.0 <= progress_fraction(2, 9, "failed") <= 1.0


def test_tables_equal_old_app_lists():
    expected_phase1 = [
        (["Fast-ranking"], "Phase 1: Fast-ranking all tickers with indicators..."),
        (["[Phase 1] Done"], "Phase 1 complete. Collecting prices..."),
        (["Collecting price", "Phase 1 Only]"], "Collecting price data..."),
        (["Ranking cache saved"], "Saving ranking cache..."),
        (["AutoStockAnalyzer --"], "Building results table..."),
    ]
    expected_full_markers = [
        ["Fast-ranking"], ["[Phase 1] Done"], ["Training ML models"], ["ML training done"],
        ["Re-ranking top"], ["Re-rank done"], ["Collecting price", "[Output]"],
        ["Ranking cache saved"], ["AutoStockAnalyzer --"],
    ]
    assert PIPELINE_STEPS_PHASE1 == expected_phase1
    assert [m for m, _ in PIPELINE_STEPS_FULL] == expected_full_markers
    assert len(PIPELINE_STEPS_FULL) == 9
    assert all(isinstance(lbl, str) and lbl for _, lbl in PIPELINE_STEPS_FULL)


def test_reuse_and_fallback_markers_do_not_hit_other_markers():
    other = [m for m, _ in PIPELINE_STEPS_PHASE1] + [m for m, _ in PIPELINE_STEPS_FULL] \
        + [m for m, _ in PIPELINE_STEPS_PHASE2_ONLY[1:]]
    for const in (PHASE1_REUSE_MARKER, PHASE1_FALLBACK_MARKER):
        for markers in other:
            for m in markers:
                assert m not in const, f"{m!r} found in {const!r}"
    assert PHASE1_REUSE_MARKER not in PHASE1_FALLBACK_MARKER
    assert PHASE1_FALLBACK_MARKER not in PHASE1_REUSE_MARKER
