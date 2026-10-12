"""
Purpose: Shared pipeline step tables, stdout marker constants and step-advance parsing.

Connections:
  - cli/pipeline.py: prints lines containing these markers; imports the PHASE1_* constants
  - cli/pipeline_job.py: parses pipeline stdout with advance_step to track progress
  - dashboard/app.py: step tables and _advance_step were moved here from this file
  - src/pipeline_runner/run_state.py: uses the same RunMode / RunState values

In:  a stdout line (str), a step table, the current step index, run mode and state
Out: step tables, next step index, progress fraction in [0, 1]
"""

from typing import Literal

# (substring markers, user-facing label)
Step = tuple[list[str], str]

# Local copies of the run_state aliases so this module has no dependency on run_state.
RunMode = Literal["phase1", "full"]
RunState = Literal["running", "done", "failed"]

# Exact stdout strings shared with cli/pipeline.py. pipeline.py imports these
# constants rather than retyping them.
PHASE1_REUSE_MARKER = "Reusing saved Phase 1 result"
PHASE1_FALLBACK_MARKER = "Saved Phase 1 result not usable"

PIPELINE_STEPS_PHASE1: list[Step] = [
    (["Fast-ranking"],                        "Phase 1: Fast-ranking all tickers with indicators..."),
    (["[Phase 1] Done"],                      "Phase 1 complete. Collecting prices..."),
    (["Collecting price", "Phase 1 Only]"],   "Collecting price data..."),
    (["Ranking cache saved"],                 "Saving ranking cache..."),
    (["AutoStockAnalyzer --"],                "Building results table..."),
]

PIPELINE_STEPS_FULL: list[Step] = [
    (["Fast-ranking"],            "Phase 1: Fast-ranking all tickers with indicators..."),
    (["[Phase 1] Done"],          "Phase 1 complete — starting ML training..."),
    (["Training ML models"],      "Phase 2: Training ML models on top candidates..."),
    (["ML training done"],        "ML training done — re-ranking with ML signal..."),
    (["Re-ranking top"],          "Phase 2: Re-ranking all candidates with ML signal..."),
    (["Re-rank done"],            "Re-ranking done — collecting price targets..."),
    (["Collecting price", "[Output]"], "Collecting price targets..."),
    (["Ranking cache saved"],     "Saving ranking cache..."),
    (["AutoStockAnalyzer --"],    "Building results table..."),
]

# Used when a full run reuses today's saved Phase 1 result (Phase 2 only).
PIPELINE_STEPS_PHASE2_ONLY: list[Step] = [
    ([PHASE1_REUSE_MARKER],       "Loading today's Phase 1 result..."),
    (["Training ML models"],      "Phase 2: Training ML models on top candidates..."),
    (["ML training done"],        "ML training done — re-ranking with ML signal..."),
    (["Re-ranking top"],          "Phase 2: Re-ranking all candidates with ML signal..."),
    (["Re-rank done"],            "Re-ranking done — collecting price targets..."),
    (["Collecting price", "[Output]"], "Collecting price targets..."),
    (["Ranking cache saved"],     "Saving ranking cache..."),
    (["AutoStockAnalyzer --"],    "Building results table..."),
]


def steps_for_run(mode: RunMode, phase1_reused: bool = False) -> list[Step]:
    """Return the step table for a run.

    Args:
        mode: "phase1" or "full".
        phase1_reused: Full runs only; True when Phase 1 is skipped.

    Returns:
        The matching step table.

    Raises:
        ValueError: If mode is unknown.
    """
    if mode == "phase1":
        return PIPELINE_STEPS_PHASE1
    if mode == "full":
        return PIPELINE_STEPS_PHASE2_ONLY if phase1_reused else PIPELINE_STEPS_FULL
    raise ValueError(f"Unknown run mode: {mode!r}")


def advance_step(line: str, steps: list[Step], current: int) -> int:
    """Return the next step index if a later step's marker appears in line.

    Never goes backwards. Matching is case-sensitive substring matching.

    Args:
        line: One line of pipeline stdout.
        steps: Step table in use.
        current: Current 0-based step index.

    Returns:
        The first later matching index, else current.
    """
    for i in range(current + 1, len(steps)):
        if any(m in line for m in steps[i][0]):
            return i
    return current


def progress_fraction(step_index: int, step_total: int, state: RunState) -> float:
    """Return progress in [0, 1]; 1.0 only when the run is done.

    Args:
        step_index: Current 0-based step index.
        step_total: Number of steps in the table.
        state: "running", "done" or "failed".

    Returns:
        Progress fraction, capped at 0.99 until done.
    """
    if state == "done":
        return 1.0
    if step_total <= 1:
        return 0.0
    limit = 10 ** 9   # clamp so int/int true division cannot overflow on absurd values
    step_index = max(-limit, min(step_index, limit))
    step_total = max(-limit, min(step_total, limit))
    if step_total <= 1:
        return 0.0
    frac = min(step_index / (step_total - 1), 0.99)
    return max(0.0, min(frac, 1.0))
