"""Tests for cli/pipeline.py: Phase 1 artifact write and --skip-phase1 (plan section 11, items 51-57).

All heavy calls (tickers, ranking, ML, price lookups) are stubbed on the cli.pipeline module and
settings point at tmp_path, so nothing touches the network, ML code or the repo's data/.
"""

from __future__ import annotations

import json
import os
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cli import pipeline as pl
from src.pipeline_runner.phase1_store import (
    check_phase1_reusable,
    phase1_result_path,
    save_phase1_result,
)
from src.pipeline_runner.steps import (
    PHASE1_FALLBACK_MARKER,
    PHASE1_REUSE_MARKER,
    PIPELINE_STEPS_FULL,
    PIPELINE_STEPS_PHASE1,
    PIPELINE_STEPS_PHASE2_ONLY,
    advance_step,
)

UNIVERSE = ["T0", "T1", "T2", "T3", "T4", "T5", "BAD"]
NEG_INF = float("-inf")


def _entry(t, rank, score):
    return types.SimpleNamespace(ticker=t, rank=rank, composite_score=score, signals_available=3,
                                 signals={}, details={})


def _phase1_entries():
    ents = [_entry(t, i + 1, 0.9 - i * 0.1) for i, t in enumerate(UNIVERSE[:-1])]
    ents.append(_entry("BAD", 0, NEG_INF))
    return ents


class Recorder:
    def __init__(self):
        self.rank_calls = []        # (tickers, include_ml)
        self.ml_calls = []          # tickers
        self.tickers_calls = 0
        self.cache_calls = 0


@pytest.fixture
def rec(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    raw = data_dir / "raw" / "daily"
    raw.mkdir(parents=True)
    old = (datetime.now(timezone.utc) - timedelta(hours=6)).timestamp()
    for t in UNIVERSE:
        f = raw / f"{t}.parquet"
        f.write_bytes(b"x")
        os.utime(f, (old, old))
    st = types.SimpleNamespace(data_dir=data_dir, raw_daily_dir=raw, ticker_source="sp500",
                               forecasts_dir=data_dir / "forecasts")
    r = Recorder()
    r.settings = st

    def fake_get_tickers(src):
        r.tickers_calls += 1
        return list(UNIVERSE)

    def fake_rank(tickers, settings=None, include_ml=False, include_indicators=True, **kw):
        r.rank_calls.append((list(tickers), include_ml))
        if include_ml:
            return [_entry(t, i + 1, 0.8 - i * 0.05) for i, t in enumerate(tickers)]
        ents = _phase1_entries()
        return [e for e in ents if e.ticker in tickers]

    def fake_save_cache(entries, settings=None, **kw):
        r.cache_calls += 1
        return Path("cache.parquet")

    fake_runner = types.ModuleType("src.ml.runner")
    fake_runner.run_tickers = lambda tickers, settings, **kw: r.ml_calls.append(list(tickers))
    fake_runner.predict_latest = lambda *a, **k: 0.05

    monkeypatch.setattr(pl, "get_settings", lambda: st)
    monkeypatch.setattr(pl, "get_tickers", fake_get_tickers)
    monkeypatch.setattr(pl, "rank_tickers", fake_rank)
    monkeypatch.setattr(pl, "save_ranking_cache", fake_save_cache)
    monkeypatch.setattr(pl, "_current_price", lambda t, s: 10.0)
    monkeypatch.setattr(pl, "_forecast_target", lambda t, s: (11.0, "2026-12-01"))
    monkeypatch.setattr(pl, "_ml_price_target", lambda t, s, target_days=20: (12.0, "2026-11-01"))
    monkeypatch.setitem(sys.modules, "src.ml.runner", fake_runner)
    return r


def _artifact(rec_):
    return json.loads(phase1_result_path(rec_.settings.data_dir).read_text(encoding="utf-8"))


def _save_artifact(rec_, day=None, entries=None):
    save_phase1_result(entries or _phase1_entries(), rec_.settings, len(UNIVERSE),
                       now=datetime.now(timezone.utc) - timedelta(hours=1), day=day)


# ------------------------------------------------------------------ 51 / 52 artifact write


def test_full_run_writes_artifact_in_phase1_order(rec, capsys):
    pl.run_pipeline(pool=4, top=3, no_deep=True)
    art = _artifact(rec)
    assert [e["ticker"] for e in art["entries"]] == UNIVERSE
    assert art["entries"][-1] == {"ticker": "BAD", "rank": 0, "score": None}
    assert art["entries"][0]["rank"] == 1
    assert art["universe_size"] == len(UNIVERSE) and art["ticker_source"] == "sp500"
    assert rec.rank_calls[0][1] is False and rec.rank_calls[1][1] is True
    assert rec.rank_calls[1][0] == UNIVERSE[:4]           # pool slice of phase1 order
    assert rec.ml_calls == [UNIVERSE[:4]]
    assert check_phase1_reusable(rec.settings, pool=4).usable


def test_phase1_only_also_writes_artifact(rec, capsys):
    pl.run_pipeline(pool=4, top=3, phase1_only=True)
    assert [e["ticker"] for e in _artifact(rec)["entries"]] == UNIVERSE
    assert len(rec.rank_calls) == 1 and rec.rank_calls[0][1] is False
    assert rec.ml_calls == [] and rec.cache_calls == 1


# ------------------------------------------------------------------ 53 reuse branch


def test_skip_phase1_reuses_usable_artifact(rec, capsys):
    _save_artifact(rec)
    pl.run_pipeline(pool=4, top=3, no_deep=True, skip_phase1=True)
    out = capsys.readouterr().out
    assert PHASE1_REUSE_MARKER in out
    assert PHASE1_FALLBACK_MARKER not in out
    assert "Fast-ranking" not in out and "[Phase 1] Done" not in out
    assert rec.tickers_calls == 0
    assert rec.rank_calls == [(UNIVERSE[:4], True)]       # exactly one call, ML on, artifact order
    assert rec.ml_calls == [UNIVERSE[:4]]
    assert "(7 ranked)" in out and "top 4 candidates" in out


def test_skip_phase1_does_not_rewrite_artifact(rec, capsys):
    _save_artifact(rec)
    before = phase1_result_path(rec.settings.data_dir).read_text(encoding="utf-8")
    pl.run_pipeline(pool=4, top=3, skip_phase1=True)
    assert phase1_result_path(rec.settings.data_dir).read_text(encoding="utf-8") == before


def test_skip_phase1_pool_larger_than_saved_uses_all(rec, capsys):
    _save_artifact(rec)
    pl.run_pipeline(pool=500, top=3, skip_phase1=True)
    assert rec.rank_calls == [(UNIVERSE, True)]


# ------------------------------------------------------------------ 54 fallback branch


@pytest.mark.parametrize("setup,reason", [
    ("missing", "missing"),
    ("stale_day", "not from today"),
    ("source", "ticker source changed"),
    ("newer_prices", "price data updated after Phase 1 ran"),
])
def test_skip_phase1_falls_back_when_not_usable(rec, capsys, setup, reason):
    if setup == "stale_day":
        _save_artifact(rec, day="2000-01-01")
    elif setup == "source":
        _save_artifact(rec)
        rec.settings.ticker_source = "total_market"
    elif setup == "newer_prices":
        _save_artifact(rec)
        f = rec.settings.raw_daily_dir / "T0.parquet"
        fut = datetime.now(timezone.utc).timestamp() + 30
        os.utime(f, (fut, fut))
    pl.run_pipeline(pool=4, top=3, no_deep=True, skip_phase1=True)
    out = capsys.readouterr().out
    assert PHASE1_FALLBACK_MARKER in out and f"not usable: {reason}" in out
    assert PHASE1_REUSE_MARKER not in out
    assert rec.tickers_calls == 1
    assert [c[1] for c in rec.rank_calls] == [False, True]
    assert rec.rank_calls[1][0] == UNIVERSE[:4]
    art = _artifact(rec)
    assert art["run_date"] == datetime.now().date().isoformat()           # (re)written for today
    assert [e["ticker"] for e in art["entries"]] == UNIVERSE


def test_fallback_output_order_and_reason(rec, capsys):
    _save_artifact(rec, day="2000-01-01")
    pl.run_pipeline(pool=4, top=3, no_deep=True, skip_phase1=True)
    lines = capsys.readouterr().out.splitlines()
    idx_fb = next(i for i, ln in enumerate(lines) if PHASE1_FALLBACK_MARKER in ln)
    idx_fast = next(i for i, ln in enumerate(lines) if "Fast-ranking" in ln)
    idx_done = next(i for i, ln in enumerate(lines) if "[Phase 1] Done" in ln)
    idx_train = next(i for i, ln in enumerate(lines) if "Training ML models" in ln)
    assert idx_fb < idx_fast < idx_done < idx_train
    assert "not from today (2000-01-01)" in lines[idx_fb]
    assert not any(PHASE1_REUSE_MARKER in ln for ln in lines)


# ------------------------------------------------------------------ 55 mutual exclusion


def test_phase1_only_and_skip_phase1_raises(rec):
    with pytest.raises(ValueError):
        pl.run_pipeline(phase1_only=True, skip_phase1=True)
    assert rec.rank_calls == [] and rec.tickers_calls == 0


def test_argparse_rejects_both_flags(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pipeline.py", "--phase1-only", "--skip-phase1"])
    with pytest.raises(SystemExit) as e:
        pl._parse_args()
    assert e.value.code == 2


def test_argparse_accepts_each_flag_alone(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pipeline.py", "--skip-phase1", "--pool", "100"])
    a = pl._parse_args()
    assert a.skip_phase1 is True and a.phase1_only is False and a.pool == 100
    monkeypatch.setattr(sys, "argv", ["pipeline.py"])
    a = pl._parse_args()
    assert a.skip_phase1 is False and a.phase1_only is False


# ------------------------------------------------------------------ 56 save failure


def test_artifact_save_failure_does_not_fail_run(rec, capsys, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(pl, "save_phase1_result", boom)
    pl.run_pipeline(pool=4, top=3, no_deep=True)
    out = capsys.readouterr().out
    assert "[Phase 1] Warning: could not save Phase 1 result: disk full" in out
    assert "[Phase 1] Done" in out and "Training ML models" in out
    assert rec.cache_calls == 1
    assert not phase1_result_path(rec.settings.data_dir).exists()


def test_artifact_warning_printed_before_phase1_done(rec, capsys, monkeypatch):
    monkeypatch.setattr(pl, "save_phase1_result", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    pl.run_pipeline(pool=4, top=3, phase1_only=True)
    lines = capsys.readouterr().out.splitlines()
    w = next(i for i, ln in enumerate(lines) if "Warning: could not save" in ln)
    d = next(i for i, ln in enumerate(lines) if "[Phase 1] Done" in ln)
    assert w < d


# ------------------------------------------------------------------ 57 default markers unchanged


def _walk(lines, steps, start=0):
    idx = start
    for ln in lines:
        idx = advance_step(ln, steps, idx)
    return idx


def test_default_full_run_markers_walk_full_table(rec, capsys):
    pl.run_pipeline(pool=4, top=3, no_deep=True)
    lines = capsys.readouterr().out.splitlines()
    assert _walk(lines, PIPELINE_STEPS_FULL) == len(PIPELINE_STEPS_FULL) - 1
    assert not any(PHASE1_REUSE_MARKER in ln or PHASE1_FALLBACK_MARKER in ln for ln in lines)


def test_default_phase1_only_markers_walk_phase1_table(rec, capsys):
    pl.run_pipeline(pool=4, top=3, phase1_only=True)
    lines = capsys.readouterr().out.splitlines()
    assert _walk(lines, PIPELINE_STEPS_PHASE1) == len(PIPELINE_STEPS_PHASE1) - 1


def test_reuse_run_markers_walk_phase2_only_table(rec, capsys):
    _save_artifact(rec)
    pl.run_pipeline(pool=4, top=3, skip_phase1=True)
    lines = capsys.readouterr().out.splitlines()
    assert _walk(lines, PIPELINE_STEPS_PHASE2_ONLY) == len(PIPELINE_STEPS_PHASE2_ONLY) - 1


def test_fallback_run_walks_full_table_after_switch(rec, capsys):
    pl.run_pipeline(pool=4, top=3, skip_phase1=True)      # no artifact -> fallback
    lines = capsys.readouterr().out.splitlines()
    fb = next(i for i, ln in enumerate(lines) if PHASE1_FALLBACK_MARKER in ln)
    assert _walk(lines[fb + 1:], PIPELINE_STEPS_FULL) == len(PIPELINE_STEPS_FULL) - 1


def test_reuse_and_fallback_lines_contain_no_other_step_marker(rec, capsys):
    _save_artifact(rec)
    pl.run_pipeline(pool=4, top=3, skip_phase1=True)
    reuse_line = next(ln for ln in capsys.readouterr().out.splitlines() if PHASE1_REUSE_MARKER in ln)
    for steps in (PIPELINE_STEPS_FULL, PIPELINE_STEPS_PHASE1):
        assert advance_step(reuse_line, steps, 0) == 0
    assert advance_step(reuse_line, PIPELINE_STEPS_PHASE2_ONLY, 0) == 0   # index 0 is itself, never "later"
