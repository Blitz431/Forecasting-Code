"""Tests for src/pipeline_runner/phase1_store.py (plan section 11, items 9-12)."""

from __future__ import annotations

import json
import os
import types
from datetime import datetime, timedelta, timezone

import pytest

from src.pipeline_runner.phase1_store import (
    PHASE1_SCHEMA_VERSION,
    check_phase1_reusable,
    phase1_result_path,
    save_phase1_result,
)

NEG_INF = float("-inf")


def _entries():
    E = types.SimpleNamespace
    return [
        E(ticker="NVDA", rank=1, composite_score=0.41234567),
        E(ticker="AAPL", rank=2, composite_score=0.2),
        E(ticker="MSFT", rank=3, composite_score=-0.1),
        E(ticker="BAD", rank=0, composite_score=NEG_INF),
    ]


@pytest.fixture
def st(settings):
    return settings


def _save(st, **kw):
    kw.setdefault("now", datetime.now(timezone.utc) - timedelta(hours=1))
    return save_phase1_result(_entries(), st, universe_size=498, **kw)


def _raw(st):
    return json.loads(phase1_result_path(st.data_dir).read_text(encoding="utf-8"))


def test_save_then_check_round_trip(st):
    path = _save(st)
    assert path == phase1_result_path(st.data_dir)
    res = check_phase1_reusable(st, pool=3)
    assert res.usable is True
    assert res.reason == "ok"
    assert res.pool_tickers == ["NVDA", "AAPL", "MSFT"]
    assert res.ranked_count == 4
    assert res.created_at


def test_minus_inf_saved_as_null_and_keeps_position(st):
    _save(st)
    text = phase1_result_path(st.data_dir).read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    data = json.loads(text)
    assert data["schema_version"] == PHASE1_SCHEMA_VERSION
    assert data["entries"][-1] == {"ticker": "BAD", "rank": 0, "score": None}
    assert data["entries"][0]["score"] == 0.412346   # rounded to 6 decimals
    assert [e["ticker"] for e in data["entries"]] == ["NVDA", "AAPL", "MSFT", "BAD"]
    assert data["universe_size"] == 498
    assert data["ticker_source"] == "sp500"


def test_pool_includes_failed_entry_in_saved_order(st):
    _save(st)
    assert check_phase1_reusable(st, pool=4).pool_tickers == ["NVDA", "AAPL", "MSFT", "BAD"]


def test_pool_larger_than_ranked_returns_all(st):
    _save(st)
    res = check_phase1_reusable(st, pool=1000)
    assert res.usable and len(res.pool_tickers) == 4


def test_pool_none_gives_none_tickers(st):
    _save(st)
    res = check_phase1_reusable(st)
    assert res.usable and res.pool_tickers is None


def test_missing(st):
    res = check_phase1_reusable(st, pool=10)
    assert (res.usable, res.reason, res.pool_tickers) == (False, "missing", None)


@pytest.mark.parametrize("content", [
    "{not json",
    "[]",
    "",
    json.dumps({"schema_version": 99, "created_at": "x"}),
    json.dumps({"schema_version": 1, "created_at": 5, "run_date": "d", "ticker_source": "sp500", "entries": []}),
    json.dumps({"schema_version": 1, "created_at": "2026-01-01T00:00:00+00:00", "run_date": "d",
                "ticker_source": "sp500", "entries": "nope"}),
    json.dumps({"schema_version": 1, "created_at": "2026-01-01T00:00:00+00:00", "run_date": "d",
                "ticker_source": "sp500", "entries": [{"rank": 1}]}),
    json.dumps({"schema_version": 1, "created_at": "garbage", "run_date": "d",
                "ticker_source": "sp500", "entries": [{"ticker": "A"}]}),
])
def test_unreadable_variants(st, content):
    p = phase1_result_path(st.data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    res = check_phase1_reusable(st, pool=10)
    assert res.usable is False
    assert res.reason == "unreadable"


def test_not_from_today(st):
    _save(st, day="2000-01-01")
    res = check_phase1_reusable(st, pool=10)
    assert res.usable is False
    assert res.reason == "not from today (2000-01-01)"


def test_today_parameter(st):
    _save(st, day="2000-01-01")
    assert check_phase1_reusable(st, pool=10, today="2000-01-01").usable is True


def test_ticker_source_changed(st):
    _save(st)
    st.ticker_source = "total_market"
    res = check_phase1_reusable(st, pool=10)
    assert (res.usable, res.reason) == (False, "ticker source changed")


def test_empty_entries(st):
    save_phase1_result([], st, universe_size=0, now=datetime.now(timezone.utc) - timedelta(hours=1))
    res = check_phase1_reusable(st, pool=10)
    assert (res.usable, res.reason) == (False, "empty")


def test_price_data_newer_than_phase1(st):
    _save(st)
    f = st.raw_daily_dir / "NEW.parquet"
    f.write_bytes(b"y")
    future = datetime.now(timezone.utc).timestamp() + 5
    os.utime(f, (future, future))
    res = check_phase1_reusable(st, pool=10)
    assert (res.usable, res.reason) == (False, "price data updated after Phase 1 ran")


def test_reason_has_no_newlines(st):
    _save(st, day="2000-01-01")
    assert "\n" not in check_phase1_reusable(st, pool=1).reason


def test_never_raises_on_broken_settings():
    res = check_phase1_reusable(object(), pool=5)
    assert res.usable is False and res.reason == "unreadable"


def test_atomic_write_leaves_no_tmp(st):
    for _ in range(3):
        _save(st)
    leftovers = [p.name for p in phase1_result_path(st.data_dir).parent.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []
    assert phase1_result_path(st.data_dir).exists()


def test_save_overwrites_previous(st):
    _save(st)
    save_phase1_result(_entries()[:1], st, universe_size=1, now=datetime.now(timezone.utc) - timedelta(hours=1))
    assert [e["ticker"] for e in _raw(st)["entries"]] == ["NVDA"]
