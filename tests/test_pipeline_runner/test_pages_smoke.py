"""Streamlit AppTest smokes for pages 6 and 11 (plan section 11, dashboard items).

The pages run in-process with stubbed settings, ranker, launcher, market clock and progress bar,
so there is no network access, no real ranking, no spawned process and no touching of the
repo's data/ directory or .env.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st

from config import settings as config_settings
from src.pipeline_runner import launcher, run_state
from src.pipeline_runner.launcher import DEFAULT_POOL, DEFAULT_TOP, StartResult

ROOT = Path(__file__).resolve().parents[2]
PAGE6 = str(ROOT / "dashboard" / "pages" / "6_stock_rankings.py")
PAGE11 = str(ROOT / "dashboard" / "pages" / "11_sector_analysis.py")

REFUSED = StartResult(False, "running", "x", None, False, "A ranking run is already in progress.")


@pytest.fixture
def stubs(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    raw = data_dir / "raw" / "daily"
    raw.mkdir(parents=True)
    (raw / "AAA.parquet").write_bytes(b"x")
    ns = types.SimpleNamespace(data_dir=data_dir, raw_daily_dir=raw, ticker_source="sp500",
                               auto_rank_enabled=False)
    monkeypatch.setattr(config_settings, "get_settings", lambda: ns)

    import dashboard.components.market_clock as market_clock
    import dashboard.components.pipeline_progress as progress
    monkeypatch.setattr(market_clock, "render_market_clock", lambda: None)
    monkeypatch.setattr(progress, "render_pipeline_progress", lambda: None)

    rec = types.SimpleNamespace(top_picks=[], load_cache=[], start=[], active=False, cache=[])
    fake_ranker = types.ModuleType("src.ranking.ranker")

    def top_picks(*a, **k):
        rec.top_picks.append((a, k))
        return []

    def load_ranking_cache(settings=None):
        rec.load_cache.append(1)
        return rec.cache or None

    fake_ranker.top_picks = top_picks
    fake_ranker.load_ranking_cache = load_ranking_cache
    fake_ranker.to_dataframe = lambda picks: pd.DataFrame()
    monkeypatch.setitem(sys.modules, "src.ranking.ranker", fake_ranker)

    def fake_start(*args, **kwargs):
        rec.start.append((args, kwargs))
        return REFUSED

    monkeypatch.setattr(launcher, "start_pipeline_run", fake_start)
    monkeypatch.setattr(run_state, "is_run_active", lambda paths, now=None: rec.active)
    monkeypatch.setattr(launcher, "reap_children", lambda: None)
    st.cache_data.clear()
    yield rec
    st.cache_data.clear()


def _run(page: str):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(page, default_timeout=90)
    at.run()
    return at


def _rerun_button(at):
    btns = [b for b in at.button if "Re-run Ranking" in b.label]
    assert len(btns) == 1, [b.label for b in at.button]
    return btns[0]


# ------------------------------------------------------------------ page 6


def test_page6_no_cache_shows_info_and_never_scores(stubs):
    at = _run(PAGE6)
    assert not at.exception
    assert any("No ranking cache found" in i.value for i in at.info)
    assert stubs.top_picks == []                    # never scores the universe on page open
    assert stubs.start == []                        # and never starts a run by itself


def test_page6_rerun_without_custom_tickers_starts_fast_rank(stubs):
    at = _run(PAGE6)
    _rerun_button(at).click()
    at.run()
    assert not at.exception
    assert stubs.start == [(("phase1", DEFAULT_POOL, DEFAULT_TOP), {"trigger": "button"})]
    assert any("already in progress" in w.value for w in at.warning)   # refused result is surfaced
    assert stubs.top_picks == []


def test_page6_rerun_with_custom_tickers_does_not_start_run(stubs):
    at = _run(PAGE6)
    at.sidebar.text_input[0].input("AAPL,MSFT").run()
    _rerun_button(at).click()
    at.run()
    assert not at.exception
    assert stubs.start == []


def test_page6_rerun_disabled_while_active(stubs):
    stubs.active = True
    at = _run(PAGE6)
    assert _rerun_button(at).disabled is True
    at.sidebar.text_input[0].input("AAPL").run()
    assert _rerun_button(at).disabled is False       # custom list still allowed while a run is active
    assert stubs.start == []


# ------------------------------------------------------------------ page 11


def test_page11_no_cache_shows_info_and_never_scores(stubs):
    at = _run(PAGE11)
    assert not at.exception
    assert any("No ranking cache yet" in i.value for i in at.info)
    assert stubs.top_picks == []
    assert stubs.load_cache                          # it did look at the cache
    assert stubs.start == []
