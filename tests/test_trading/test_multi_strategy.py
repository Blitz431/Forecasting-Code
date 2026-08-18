"""Tests for strategy tagging: TradeJournal.latest_open_strategy() and
MultiStrategyManager._mean_rev_tickers(), which fixed the previous
`_is_mean_rev` stub that always returned False.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.trading.multi_strategy import MultiStrategyManager
from src.trading.trade_journal import TradeJournal


@dataclass
class _FakeSettings:
    trade_journal_dir: "Path"
    data_dir: "Path"
    raw_daily_dir: "Path" = None


@pytest.fixture
def settings(tmp_path):
    from pathlib import Path
    return _FakeSettings(
        trade_journal_dir=tmp_path / "trade_journal",
        data_dir=tmp_path,
        raw_daily_dir=tmp_path / "raw",
    )


def test_latest_open_strategy_empty_journal(settings):
    journal = TradeJournal(settings)
    assert journal.latest_open_strategy() == {}


def test_latest_open_strategy_open_position(settings):
    journal = TradeJournal(settings)
    journal.log_entry("AAPL", price=100.0, shares=10, strategy="mean_reversion")
    assert journal.latest_open_strategy() == {"AAPL": "mean_reversion"}


def test_latest_open_strategy_closed_position_omitted(settings):
    journal = TradeJournal(settings)
    journal.log_entry("AAPL", price=100.0, shares=10, strategy="mean_reversion")
    journal.log_exit(
        "AAPL", price=105.0, shares=10,
        entry_price=100.0, entry_date="2026-08-01T00:00:00",
        strategy="mean_reversion",
    )
    # Most recent row for AAPL is now a SELL -> no open position per journal.
    assert journal.latest_open_strategy() == {}


def test_latest_open_strategy_rebuy_after_close_uses_latest(settings):
    journal = TradeJournal(settings)
    journal.log_entry("AAPL", price=100.0, shares=10, strategy="mean_reversion")
    journal.log_exit(
        "AAPL", price=105.0, shares=10,
        entry_price=100.0, entry_date="2026-08-01T00:00:00",
        strategy="mean_reversion",
    )
    journal.log_entry("AAPL", price=110.0, shares=5, strategy="momentum")
    assert journal.latest_open_strategy() == {"AAPL": "momentum"}


def test_mean_rev_tickers_uses_journal(monkeypatch, settings):
    manager = MultiStrategyManager.__new__(MultiStrategyManager)
    manager._s = settings

    journal = TradeJournal(settings)
    journal.log_entry("AAPL", price=100.0, shares=10, strategy="mean_reversion")
    journal.log_entry("MSFT", price=200.0, shares=5, strategy="momentum")

    result = manager._mean_rev_tickers({"AAPL", "MSFT", "GOOG"})
    assert result == {"AAPL"}


def test_mean_rev_tickers_empty_input_short_circuits(settings):
    manager = MultiStrategyManager.__new__(MultiStrategyManager)
    manager._s = settings
    assert manager._mean_rev_tickers(set()) == set()


def test_mean_rev_tickers_journal_failure_returns_empty_set(monkeypatch, settings):
    manager = MultiStrategyManager.__new__(MultiStrategyManager)
    manager._s = settings

    def _boom(self):
        raise OSError("disk unavailable")

    monkeypatch.setattr(TradeJournal, "latest_open_strategy", _boom)
    assert manager._mean_rev_tickers({"AAPL"}) == set()
