"""Tests for the AUTO_RANK_ENABLED setting in config/settings.py (plan section 11, item 61).

.env is never read: every settings class has its env_file disabled for these tests.
"""

from __future__ import annotations

import logging

import pytest

from config.settings import AlertSettings, AlpacaSettings, FredSettings, Settings

BOGUS = "zzq-bogus-value"


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    for cls in (Settings, AlpacaSettings, FredSettings, AlertSettings):
        monkeypatch.setitem(cls.model_config, "env_file", None)
    monkeypatch.delenv("AUTO_RANK_ENABLED", raising=False)


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING and "AUTO_RANK_ENABLED" in r.getMessage()]


def test_default_is_true(caplog):
    caplog.set_level(logging.WARNING)
    assert Settings().auto_rank_enabled is True
    assert _warnings(caplog) == []


@pytest.mark.parametrize("raw", ["", " ", "   \t", None])
def test_blank_or_none_is_true(raw):
    assert Settings(auto_rank_enabled=raw).auto_rank_enabled is True


@pytest.mark.parametrize("raw", ["true", "TRUE", " True ", "1", "yes", "on", "t", "y", "Y"])
def test_true_strings(raw, caplog):
    caplog.set_level(logging.WARNING)
    assert Settings(auto_rank_enabled=raw).auto_rank_enabled is True
    assert _warnings(caplog) == []


@pytest.mark.parametrize("raw", ["false", "FALSE", " False ", "0", "no", "off", "f", "n", "N"])
def test_false_strings(raw, caplog):
    caplog.set_level(logging.WARNING)
    assert Settings(auto_rank_enabled=raw).auto_rank_enabled is False
    assert _warnings(caplog) == []


def test_bool_passthrough():
    assert Settings(auto_rank_enabled=True).auto_rank_enabled is True
    assert Settings(auto_rank_enabled=False).auto_rank_enabled is False


@pytest.mark.parametrize("raw", ["maybe", BOGUS, "2", "enabled", "truee"])
def test_bogus_value_is_false_with_warning_and_no_value_in_message(raw, caplog):
    caplog.set_level(logging.WARNING)
    s = Settings(auto_rank_enabled=raw)           # must not raise
    assert s.auto_rank_enabled is False
    warns = _warnings(caplog)
    assert warns, "expected a warning"
    for r in warns:
        assert raw not in r.getMessage()
        assert raw not in str(r.args)


@pytest.mark.parametrize("env,expected", [
    ("", True), ("false", False), ("FALSE", False), ("true", True), ("0", False), ("1", True),
])
def test_read_from_environment(monkeypatch, env, expected):
    monkeypatch.setenv("AUTO_RANK_ENABLED", env)
    assert Settings().auto_rank_enabled is expected


def test_bogus_environment_value_warns_without_leaking_it(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv("AUTO_RANK_ENABLED", BOGUS)
    assert Settings().auto_rank_enabled is False
    warns = _warnings(caplog)
    assert warns and all(BOGUS not in r.getMessage() for r in warns)
    assert BOGUS not in caplog.text
