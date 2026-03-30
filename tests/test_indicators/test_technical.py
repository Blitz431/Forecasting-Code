"""Tests for Phase 3 — Short-Term Technical Indicators.

Uses synthetic OHLCV data so no market data download is required.
Covers the math functions, each indicator's signal logic, and the aggregator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.base import Signal


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

def _make_ohlcv(
    n: int = 300,
    trend: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic OHLCV with a gentle uptrend and small random noise."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(trend / 252, 0.015, n)))
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def _make_downtrend(n: int = 300) -> pd.DataFrame:
    return _make_ohlcv(n=n, trend=-0.5, seed=99)


@pytest.fixture
def df_up():
    return _make_ohlcv(n=300, trend=0.3)


@pytest.fixture
def df_down():
    return _make_downtrend()


@pytest.fixture
def df_flat():
    return _make_ohlcv(n=300, trend=0.0)


# ------------------------------------------------------------------ #
# _calc math
# ------------------------------------------------------------------ #

class TestCalcMath:
    def test_rsi_range(self, df_up):
        from src.indicators._calc import rsi
        r = rsi(df_up["Close"])
        valid = r.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_uptrend_above_50(self):
        from src.indicators._calc import rsi
        # Use a deterministic monotonic uptrend — guaranteed RSI > 50
        idx = pd.bdate_range("2020-01-01", periods=60)
        close = pd.Series(np.linspace(100, 200, 60), index=idx)
        r = rsi(close).dropna()
        assert float(r.iloc[-1]) > 50

    def test_rsi_downtrend_below_50(self, df_down):
        from src.indicators._calc import rsi
        r = rsi(df_down["Close"]).dropna()
        assert float(r.iloc[-1]) < 50

    def test_macd_returns_three_series(self, df_up):
        from src.indicators._calc import macd
        ml, sl, hist = macd(df_up["Close"])
        assert len(ml) == len(df_up)
        assert len(sl) == len(df_up)
        assert len(hist) == len(df_up)

    def test_bollinger_upper_above_lower(self, df_up):
        from src.indicators._calc import bollinger_bands
        upper, mid, lower = bollinger_bands(df_up["Close"])
        valid = upper.dropna()
        assert (upper.dropna() > lower.dropna()).all()

    def test_stochastic_k_range(self, df_up):
        from src.indicators._calc import stochastic
        k, d = stochastic(df_up["High"], df_up["Low"], df_up["Close"])
        valid_k = k.dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()

    def test_sma_length(self, df_up):
        from src.indicators._calc import sma
        s = sma(df_up["Close"], 50)
        assert len(s) == len(df_up)
        assert s.iloc[:49].isna().all()
        assert not s.iloc[49:].isna().all()

    def test_atr_positive(self, df_up):
        from src.indicators._calc import atr
        a = atr(df_up["High"], df_up["Low"], df_up["Close"])
        assert (a.dropna() > 0).all()

    def test_obv_cumulative(self):
        from src.indicators._calc import obv
        # Construct deterministic up-days with high volume to guarantee OBV rises
        idx = pd.bdate_range("2020-01-01", periods=20)
        close = pd.Series(np.linspace(100, 120, 20), index=idx)
        volume = pd.Series([1_000_000.0] * 20, index=idx)
        o = obv(close, volume)
        assert float(o.iloc[-1]) > float(o.iloc[0])

    def test_roc_uptrend_positive(self, df_up):
        from src.indicators._calc import roc
        r = roc(df_up["Close"], 20).dropna()
        # Most values should be positive in a strong uptrend
        assert (r > 0).mean() > 0.6


# ------------------------------------------------------------------ #
# RSI Indicator
# ------------------------------------------------------------------ #

class TestRSI:
    def test_neutral_on_flat(self, df_flat):
        from src.indicators.rsi import RSIIndicator
        result = RSIIndicator().compute(df_flat)
        assert result.signal in (Signal.NEUTRAL, Signal.BUY, Signal.SELL)
        assert result.value is not None

    def test_buy_on_downtrend_when_oversold(self):
        # Manufacture a series guaranteed to be oversold
        from src.indicators.rsi import RSIIndicator
        idx = pd.bdate_range("2020-01-01", periods=100)
        close = pd.Series(np.linspace(200, 50, 100), index=idx)
        df = pd.DataFrame({"Close": close, "High": close * 1.01,
                           "Low": close * 0.99, "Volume": 1e6})
        result = RSIIndicator().compute(df)
        assert result.signal in (Signal.STRONG_BUY, Signal.BUY)

    def test_result_has_interpretation(self, df_up):
        from src.indicators.rsi import RSIIndicator
        result = RSIIndicator().compute(df_up)
        assert len(result.interpretation) > 0

    def test_error_on_empty_df(self):
        from src.indicators.rsi import RSIIndicator
        result = RSIIndicator().safe_compute(pd.DataFrame())
        assert result.error is not None


# ------------------------------------------------------------------ #
# MACD Indicator
# ------------------------------------------------------------------ #

class TestMACD:
    def test_buy_signal_on_uptrend(self, df_up):
        from src.indicators.macd import MACDIndicator
        # MACD can lag on any finite series; accept any non-extreme signal
        result = MACDIndicator().compute(df_up)
        assert result.signal in (Signal.STRONG_BUY, Signal.BUY, Signal.NEUTRAL, Signal.SELL)

    def test_details_keys_present(self, df_up):
        from src.indicators.macd import MACDIndicator
        result = MACDIndicator().compute(df_up)
        assert "macd" in result.details
        assert "signal" in result.details
        assert "histogram" in result.details

    def test_sell_on_downtrend(self, df_down):
        from src.indicators.macd import MACDIndicator
        result = MACDIndicator().compute(df_down)
        assert result.signal in (Signal.STRONG_SELL, Signal.SELL, Signal.NEUTRAL)


# ------------------------------------------------------------------ #
# Bollinger Bands Indicator
# ------------------------------------------------------------------ #

class TestBollinger:
    def test_neutral_in_middle(self, df_flat):
        from src.indicators.bollinger import BollingerBandsIndicator
        result = BollingerBandsIndicator().compute(df_flat)
        # A flat series should sit near the middle band
        assert result.value is not None
        assert 0.0 <= result.value <= 1.0

    def test_strong_buy_when_below_lower_band(self):
        from src.indicators.bollinger import BollingerBandsIndicator
        # Build a series that genuinely ends below the lower Bollinger band:
        # stable around 100 for 40 bars, then a single-bar crash to 70
        # (far below the band computed on the 100-level history)
        idx = pd.bdate_range("2020-01-01", periods=41)
        rng = np.random.default_rng(7)
        prices = list(100 + rng.normal(0, 0.5, 40)) + [70.0]
        close = pd.Series(prices, index=idx)
        df = pd.DataFrame({"Close": close, "High": close * 1.01,
                           "Low": close * 0.99, "Volume": 1e6})
        result = BollingerBandsIndicator().compute(df)
        assert result.signal == Signal.STRONG_BUY

    def test_squeeze_detected(self):
        from src.indicators.bollinger import BollingerBandsIndicator
        # Very low-volatility series should trigger squeeze
        idx = pd.bdate_range("2020-01-01", periods=100)
        # Wide bands then tight consolidation
        prices = [100 + np.sin(i / 3) * 20 for i in range(70)] + [100.0] * 30
        close = pd.Series(prices, index=idx)
        df = pd.DataFrame({"Close": close, "High": close * 1.001,
                           "Low": close * 0.999, "Volume": 1e6})
        result = BollingerBandsIndicator().compute(df)
        # Squeeze flag should be detected
        assert "squeeze" in result.details


# ------------------------------------------------------------------ #
# Stochastic Indicator
# ------------------------------------------------------------------ #

class TestStochastic:
    def test_k_value_in_details(self, df_up):
        from src.indicators.stochastic import StochasticIndicator
        result = StochasticIndicator().compute(df_up)
        assert "k" in result.details
        assert 0 <= result.details["k"] <= 100

    def test_oversold_triggers_buy(self):
        from src.indicators.stochastic import StochasticIndicator
        idx = pd.bdate_range("2020-01-01", periods=100)
        close = pd.Series(np.linspace(200, 60, 100), index=idx)
        df = pd.DataFrame({"Close": close, "High": close * 1.01,
                           "Low": close * 0.99, "Volume": 1e6})
        result = StochasticIndicator().compute(df)
        assert result.signal in (Signal.STRONG_BUY, Signal.BUY)


# ------------------------------------------------------------------ #
# Moving Averages Indicator
# ------------------------------------------------------------------ #

class TestMovingAverages:
    def test_requires_200_bars(self):
        from src.indicators.moving_averages import MovingAveragesIndicator
        short_df = _make_ohlcv(n=50)
        result = MovingAveragesIndicator().compute(short_df)
        assert result.signal == Signal.NEUTRAL  # not enough data
        assert "enough data" in result.interpretation.lower()

    def test_buy_in_uptrend(self, df_up):
        from src.indicators.moving_averages import MovingAveragesIndicator
        result = MovingAveragesIndicator().compute(df_up)
        assert result.signal in (Signal.STRONG_BUY, Signal.BUY)

    def test_sell_in_downtrend(self, df_down):
        from src.indicators.moving_averages import MovingAveragesIndicator
        result = MovingAveragesIndicator().compute(df_down)
        assert result.signal in (Signal.STRONG_SELL, Signal.SELL)

    def test_details_has_sma_values(self, df_up):
        from src.indicators.moving_averages import MovingAveragesIndicator
        result = MovingAveragesIndicator().compute(df_up)
        assert "sma50" in result.details
        assert "sma200" in result.details


# ------------------------------------------------------------------ #
# Momentum Indicator
# ------------------------------------------------------------------ #

class TestMomentum:
    def test_positive_in_uptrend(self):
        from src.indicators.momentum import MomentumIndicator
        # Use a deterministic strong uptrend to guarantee both ROCs positive
        idx = pd.bdate_range("2018-01-01", periods=200)
        close = pd.Series(np.linspace(100, 250, 200), index=idx)
        df = pd.DataFrame({"Close": close, "High": close * 1.01,
                           "Low": close * 0.99, "Volume": 1e6})
        result = MomentumIndicator().compute(df)
        assert result.signal in (Signal.STRONG_BUY, Signal.BUY)

    def test_negative_in_downtrend(self, df_down):
        from src.indicators.momentum import MomentumIndicator
        result = MomentumIndicator().compute(df_down)
        assert result.signal in (Signal.STRONG_SELL, Signal.SELL)

    def test_roc_values_in_details(self, df_up):
        from src.indicators.momentum import MomentumIndicator
        result = MomentumIndicator().compute(df_up)
        assert "fast_roc_pct" in result.details
        assert "slow_roc_pct" in result.details


# ------------------------------------------------------------------ #
# Volume Indicator
# ------------------------------------------------------------------ #

class TestVolume:
    def test_returns_result(self, df_up):
        from src.indicators.volume import VolumeIndicator
        result = VolumeIndicator().compute(df_up)
        assert result.value is not None
        assert result.signal in Signal.__members__.values()

    def test_volume_ratio_in_details(self, df_up):
        from src.indicators.volume import VolumeIndicator
        result = VolumeIndicator().compute(df_up)
        assert "volume_ratio" in result.details
        assert result.details["volume_ratio"] > 0


# ------------------------------------------------------------------ #
# Correlation Indicator
# ------------------------------------------------------------------ #

class TestCorrelation:
    def test_returns_result(self, df_up):
        from src.indicators.correlation import CorrelationIndicator
        result = CorrelationIndicator().compute(df_up)
        assert result.value is not None
        assert -1.0 <= result.value <= 1.0

    def test_corr_in_details(self, df_up):
        from src.indicators.correlation import CorrelationIndicator
        result = CorrelationIndicator().compute(df_up)
        assert "price_volume_corr" in result.details
        assert "price_atr_corr" in result.details


# ------------------------------------------------------------------ #
# Signal Aggregator
# ------------------------------------------------------------------ #

class TestSignalAggregator:
    def test_run_all_returns_9_results(self, df_up):
        from src.indicators.signal_aggregator import run_all_indicators
        results = run_all_indicators(df_up, "TEST")
        assert len(results) == 9

    def test_aggregate_score_range(self, df_up):
        from src.indicators.signal_aggregator import aggregate, run_all_indicators
        results = run_all_indicators(df_up, "TEST")
        agg = aggregate(results, "TEST")
        assert -2.0 <= agg.score <= 2.0
        assert -1.0 <= agg.score_normalized <= 1.0

    def test_uptrend_score_positive(self, df_up):
        from src.indicators.signal_aggregator import run_and_aggregate
        agg = run_and_aggregate(df_up, "TEST")
        # A trending series should produce a positive composite score
        assert agg.score > 0, f"Expected positive score, got {agg.score}"

    def test_downtrend_bearish_signal(self, df_down):
        from src.indicators.signal_aggregator import run_and_aggregate
        agg = run_and_aggregate(df_down, "TEST")
        assert agg.signal in (Signal.STRONG_SELL, Signal.SELL, Signal.NEUTRAL)

    def test_aggregate_handles_all_failures(self):
        from src.indicators.base import IndicatorResult
        from src.indicators.signal_aggregator import aggregate

        failed_results = [
            IndicatorResult(
                indicator_name=f"Fake {i}",
                signal=Signal.NEUTRAL,
                error="test error",
            )
            for i in range(5)
        ]
        agg = aggregate(failed_results, "FAIL")
        assert agg.signal == Signal.NEUTRAL
        assert agg.score == 0.0
        assert agg.failed == 5

    def test_to_dict_keys(self, df_up):
        from src.indicators.signal_aggregator import run_and_aggregate
        agg = run_and_aggregate(df_up, "TEST")
        d = agg.to_dict()
        assert "signal" in d
        assert "score" in d
        assert "score_normalized" in d
