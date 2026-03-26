"""Tests for the Phase 2 Long-Term Forecasting Engine.

Uses synthetic quarterly price series so tests run without real market data.
All 12 methods are exercised; AutoBest is verified to select a method and
produce finite forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

def _make_series(n: int = 40, seed: int = 42) -> pd.Series:
    """Generate a synthetic quarterly price series with trend + seasonality."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2015-01-01", periods=n, freq="QE")
    t = np.arange(n)
    # Trend + quarterly seasonality + small noise
    seasonal = np.tile([5, -3, 8, -10], n // 4 + 1)[:n]
    noise = rng.normal(0, 2, n)
    prices = 100 + 0.5 * t + seasonal + noise
    return pd.Series(prices.clip(1), index=index, name="TEST")


@pytest.fixture
def series() -> pd.Series:
    return _make_series(n=40)


@pytest.fixture
def short_series() -> pd.Series:
    return _make_series(n=14)  # just enough for most methods


# ------------------------------------------------------------------ #
# metrics
# ------------------------------------------------------------------ #

class TestMetrics:
    def test_rmse_perfect(self):
        from src.forecasting.metrics import compute_rmse
        a = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(a, a) == 0.0

    def test_mae_perfect(self):
        from src.forecasting.metrics import compute_mae
        a = np.array([1.0, 2.0, 3.0])
        assert compute_mae(a, a) == 0.0

    def test_mape_zero_actuals(self):
        from src.forecasting.metrics import compute_mape
        a = np.array([0.0, 0.0])
        p = np.array([1.0, 2.0])
        assert compute_mape(a, p) == float("inf")

    def test_rmse_known_value(self):
        from src.forecasting.metrics import compute_rmse
        a = np.array([1.0, 2.0])
        p = np.array([2.0, 4.0])
        expected = np.sqrt(((1.0) ** 2 + (2.0) ** 2) / 2)
        assert abs(compute_rmse(a, p) - expected) < 1e-10


# ------------------------------------------------------------------ #
# Decomposition Methods 1-5
# ------------------------------------------------------------------ #

class TestDecomposition:
    @pytest.mark.parametrize("method_cls", [
        "AdditiveDecomposition",
        "MultiplicativeDecomposition",
        "FlatTrendDecomposition",
        "LinearTrendDecomposition",
        "EnsembleDecomposition",
    ])
    def test_fit_predict(self, series, method_cls):
        import importlib
        mod = importlib.import_module("src.forecasting.decomposition")
        cls = getattr(mod, method_cls)
        m = cls()
        m.fit(series)
        preds = m.predict(4)
        assert len(preds) == 4
        assert preds.isna().sum() == 0

    def test_too_short_raises(self):
        from src.forecasting.decomposition import AdditiveDecomposition
        short = _make_series(n=6)
        with pytest.raises(ValueError, match="quarters"):
            AdditiveDecomposition().fit(short)

    def test_multiplicative_requires_positive(self):
        from src.forecasting.decomposition import MultiplicativeDecomposition
        s = _make_series(n=20)
        s.iloc[5] = -1.0
        with pytest.raises(ValueError):
            MultiplicativeDecomposition().fit(s)

    def test_fitted_values_length(self, series):
        from src.forecasting.decomposition import AdditiveDecomposition
        m = AdditiveDecomposition()
        m.fit(series)
        fv = m.fitted_values()
        assert len(fv) > 0
        assert (fv.index <= series.index[-1]).all()


# ------------------------------------------------------------------ #
# Moving Average Methods 6-7
# ------------------------------------------------------------------ #

class TestMovingAverages:
    def test_sma_predict_shape(self, series):
        from src.forecasting.moving_average import SimpleMovingAverage
        m = SimpleMovingAverage(window=4)
        m.fit(series)
        preds = m.predict(4)
        assert len(preds) == 4
        assert all(p > 0 for p in preds)

    def test_wma_predict_shape(self, series):
        from src.forecasting.moving_average import WeightedMovingAverage
        m = WeightedMovingAverage(window=8)
        m.fit(series)
        preds = m.predict(4)
        assert len(preds) == 4

    def test_sma_too_short_raises(self):
        from src.forecasting.moving_average import SimpleMovingAverage
        s = _make_series(n=2)
        with pytest.raises(ValueError):
            SimpleMovingAverage(window=4).fit(s)

    def test_forecast_dates_are_future(self, series):
        from src.forecasting.moving_average import SimpleMovingAverage
        m = SimpleMovingAverage()
        m.fit(series)
        preds = m.predict(4)
        assert (preds.index > series.index[-1]).all()


# ------------------------------------------------------------------ #
# Exponential Smoothing Methods 8-10
# ------------------------------------------------------------------ #

class TestExponentialSmoothing:
    def test_ses(self, series):
        from src.forecasting.exponential_smoothing import SimpleExpSmoothing
        m = SimpleExpSmoothing()
        m.fit(series)
        preds = m.predict(4)
        assert len(preds) == 4
        # SES produces a flat forecast — all predicted values should be equal
        assert np.allclose(preds.values, preds.values[0], atol=1e-6)

    def test_holt(self, series):
        from src.forecasting.exponential_smoothing import HoltLinearTrend
        m = HoltLinearTrend()
        m.fit(series)
        preds = m.predict(4)
        assert len(preds) == 4

    def test_holt_winters(self, series):
        from src.forecasting.exponential_smoothing import HoltWinters
        m = HoltWinters()
        m.fit(series)
        preds = m.predict(4)
        assert len(preds) == 4
        assert preds.isna().sum() == 0

    def test_ses_too_short(self):
        from src.forecasting.exponential_smoothing import SimpleExpSmoothing
        s = _make_series(n=2)
        with pytest.raises(ValueError):
            SimpleExpSmoothing().fit(s)


# ------------------------------------------------------------------ #
# OLS Regression Method 11
# ------------------------------------------------------------------ #

class TestRegression:
    def test_fit_predict(self, series):
        from src.forecasting.regression import OLSRegression
        m = OLSRegression()
        m.fit(series)
        preds = m.predict(4)
        assert len(preds) == 4
        assert preds.isna().sum() == 0

    def test_fitted_values_match_length(self, series):
        from src.forecasting.regression import OLSRegression
        m = OLSRegression()
        m.fit(series)
        fv = m.fitted_values()
        assert len(fv) == len(series)

    def test_future_dates(self, series):
        from src.forecasting.regression import OLSRegression
        m = OLSRegression()
        m.fit(series)
        preds = m.predict(4)
        assert (preds.index > series.index[-1]).all()


# ------------------------------------------------------------------ #
# AutoBest Method 12
# ------------------------------------------------------------------ #

class TestAutoBest:
    def test_selects_winner(self, series):
        from src.forecasting.auto_best import AutoBest
        m = AutoBest(holdout=8)
        m.fit(series)
        assert m.best_method_name != ""
        assert np.isfinite(m.best_combined_rmse)

    def test_predict_after_fit(self, series):
        from src.forecasting.auto_best import AutoBest
        m = AutoBest(holdout=8)
        m.fit(series)
        preds = m.predict(4)
        assert len(preds) == 4
        assert preds.isna().sum() == 0

    def test_scores_dataframe(self, series):
        from src.forecasting.auto_best import AutoBest
        m = AutoBest(holdout=8)
        m.fit(series)
        df = m.scores_dataframe()
        assert len(df) == 11  # methods 1-11 evaluated
        assert "combined_rmse" in df.columns

    def test_raises_if_not_fit(self):
        from src.forecasting.auto_best import AutoBest
        m = AutoBest()
        with pytest.raises(RuntimeError):
            m.predict(4)


# ------------------------------------------------------------------ #
# Runner
# ------------------------------------------------------------------ #

class TestRunner:
    def test_run_all_methods_returns_12(self, series):
        from src.forecasting.runner import run_all_methods
        results = run_all_methods(series, horizons=4, holdout=8)
        assert len(results) == 12

    def test_best_result_is_finite(self, series):
        from src.forecasting.runner import best_result, run_all_methods
        results = run_all_methods(series, horizons=4, holdout=8)
        br = best_result(results)
        assert br is not None
        assert np.isfinite(br.rmse)

    def test_comparison_table_columns(self, series):
        from src.forecasting.runner import comparison_table, run_all_methods
        results = run_all_methods(series, horizons=4, holdout=8)
        df = comparison_table(results)
        assert "#" in df.columns
        assert "Method" in df.columns
        assert "RMSE" in df.columns

    def test_method_numbers_are_1_to_12(self, series):
        from src.forecasting.runner import run_all_methods
        results = run_all_methods(series, horizons=4, holdout=8)
        numbers = {r.method_number for r in results}
        assert numbers == set(range(1, 13))

    def test_successful_methods_have_forecasts(self, series):
        from src.forecasting.runner import run_all_methods
        results = run_all_methods(series, horizons=4, holdout=8)
        for r in results:
            if r.error is None:
                assert len(r.forecasts) == 4, f"{r.method_name} should have 4 forecast points"

    def test_evaluate_holdout_metrics_are_finite(self, series):
        from src.forecasting.runner import run_all_methods
        results = run_all_methods(series, horizons=4, holdout=8)
        succeeded = [r for r in results if r.error is None]
        assert len(succeeded) >= 5, "At least 5 methods should succeed on synthetic data"
        for r in succeeded:
            assert np.isfinite(r.rmse), f"{r.method_name} RMSE is not finite"
            assert np.isfinite(r.mae), f"{r.method_name} MAE is not finite"


# ------------------------------------------------------------------ #
# ForecastMethod.evaluate() shared logic
# ------------------------------------------------------------------ #

class TestEvaluateSharedLogic:
    def test_evaluate_produces_result(self, series):
        from src.forecasting.exponential_smoothing import HoltWinters
        m = HoltWinters()
        result = m.evaluate(series, holdout=8, horizons=4)
        assert result.rmse >= 0
        assert len(result.forecasts) == 4
        assert len(result.fitted_values) > 0

    def test_evaluate_too_short_raises(self):
        from src.forecasting.exponential_smoothing import SimpleExpSmoothing
        short = _make_series(n=4)
        m = SimpleExpSmoothing()
        with pytest.raises(ValueError, match="too short"):
            m.evaluate(short, holdout=8)

    def test_forecast_dates_are_after_series_end(self, series):
        from src.forecasting.regression import OLSRegression
        m = OLSRegression()
        result = m.evaluate(series, holdout=8, horizons=4)
        assert (result.forecasts.index > series.index[-1]).all()
