"""Edge-case tests for SES / Holt / Holt-Winters wrappers (statsmodels 0.15 fix)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing as _SES

from src.forecasting.exponential_smoothing import (
    HoltLinearTrend,
    HoltWinters,
    SimpleExpSmoothing,
)


def _mk(values, index=True) -> pd.Series:
    v = np.asarray(values)
    idx = pd.date_range("2010-01-01", periods=len(v), freq="QE") if index else None
    return pd.Series(v, index=idx)


_RNG = np.random.default_rng(1)
_BASE = 100 + np.cumsum(_RNG.normal(0, 3, 40))
ALL = [SimpleExpSmoothing, HoltLinearTrend, HoltWinters]


@pytest.mark.parametrize("cls,n", [(SimpleExpSmoothing, 3), (HoltLinearTrend, 4), (HoltWinters, 9)])
def test_minimum_length_fits_with_finite_output(cls, n):
    m = cls()
    m.fit(_mk(_BASE[:n]))
    assert len(m.predict(4)) == 4
    assert np.isfinite(m.predict(4).values).all()
    assert np.isfinite(m.fitted_values().values).all()


@pytest.mark.parametrize("cls,n", [(SimpleExpSmoothing, 2), (HoltLinearTrend, 3), (HoltWinters, 8)])
def test_below_minimum_raises_value_error(cls, n):
    with pytest.raises(ValueError):
        cls().fit(_mk(_BASE[:n]))


@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize("data", [[50.0] * 20, [0.0] * 20, list(np.linspace(-10, 10, 20))])
def test_constant_zero_negative_are_finite(cls, data):
    m = cls()
    m.fit(_mk(data))
    assert np.isfinite(m.predict(4).values).all()


@pytest.mark.parametrize("cls", ALL)
def test_integer_dtype_and_plain_dtypes(cls):
    m = cls()
    m.fit(_mk(np.arange(10, 40)))
    assert np.isfinite(m.predict(2).values).all()
    # string-numeric values are coerced via astype(float)
    m.fit(_mk([str(x) for x in range(10, 40)]))
    assert len(m.predict(2)) == 2


@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize("h", [1, 2, 4, 12])
def test_horizon_length_and_future_dates(cls, h):
    s = _mk(_BASE)
    m = cls()
    m.fit(s)
    p = m.predict(h)
    assert len(p) == h
    assert p.index[0] > s.index[-1]
    assert np.isfinite(p.values).all()
    assert len(m.fitted_values()) == len(s)


@pytest.mark.parametrize("cls", ALL)
def test_very_long_series(cls):
    s = _mk(100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 1500)))
    m = cls()
    m.fit(s)
    assert np.isfinite(m.predict(4).values).all()


def test_ses_matches_statsmodels_direct():
    s = _mk(_BASE)
    m = SimpleExpSmoothing()
    m.fit(s)
    d = _SES(s.values, initialization_method="estimated").fit(optimized=True)
    assert np.allclose(m.predict(4).values, d.forecast(4))
    assert np.allclose(m.fitted_values().values, d.fittedvalues)


def test_holt_matches_statsmodels_direct():
    s = _mk(_BASE)
    m = HoltLinearTrend()
    m.fit(s)
    d = Holt(s.values, initialization_method="estimated").fit(optimized=True)
    assert np.allclose(m.predict(4).values, d.forecast(4))
    assert np.allclose(m.fitted_values().values, d.fittedvalues)


def test_runner_short_and_constant_series_do_not_raise():
    from src.forecasting.runner import run_all_methods

    for s in (_mk(_BASE[:12]), _mk([50.0] * 30)):
        res = run_all_methods(s, horizons=4, holdout=8)
        assert len(res) == 12
        for r in res:
            if r.error is None:
                assert np.isfinite(r.forecasts.values).all()
