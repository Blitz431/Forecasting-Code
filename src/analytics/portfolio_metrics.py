"""Portfolio performance metrics: Sharpe, Sortino, max drawdown, beta, CAGR.

All functions accept a ``pd.Series`` of daily portfolio values (equity curve)
and an optional benchmark series (e.g. SPY daily close prices).
Both series must share the same DatetimeIndex.

Usage
-----
    from src.analytics.portfolio_metrics import compute_metrics

    metrics = compute_metrics(equity_curve, benchmark=spy_series)
    # -> dict with keys: sharpe, sortino, max_drawdown, cagr, beta, calmar, win_rate, ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


TRADING_DAYS = 252


@dataclass
class PortfolioMetrics:
    """Container for all computed portfolio metrics."""

    # Returns
    total_return: float = 0.0       # e.g. 0.45 = +45%
    cagr: float = 0.0               # annualised compound growth rate
    annualised_vol: float = 0.0

    # Risk-adjusted
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0             # CAGR / max_drawdown

    # Drawdown
    max_drawdown: float = 0.0       # e.g. -0.35 = -35%
    max_drawdown_duration_days: int = 0

    # Beta / correlation
    beta: float = 1.0
    alpha: float = 0.0              # annualised Jensen's alpha
    correlation: float = 1.0        # portfolio vs benchmark

    # Trade stats
    win_rate: float = 0.0           # fraction of winning periods
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0      # gross profit / gross loss

    # Benchmark comparison
    benchmark_total_return: float = 0.0
    excess_return: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


def compute_metrics(
    equity_curve: pd.Series,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.04,   # annual, e.g. 4% T-bill
) -> PortfolioMetrics:
    """Compute full suite of portfolio metrics from an equity curve.

    Parameters
    ----------
    equity_curve:
        Daily portfolio values (dollar amounts), DatetimeIndex.
    benchmark:
        Daily prices of benchmark (e.g. SPY).  Same index as equity_curve.
    risk_free_rate:
        Annual risk-free rate used for Sharpe/Sortino/alpha.

    Returns
    -------
    PortfolioMetrics
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return PortfolioMetrics()

    equity_curve = equity_curve.dropna().sort_index()
    daily_returns = equity_curve.pct_change().dropna()

    if daily_returns.empty:
        return PortfolioMetrics()

    m = PortfolioMetrics()

    # ------------------------------------------------------------------
    # Total return & CAGR
    # ------------------------------------------------------------------
    m.total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0

    n_days = max((equity_curve.index[-1] - equity_curve.index[0]).days, 1)
    n_years = n_days / 365.25
    if n_years > 0:
        m.cagr = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years)) - 1.0

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------
    m.annualised_vol = float(daily_returns.std() * np.sqrt(TRADING_DAYS))

    # ------------------------------------------------------------------
    # Sharpe ratio
    # ------------------------------------------------------------------
    rfr_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess_daily = daily_returns - rfr_daily
    if daily_returns.std() > 0:
        m.sharpe = float(excess_daily.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS))

    # ------------------------------------------------------------------
    # Sortino ratio (downside deviation only)
    # ------------------------------------------------------------------
    downside = daily_returns[daily_returns < rfr_daily] - rfr_daily
    downside_std = float(np.sqrt((downside ** 2).mean())) if len(downside) > 0 else 0.0
    if downside_std > 0:
        m.sortino = float(excess_daily.mean() / downside_std * np.sqrt(TRADING_DAYS))

    # ------------------------------------------------------------------
    # Max drawdown
    # ------------------------------------------------------------------
    rolling_max = equity_curve.cummax()
    drawdown_series = (equity_curve - rolling_max) / rolling_max
    m.max_drawdown = float(drawdown_series.min())

    # Duration: longest period below previous peak
    in_drawdown = drawdown_series < 0
    max_dur = 0
    cur_dur = 0
    for v in in_drawdown:
        if v:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    m.max_drawdown_duration_days = max_dur

    # ------------------------------------------------------------------
    # Calmar
    # ------------------------------------------------------------------
    if m.max_drawdown < 0:
        m.calmar = float(m.cagr / abs(m.max_drawdown))

    # ------------------------------------------------------------------
    # Win rate, avg win/loss, profit factor
    # ------------------------------------------------------------------
    wins  = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    total_periods = len(daily_returns)
    if total_periods > 0:
        m.win_rate = float(len(wins) / total_periods)
    m.avg_win  = float(wins.mean())  if len(wins)   > 0 else 0.0
    m.avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    if losses.sum() != 0:
        m.profit_factor = float(wins.sum() / abs(losses.sum()))

    # ------------------------------------------------------------------
    # Beta, alpha, correlation vs benchmark
    # ------------------------------------------------------------------
    if benchmark is not None and not benchmark.empty:
        bench = benchmark.reindex(equity_curve.index).dropna()
        port_aligned = equity_curve.reindex(bench.index).dropna()
        bench = bench.reindex(port_aligned.index)

        bench_ret = bench.pct_change().dropna()
        port_ret  = port_aligned.pct_change().dropna()

        common_idx = bench_ret.index.intersection(port_ret.index)
        if len(common_idx) > 2:
            br = bench_ret.loc[common_idx]
            pr = port_ret.loc[common_idx]

            cov_matrix = np.cov(pr, br)
            bench_var  = float(cov_matrix[1, 1])
            if bench_var > 0:
                m.beta = float(cov_matrix[0, 1] / bench_var)

            # Jensen's alpha (annualised)
            bench_cagr = float((bench.iloc[-1] / bench.iloc[0]) ** (1 / max(n_years, 0.01))) - 1
            m.alpha = m.cagr - (risk_free_rate + m.beta * (bench_cagr - risk_free_rate))

            corr = np.corrcoef(pr, br)
            m.correlation = float(corr[0, 1])

            m.benchmark_total_return = float(bench.iloc[-1] / bench.iloc[0]) - 1.0
            m.excess_return = m.total_return - m.benchmark_total_return

    return m


def rolling_sharpe(equity_curve: pd.Series, window: int = 63) -> pd.Series:
    """Rolling Sharpe ratio over a sliding window."""
    daily_returns = equity_curve.pct_change().dropna()
    roll_mean = daily_returns.rolling(window).mean()
    roll_std  = daily_returns.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(TRADING_DAYS)).rename("rolling_sharpe")


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Return the drawdown at each point in time (0 = at peak, -0.3 = 30% below peak)."""
    rolling_max = equity_curve.cummax()
    return ((equity_curve - rolling_max) / rolling_max).rename("drawdown")
