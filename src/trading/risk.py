from __future__ import annotations

import math
from dataclasses import dataclass

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Position sizing and risk limit enforcement — max % per stock, regime scaling, optional Kelly Criterion.

Connections:
  - src/analytics/market_regime.py: reads current regime to halve position sizes in Bear/High-Vol
  - src/analytics/correlation.py: receives highly_correlated pairs to further reduce correlated positions
  - config/settings.py: max_position_pct, kelly_criterion_enabled, bear_regime_position_scale
  - cli/trade.py and src/trading/backtester.py: calls position_size() and check_order() before every trade

In:  ticker, portfolio_value, current_price, current_regime
Out: SizingResult (ticker, shares, dollar_amount); check_order() returns (bool ok, reason str)
"""


@dataclass
class SizingResult:
    ticker: str
    shares: int
    dollar_amount: float
    pct_of_portfolio: float
    regime_scaled: bool
    method: str             # "kelly" | "fixed_pct"
    reason: str = ""


class RiskManager:
    """Compute position sizes and enforce hard risk limits."""

    def __init__(self, settings=None):
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        self._s = settings

    # ---------------------------------------------------------------------- #
    # Current regime (lazy import to avoid circular deps)
    # ---------------------------------------------------------------------- #

    def _current_regime(self) -> str:
        try:
            from src.analytics.market_regime import MarketRegimeAnalyzer
            analyzer = MarketRegimeAnalyzer(self._s)
            snap = analyzer.current_snapshot(compute_breadth=False)
            return snap.regime.value
        except Exception as exc:
            logger.debug(f"Could not determine regime: {exc}")
            return "Unknown"

    def _effective_max_pct(self, regime: str | None = None) -> float:
        """Return the effective max position % given current regime."""
        if regime is None:
            regime = self._current_regime()
        base = self._s.max_position_pct / 100.0   # convert to fraction
        if regime in ("Bear", "High-Volatility"):
            return base * self._s.bear_regime_position_scale
        return base

    # ---------------------------------------------------------------------- #
    # Sizing
    # ---------------------------------------------------------------------- #

    def position_size(
        self,
        ticker: str,
        portfolio_value: float,
        current_price: float,
        win_rate: float | None = None,
        avg_win_loss_ratio: float | None = None,
        regime: str | None = None,
    ) -> SizingResult:
        """Compute how many shares to buy.

        Parameters
        ----------
        ticker:             Stock symbol.
        portfolio_value:    Total portfolio value in dollars.
        current_price:      Current share price.
        win_rate:           Historical win rate [0, 1] — required for Kelly.
        avg_win_loss_ratio: Average win / average loss — required for Kelly.
        regime:             Override regime string; fetched live if None.
        """
        if regime is None:
            regime = self._current_regime()

        max_pct  = self._effective_max_pct(regime)
        is_scaled = regime in ("Bear", "High-Volatility")

        if (
            self._s.kelly_criterion_enabled
            and win_rate is not None
            and avg_win_loss_ratio is not None
            and avg_win_loss_ratio > 0
        ):
            kelly_pct = self._kelly_pct(win_rate, avg_win_loss_ratio)
            target_pct = min(kelly_pct, max_pct)
            method = "kelly"
        else:
            target_pct = max_pct
            method = "fixed_pct"

        dollar_amount = portfolio_value * target_pct
        shares = int(dollar_amount / current_price) if current_price > 0 else 0

        # Ensure at least 1 share if we're allowed to trade at all
        if shares == 0 and dollar_amount >= current_price:
            shares = 1

        actual_pct = (shares * current_price / portfolio_value * 100) if portfolio_value > 0 else 0.0

        logger.debug(
            f"[{ticker}] regime={regime} max={max_pct*100:.1f}% "
            f"target={target_pct*100:.1f}% shares={shares} "
            f"${shares * current_price:,.2f} ({actual_pct:.2f}%)"
        )

        return SizingResult(
            ticker=ticker,
            shares=shares,
            dollar_amount=float(shares * current_price),
            pct_of_portfolio=actual_pct,
            regime_scaled=is_scaled,
            method=method,
        )

    # ---------------------------------------------------------------------- #
    # Hard stop check
    # ---------------------------------------------------------------------- #

    def check_order(
        self,
        ticker: str,
        qty: float,
        current_price: float,
        portfolio_value: float,
        current_position_value: float = 0.0,
        regime: str | None = None,
    ) -> tuple[bool, str]:
        """Check whether an order passes risk limits.

        Returns
        -------
        (allowed: bool, reason: str)
        """
        if portfolio_value <= 0:
            return False, "Portfolio value is zero or negative."

        if regime is None:
            regime = self._current_regime()

        max_pct = self._effective_max_pct(regime)
        new_position_value = current_position_value + qty * current_price
        new_pct = new_position_value / portfolio_value

        if new_pct > max_pct + 0.001:   # 0.1% tolerance for rounding
            return False, (
                f"Order would put {ticker} at {new_pct*100:.1f}% of portfolio "
                f"(limit: {max_pct*100:.1f}% in {regime} regime)."
            )

        return True, "OK"

    # ---------------------------------------------------------------------- #
    # Kelly formula
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _kelly_pct(win_rate: float, avg_win_loss_ratio: float) -> float:
        """Full Kelly fraction, scaled by settings.kelly_fraction."""
        # Kelly: f* = (p * b - q) / b  where b = avg_win/avg_loss, p = win rate, q = 1-p
        b = avg_win_loss_ratio
        p = win_rate
        q = 1.0 - p
        full_kelly = (p * b - q) / b if b > 0 else 0.0
        full_kelly = max(full_kelly, 0.0)

        from config.settings import get_settings
        frac = get_settings().kelly_fraction
        return full_kelly * frac
