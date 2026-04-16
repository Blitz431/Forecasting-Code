"""Entry / exit rules for the main momentum strategy.

Rules
-----
Entry
  - Buy stocks that rank in the top-N from ``ranking/ranker.py`` and are not
    already held and pass risk checks.

Trailing stop
  - Once a position has been profitable for 3 *consecutive* days, activate a
    5% trailing stop (configurable via settings).
  - The stop price is ``peak_price * (1 - trailing_stop_pct/100)``.
  - If current price falls below stop price, exit.

Signal flip exit
  - If the ranker composite score flips to SELL (< -0.2), exit the position.

State
-----
Position metadata (peak price, consecutive profitable days, entry info) is
held in memory during a session.  The caller (trade.py CLI) is responsible for
reloading it at each run.

Usage
-----
    from src.trading.strategy import MomentumStrategy

    strat = MomentumStrategy(settings)
    buys  = strat.generate_entries(top_picks, current_positions, portfolio_value)
    exits = strat.generate_exits(current_positions, current_prices, ranker_scores)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

SELL_THRESHOLD = -0.2   # composite score below this → SELL signal


# ---------------------------------------------------------------------------#
# State per position
# ---------------------------------------------------------------------------#

@dataclass
class PositionState:
    ticker: str
    entry_price: float
    entry_date: str
    peak_price: float
    trailing_stop_active: bool = False
    trailing_stop_price: float = 0.0
    profitable_days_streak: int = 0
    last_checked_date: str = ""

    def update_peak(self, current_price: float) -> None:
        if current_price > self.peak_price:
            self.peak_price = current_price

    def stop_price(self, trailing_pct: float) -> float:
        return self.peak_price * (1.0 - trailing_pct / 100.0)


# ---------------------------------------------------------------------------#
# Strategy
# ---------------------------------------------------------------------------#

@dataclass
class TradeSignal:
    ticker: str
    action: str              # "BUY" | "SELL"
    reason: str
    score: float = 0.0
    suggested_shares: int = 0


class MomentumStrategy:
    """Entry/exit signal generator for the ranked momentum strategy."""

    def __init__(self, settings=None):
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        self._s = settings
        # In-session position state: ticker -> PositionState
        self._states: dict[str, PositionState] = {}

    # ---------------------------------------------------------------------- #
    # Entry signals
    # ---------------------------------------------------------------------- #

    def generate_entries(
        self,
        top_picks: list,                    # list[RankEntry] from ranker
        held_tickers: set[str],
        portfolio_value: float,
        max_new_positions: int = 5,
    ) -> list[TradeSignal]:
        """Return BUY signals for top picks not already held.

        Parameters
        ----------
        top_picks:        RankEntry list from ranker.top_picks().
        held_tickers:     Tickers currently in the portfolio.
        portfolio_value:  Total portfolio value.
        max_new_positions: Cap on how many new entries to open in one session.
        """
        signals: list[TradeSignal] = []

        for entry in top_picks:
            if len(signals) >= max_new_positions:
                break
            if entry.ticker in held_tickers:
                continue
            if entry.composite_score < 0.1:
                # Require at least a mildly positive signal to enter
                continue

            signals.append(TradeSignal(
                ticker=entry.ticker,
                action="BUY",
                reason=f"rank={entry.rank} score={entry.composite_score:.3f}",
                score=entry.composite_score,
            ))
            logger.info(f"Entry signal: BUY {entry.ticker} (rank={entry.rank}, score={entry.composite_score:.3f})")

        return signals

    # ---------------------------------------------------------------------- #
    # Exit signals
    # ---------------------------------------------------------------------- #

    def generate_exits(
        self,
        positions: list,                    # list[PositionInfo] from Alpaca
        current_scores: dict[str, float],   # {ticker: composite_score}
        today: str | None = None,
    ) -> list[TradeSignal]:
        """Return SELL signals for positions that trigger exit rules.

        Parameters
        ----------
        positions:      PositionInfo list from AlpacaClient.list_positions().
        current_scores: {ticker: composite_score} from latest ranker run.
        today:          Date string for streak tracking (YYYY-MM-DD).
        """
        if today is None:
            today = date.today().isoformat()

        signals: list[TradeSignal] = []

        for pos in positions:
            ticker = pos.ticker
            state  = self._get_or_create_state(pos)
            price  = pos.current_price

            # --- Update peak ---
            state.update_peak(price)

            # --- Profitable day streak ---
            if today != state.last_checked_date:
                if pos.unrealized_plpc > 0:
                    state.profitable_days_streak += 1
                else:
                    state.profitable_days_streak = 0
                state.last_checked_date = today

            # --- Activate trailing stop after N profitable days ---
            min_days = self._s.trailing_stop_days
            if not state.trailing_stop_active and state.profitable_days_streak >= min_days:
                state.trailing_stop_active = True
                state.trailing_stop_price = state.stop_price(self._s.trailing_stop_pct)
                logger.info(
                    f"[{ticker}] Trailing stop activated after {state.profitable_days_streak} "
                    f"profitable days. Stop @ ${state.trailing_stop_price:.2f}"
                )

            # --- Check trailing stop ---
            if state.trailing_stop_active:
                # Keep stop ratcheted to peak
                state.trailing_stop_price = max(
                    state.trailing_stop_price,
                    state.stop_price(self._s.trailing_stop_pct),
                )
                if price <= state.trailing_stop_price:
                    signals.append(TradeSignal(
                        ticker=ticker,
                        action="SELL",
                        reason=(
                            f"trailing_stop: price ${price:.2f} <= stop ${state.trailing_stop_price:.2f} "
                            f"(peak ${state.peak_price:.2f})"
                        ),
                        score=current_scores.get(ticker, 0.0),
                    ))
                    logger.info(f"Exit signal: {ticker} trailing stop hit @ ${price:.2f}")
                    continue

            # --- Signal flip to SELL ---
            score = current_scores.get(ticker, 0.0)
            if score < SELL_THRESHOLD:
                signals.append(TradeSignal(
                    ticker=ticker,
                    action="SELL",
                    reason=f"signal_flip: score={score:.3f} < {SELL_THRESHOLD}",
                    score=score,
                ))
                logger.info(f"Exit signal: {ticker} signal flip SELL (score={score:.3f})")

        return signals

    # ---------------------------------------------------------------------- #
    # State management
    # ---------------------------------------------------------------------- #

    def _get_or_create_state(self, pos) -> PositionState:
        """Return existing state or create a fresh one for a position."""
        ticker = pos.ticker
        if ticker not in self._states:
            self._states[ticker] = PositionState(
                ticker=ticker,
                entry_price=pos.avg_entry_price,
                entry_date=date.today().isoformat(),
                peak_price=pos.current_price,
            )
        return self._states[ticker]

    def seed_states(self, states: dict[str, dict]) -> None:
        """Load previously serialised position states (for restart continuity)."""
        for ticker, d in states.items():
            self._states[ticker] = PositionState(**d)

    def export_states(self) -> dict[str, dict]:
        """Export current states for persistence between runs."""
        from dataclasses import asdict
        return {ticker: asdict(s) for ticker, s in self._states.items()}

    def clear_state(self, ticker: str) -> None:
        """Remove state for a closed position."""
        self._states.pop(ticker, None)
