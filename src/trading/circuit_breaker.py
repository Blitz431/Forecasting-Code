"""Circuit breaker — kill switch for runaway losses.

Triggers
--------
1. Portfolio drops ≥ 10% in a single day   → halt all new orders for the session.
2. Any single position is down ≥ 15% from entry → force-sell that position.
3. Manual emergency_stop()                 → close every open position immediately.

State is persisted to ``data/circuit_breaker_state.json`` so it survives process
restarts (e.g., a crash during the trading day).

Usage
-----
    from src.trading.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(settings)
    cb.check_portfolio(account)          # raises CircuitBreakerTripped if triggered
    cb.check_positions(positions, client)# force-sells losers
    cb.emergency_stop(client)            # flatten everything
    cb.reset()                           # clear state at start of new day
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class CircuitBreakerTripped(Exception):
    """Raised when the portfolio circuit breaker fires."""


# ---------------------------------------------------------------------------#
# Persistent state
# ---------------------------------------------------------------------------#

@dataclass
class CBState:
    halted: bool = False
    halt_reason: str = ""
    halt_time: str = ""
    session_date: str = ""           # YYYY-MM-DD — reset each new trading day
    portfolio_open_value: float = 0.0   # portfolio value at start of today's session
    force_sells_today: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CBState":
        state = cls()
        for k, v in d.items():
            if hasattr(state, k):
                setattr(state, k, v)
        return state


# ---------------------------------------------------------------------------#
# CircuitBreaker
# ---------------------------------------------------------------------------#

class CircuitBreaker:
    """Monitor portfolio and individual positions; halt on excessive loss."""

    def __init__(self, settings=None):
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        self._s = settings
        self._state_file: Path = settings.circuit_breaker_state_file
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state: CBState = self._load_state()
        self._ensure_session()

    # ---------------------------------------------------------------------- #
    # State I/O
    # ---------------------------------------------------------------------- #

    def _load_state(self) -> CBState:
        if self._state_file.exists():
            try:
                data = json.loads(self._state_file.read_text())
                return CBState.from_dict(data)
            except Exception as exc:
                logger.warning(f"Could not load circuit breaker state: {exc}. Starting fresh.")
        return CBState()

    def _save_state(self) -> None:
        try:
            self._state_file.write_text(json.dumps(self._state.to_dict(), indent=2))
        except Exception as exc:
            logger.error(f"Could not save circuit breaker state: {exc}")

    def _ensure_session(self) -> None:
        """Reset state if it's a new trading day."""
        today = date.today().isoformat()
        if self._state.session_date != today:
            logger.info(f"New trading day ({today}) — resetting circuit breaker state.")
            self._state = CBState(session_date=today)
            self._save_state()

    # ---------------------------------------------------------------------- #
    # Portfolio-level check
    # ---------------------------------------------------------------------- #

    def arm(self, portfolio_value: float) -> None:
        """Record the opening portfolio value for today's session.

        Call this once at session start (after market open).
        """
        self._state.portfolio_open_value = portfolio_value
        self._save_state()
        logger.info(f"Circuit breaker armed — opening value: ${portfolio_value:,.2f}")

    def check_portfolio(self, current_value: float) -> None:
        """Check portfolio-level drawdown. Raises CircuitBreakerTripped if triggered.

        Parameters
        ----------
        current_value: Current total portfolio value.
        """
        if self._state.halted:
            raise CircuitBreakerTripped(
                f"Circuit breaker already tripped: {self._state.halt_reason}"
            )

        open_val = self._state.portfolio_open_value
        if open_val <= 0:
            return  # not armed yet

        pct_drop = (open_val - current_value) / open_val * 100
        threshold = self._s.circuit_breaker_daily_pct

        if pct_drop >= threshold:
            reason = (
                f"Portfolio dropped {pct_drop:.1f}% today "
                f"(limit: {threshold:.0f}%). Halting all new orders."
            )
            logger.critical(f"CIRCUIT BREAKER TRIPPED: {reason}")
            self._state.halted = True
            self._state.halt_reason = reason
            self._state.halt_time = datetime.utcnow().isoformat()
            self._save_state()
            raise CircuitBreakerTripped(reason)

    def is_halted(self) -> bool:
        return self._state.halted

    # ---------------------------------------------------------------------- #
    # Position-level force-sell
    # ---------------------------------------------------------------------- #

    def check_positions(self, client) -> list[str]:
        """Force-sell any position down ≥ 15% from entry.

        Parameters
        ----------
        client: AlpacaClient instance.

        Returns
        -------
        List of tickers force-sold.
        """
        threshold = self._s.circuit_breaker_single_stock_pct
        positions = client.list_positions()
        force_sold: list[str] = []

        for pos in positions:
            if pos.unrealized_plpc <= -threshold:
                logger.warning(
                    f"Force-selling {pos.ticker}: down {pos.unrealized_plpc:.1f}% "
                    f"(threshold: -{threshold:.0f}%)"
                )
                result = client.close_position(pos.ticker)
                if result is not None:
                    force_sold.append(pos.ticker)
                    if pos.ticker not in self._state.force_sells_today:
                        self._state.force_sells_today.append(pos.ticker)

        if force_sold:
            self._save_state()

        return force_sold

    # ---------------------------------------------------------------------- #
    # Emergency stop
    # ---------------------------------------------------------------------- #

    def emergency_stop(self, client) -> list[str]:
        """Immediately close all positions and cancel all orders.

        Returns list of tickers closed.
        """
        logger.critical("EMERGENCY STOP ACTIVATED — closing all positions.")

        # Cancel pending orders first
        client.cancel_all_orders()

        # Close all positions
        results = client.close_all_positions()
        closed = [r.ticker for r in results]

        reason = f"Manual emergency stop — closed {len(closed)} positions."
        self._state.halted = True
        self._state.halt_reason = reason
        self._state.halt_time = datetime.utcnow().isoformat()
        self._save_state()

        logger.critical(f"Emergency stop complete: {closed}")
        return closed

    # ---------------------------------------------------------------------- #
    # Reset
    # ---------------------------------------------------------------------- #

    def reset(self) -> None:
        """Clear halt state (call at start of a new trading session)."""
        today = date.today().isoformat()
        self._state = CBState(session_date=today)
        self._save_state()
        logger.info("Circuit breaker state reset.")

    # ---------------------------------------------------------------------- #
    # Status summary
    # ---------------------------------------------------------------------- #

    def status(self) -> dict:
        """Return a dict summary for display in the dashboard."""
        open_val = self._state.portfolio_open_value
        return {
            "halted":               self._state.halted,
            "halt_reason":          self._state.halt_reason,
            "halt_time":            self._state.halt_time,
            "session_date":         self._state.session_date,
            "portfolio_open_value": open_val,
            "daily_drop_threshold": self._s.circuit_breaker_daily_pct,
            "single_stock_threshold": self._s.circuit_breaker_single_stock_pct,
            "force_sells_today":    self._state.force_sells_today,
        }
