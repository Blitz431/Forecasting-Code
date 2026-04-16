"""Alpaca broker wrapper — paper mode by default.

Paper trading is the default unless the environment variable
``ALPACA_LIVE_TRADING=true`` is explicitly set in ``.env``.

Public API
----------
    from src.trading.alpaca_client import AlpacaClient

    client = AlpacaClient(settings)
    account   = client.get_account()
    positions = client.list_positions()
    order     = client.place_order("AAPL", qty=1, side="buy", order_type="market")
    client.cancel_all_orders()
    client.close_position("AAPL")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------#
# Live trading guard
# ---------------------------------------------------------------------------#

def _is_live_mode() -> bool:
    """Return True only when ALPACA_LIVE_TRADING=true is explicitly set."""
    return os.getenv("ALPACA_LIVE_TRADING", "").strip().lower() == "true"


# ---------------------------------------------------------------------------#
# Return types
# ---------------------------------------------------------------------------#

@dataclass
class AccountInfo:
    status: str
    portfolio_value: float
    cash: float
    buying_power: float
    equity: float
    last_equity: float
    daytrade_count: int
    trading_blocked: bool
    account_blocked: bool
    pattern_day_trader: bool

    @property
    def daily_pnl(self) -> float:
        return self.equity - self.last_equity

    @property
    def daily_pnl_pct(self) -> float:
        if self.last_equity == 0:
            return 0.0
        return (self.equity - self.last_equity) / self.last_equity * 100


@dataclass
class PositionInfo:
    ticker: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float
    side: str


@dataclass
class OrderResult:
    order_id: str
    ticker: str
    qty: float
    side: str
    order_type: str
    status: str
    submitted_at: str
    filled_avg_price: float | None = None


# ---------------------------------------------------------------------------#
# Client
# ---------------------------------------------------------------------------#

class AlpacaClient:
    """Thin wrapper around alpaca-trade-api.

    Always initialises in paper mode unless ``ALPACA_LIVE_TRADING=true``
    is set in the environment.
    """

    def __init__(self, settings=None):
        self._settings = settings
        self._live = _is_live_mode()
        self._api = self._build_api()

    def _build_api(self):
        if self._settings is None:
            from config.settings import get_settings
            self._settings = get_settings()

        if self._live:
            base_url = "https://api.alpaca.markets"
            logger.warning(
                "LIVE TRADING MODE ACTIVE — real money at risk. "
                "Unset ALPACA_LIVE_TRADING or set it to 'false' to return to paper."
            )
        else:
            base_url = "https://paper-api.alpaca.markets"
            logger.info("Paper trading mode active (ALPACA_LIVE_TRADING not set).")

        try:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(
                self._settings.alpaca.api_key,
                self._settings.alpaca.secret_key,
                base_url,
                api_version="v2",
            )
            return api
        except ImportError:
            logger.error("alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
            return None
        except Exception as exc:
            logger.error(f"Failed to initialise Alpaca API: {exc}")
            return None

    # ---------------------------------------------------------------------- #
    # Account
    # ---------------------------------------------------------------------- #

    def get_account(self) -> AccountInfo | None:
        """Return account info or None if unavailable."""
        if self._api is None:
            return None
        try:
            a = self._api.get_account()
            return AccountInfo(
                status=a.status,
                portfolio_value=float(a.portfolio_value),
                cash=float(a.cash),
                buying_power=float(a.buying_power),
                equity=float(a.equity),
                last_equity=float(a.last_equity),
                daytrade_count=int(getattr(a, "daytrade_count", 0) or 0),
                trading_blocked=bool(a.trading_blocked),
                account_blocked=bool(a.account_blocked),
                pattern_day_trader=bool(getattr(a, "pattern_day_trader", False)),
            )
        except Exception as exc:
            logger.error(f"get_account failed: {exc}")
            return None

    # ---------------------------------------------------------------------- #
    # Positions
    # ---------------------------------------------------------------------- #

    def list_positions(self) -> list[PositionInfo]:
        """Return all open positions."""
        if self._api is None:
            return []
        try:
            positions = self._api.list_positions()
            result = []
            for p in positions:
                result.append(PositionInfo(
                    ticker=p.symbol,
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    market_value=float(p.market_value),
                    unrealized_pl=float(p.unrealized_pl),
                    unrealized_plpc=float(p.unrealized_plpc) * 100,
                    side=p.side,
                ))
            return result
        except Exception as exc:
            logger.error(f"list_positions failed: {exc}")
            return []

    def get_position(self, ticker: str) -> PositionInfo | None:
        """Return a single position or None if not held."""
        if self._api is None:
            return None
        try:
            p = self._api.get_position(ticker)
            return PositionInfo(
                ticker=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc) * 100,
                side=p.side,
            )
        except Exception:
            return None

    # ---------------------------------------------------------------------- #
    # Orders
    # ---------------------------------------------------------------------- #

    def place_order(
        self,
        ticker: str,
        qty: float,
        side: str,          # "buy" or "sell"
        order_type: str = "market",
        limit_price: float | None = None,
        time_in_force: str = "day",
    ) -> OrderResult | None:
        """Submit an order.

        Parameters
        ----------
        ticker:        Stock symbol.
        qty:           Number of shares (fractional allowed on Alpaca).
        side:          "buy" or "sell".
        order_type:    "market" | "limit" | "stop" | "stop_limit".
        limit_price:   Required for limit / stop_limit orders.
        time_in_force: "day" | "gtc" | "opg" | "cls" | "ioc" | "fok".
        """
        if self._api is None:
            logger.error("Alpaca API not initialised — cannot place order.")
            return None

        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got: {side!r}")

        kwargs: dict[str, Any] = {
            "symbol":        ticker,
            "qty":           str(qty),
            "side":          side,
            "type":          order_type,
            "time_in_force": time_in_force,
        }
        if limit_price is not None:
            kwargs["limit_price"] = str(limit_price)

        mode_tag = "LIVE" if self._live else "PAPER"
        logger.info(f"[{mode_tag}] Placing {side.upper()} {order_type} order: {qty} x {ticker}")

        try:
            o = self._api.submit_order(**kwargs)
            return OrderResult(
                order_id=o.id,
                ticker=o.symbol,
                qty=float(o.qty),
                side=o.side,
                order_type=o.order_type,
                status=o.status,
                submitted_at=str(o.submitted_at),
                filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
            )
        except Exception as exc:
            logger.error(f"place_order({ticker}, {qty}, {side}): {exc}")
            return None

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        if self._api is None:
            return 0
        try:
            cancelled = self._api.cancel_all_orders()
            n = len(cancelled) if cancelled else 0
            logger.info(f"Cancelled {n} open orders.")
            return n
        except Exception as exc:
            logger.error(f"cancel_all_orders failed: {exc}")
            return 0

    def close_position(self, ticker: str) -> OrderResult | None:
        """Submit a market order to liquidate the full position in *ticker*."""
        if self._api is None:
            return None
        try:
            o = self._api.close_position(ticker)
            logger.info(f"Closing position: {ticker}")
            return OrderResult(
                order_id=o.id,
                ticker=o.symbol,
                qty=float(o.qty),
                side=o.side,
                order_type=o.order_type,
                status=o.status,
                submitted_at=str(o.submitted_at),
                filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
            )
        except Exception as exc:
            logger.error(f"close_position({ticker}) failed: {exc}")
            return None

    def close_all_positions(self) -> list[OrderResult]:
        """Submit market sell orders for every open position."""
        positions = self.list_positions()
        results = []
        for pos in positions:
            result = self.close_position(pos.ticker)
            if result is not None:
                results.append(result)
        logger.info(f"Submitted close orders for {len(results)} positions.")
        return results

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    @property
    def is_live(self) -> bool:
        return self._live

    @property
    def connected(self) -> bool:
        return self._api is not None
