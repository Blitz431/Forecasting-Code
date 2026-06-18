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


@dataclass
class OptionPositionInfo:
    symbol: str           # OSI contract symbol
    qty: float
    avg_entry_price: float    # premium per share
    current_price: float      # current premium per share
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float    # in percent


@dataclass
class OptionContractInfo:
    symbol: str          # OSI symbol e.g. AAPL260601C00305000
    underlying: str
    expiration: str
    strike: float
    contract_type: str   # "call" or "put"
    last_price: float | None
    open_interest: int | None

    def display_label(self) -> str:
        price_str = f"  ·  last ${self.last_price:.2f}" if self.last_price else ""
        oi_str = f"  ·  OI {self.open_interest:,}" if self.open_interest else ""
        return f"{self.expiration}  ·  ${self.strike:.0f} {self.contract_type.upper()}{price_str}{oi_str}"


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
            logger.warning(
                "LIVE TRADING MODE ACTIVE — real money at risk. "
                "Unset ALPACA_LIVE_TRADING or set it to 'false' to return to paper."
            )
        else:
            logger.info("Paper trading mode active (ALPACA_LIVE_TRADING not set).")

        try:
            from alpaca.trading.client import TradingClient
            api = TradingClient(
                api_key=self._settings.alpaca.api_key,
                secret_key=self._settings.alpaca.secret_key,
                paper=not self._live,
            )
            return api
        except ImportError:
            logger.error("alpaca-py not installed. Run: pip install alpaca-py")
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
            # a.status is an AccountStatus enum; use its value for display
            status_val = a.status.value if hasattr(a.status, "value") else str(a.status)
            return AccountInfo(
                status=status_val,
                portfolio_value=float(a.portfolio_value),
                cash=float(a.cash),
                buying_power=float(a.buying_power),
                equity=float(a.equity),
                last_equity=float(a.last_equity),
                daytrade_count=int(a.daytrade_count or 0),
                trading_blocked=bool(a.trading_blocked),
                account_blocked=bool(a.account_blocked),
                pattern_day_trader=bool(a.pattern_day_trader),
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
            positions = self._api.get_all_positions()
            result = []
            for p in positions:
                side_val = p.side.value if hasattr(p.side, "value") else str(p.side)
                result.append(PositionInfo(
                    ticker=p.symbol,
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    market_value=float(p.market_value),
                    unrealized_pl=float(p.unrealized_pl),
                    unrealized_plpc=float(p.unrealized_plpc) * 100,
                    side=side_val,
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
            p = self._api.get_open_position(ticker)
            side_val = p.side.value if hasattr(p.side, "value") else str(p.side)
            return PositionInfo(
                ticker=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc) * 100,
                side=side_val,
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

        from alpaca.trading.enums import OrderSide, TimeInForce, OrderType as AlpacaOrderType
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest

        sdk_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        tif_map = {
            "day": TimeInForce.DAY, "gtc": TimeInForce.GTC,
            "opg": TimeInForce.OPG, "cls": TimeInForce.CLS,
            "ioc": TimeInForce.IOC, "fok": TimeInForce.FOK,
        }
        sdk_tif = tif_map.get(time_in_force, TimeInForce.DAY)

        mode_tag = "LIVE" if self._live else "PAPER"
        logger.info(f"[{mode_tag}] Placing {side.upper()} {order_type} order: {qty} x {ticker}")

        try:
            if order_type in ("limit", "stop_limit") and limit_price is not None:
                req = LimitOrderRequest(
                    symbol=ticker, qty=qty, side=sdk_side,
                    time_in_force=sdk_tif, limit_price=limit_price,
                )
            else:
                req = MarketOrderRequest(
                    symbol=ticker, qty=qty, side=sdk_side, time_in_force=sdk_tif,
                )
            o = self._api.submit_order(req)
            return self._order_to_result(o)
        except Exception as exc:
            logger.error(f"place_order({ticker}, {qty}, {side}): {exc}")
            return None

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        if self._api is None:
            return 0
        try:
            cancelled = self._api.cancel_orders()
            n = len(cancelled) if cancelled else 0
            logger.info(f"Cancelled {n} open orders.")
            return n
        except Exception as exc:
            logger.error(f"cancel_all_orders failed: {exc}")
            return 0

    def place_trailing_stop(
        self,
        ticker: str,
        qty: float,
        trail_percent: float,
    ) -> OrderResult | None:
        """Submit a GTC trailing-stop sell order tracked by Alpaca tick-by-tick."""
        if self._api is None:
            logger.error("Alpaca API not initialised — cannot place trailing stop.")
            return None
        try:
            from alpaca.trading.requests import TrailingStopOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            req = TrailingStopOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                trail_percent=trail_percent,
            )
            mode_tag = "LIVE" if self._live else "PAPER"
            logger.info(
                f"[{mode_tag}] Placing trailing stop SELL: {qty} x {ticker} "
                f"trail={trail_percent}%"
            )
            o = self._api.submit_order(req)
            return self._order_to_result(o)
        except Exception as exc:
            logger.error(f"place_trailing_stop({ticker}, {qty}, {trail_percent}%): {exc}")
            return None

    def list_open_orders(self, ticker: str) -> list:
        """Return open orders for *ticker*."""
        if self._api is None:
            return []
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[ticker])
            orders = self._api.get_orders(req)
            return list(orders) if orders else []
        except Exception as exc:
            logger.error(f"list_open_orders({ticker}) failed: {exc}")
            return []

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by its UUID string."""
        if self._api is None:
            return False
        try:
            import uuid
            self._api.cancel_order_by_id(uuid.UUID(order_id))
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as exc:
            logger.error(f"cancel_order({order_id}) failed: {exc}")
            return False

    def close_position(self, ticker: str) -> OrderResult | None:
        """Submit a market order to liquidate the full position in *ticker*."""
        if self._api is None:
            return None
        try:
            o = self._api.close_position(ticker)
            logger.info(f"Closing position: {ticker}")
            return self._order_to_result(o)
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
    # Options — orders & positions
    # ---------------------------------------------------------------------- #

    def place_option_order(
        self,
        symbol: str,             # OSI contract symbol e.g. "AAPL240119C00150000"
        qty: int,
        side: str,               # "buy" | "sell"
        order_type: str = "limit",
        limit_price: float | None = None,
        time_in_force: str = "day",
    ) -> OrderResult | None:
        """Submit a buy or sell order on an options contract."""
        if self._api is None:
            logger.error("Alpaca API not initialised — cannot place option order.")
            return None

        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got: {side!r}")

        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest

        sdk_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        tif_map = {
            "day": TimeInForce.DAY, "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC, "fok": TimeInForce.FOK,
        }
        sdk_tif = tif_map.get(time_in_force, TimeInForce.DAY)

        mode_tag = "LIVE" if self._live else "PAPER"
        logger.info(f"[{mode_tag}] Option {side.upper()} {order_type}: {qty} x {symbol}")

        try:
            if order_type == "limit" and limit_price is not None:
                req = LimitOrderRequest(
                    symbol=symbol, qty=qty, side=sdk_side,
                    time_in_force=sdk_tif, limit_price=limit_price,
                )
            else:
                req = MarketOrderRequest(
                    symbol=symbol, qty=qty, side=sdk_side, time_in_force=sdk_tif,
                )
            o = self._api.submit_order(req)
            return self._order_to_result(o)
        except Exception as exc:
            logger.error(f"place_option_order({symbol}, {qty}, {side}): {exc}")
            return None

    def list_option_positions(self) -> list[OptionPositionInfo]:
        """Return all open options positions (OSI symbols only)."""
        if self._api is None:
            return []
        try:
            all_positions = self._api.get_all_positions()
            result = []
            for p in all_positions:
                sym = str(p.symbol)
                # OSI symbols: 6+ chars with digits embedded after the root
                if len(sym) <= 6 or not any(c.isdigit() for c in sym[3:]):
                    continue
                result.append(OptionPositionInfo(
                    symbol=sym,
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    market_value=float(p.market_value),
                    unrealized_pl=float(p.unrealized_pl),
                    unrealized_plpc=float(p.unrealized_plpc) * 100,
                ))
            return result
        except Exception as exc:
            logger.error(f"list_option_positions failed: {exc}")
            return []

    def close_option_position(self, symbol: str) -> OrderResult | None:
        """Submit a market sell to close an options position by OSI symbol."""
        if self._api is None:
            return None
        try:
            o = self._api.close_position(symbol)
            logger.info(f"Closing option position: {symbol}")
            return self._order_to_result(o)
        except Exception as exc:
            logger.error(f"close_option_position({symbol}) failed: {exc}")
            return None

    # ---------------------------------------------------------------------- #
    # Options — chain lookup
    # ---------------------------------------------------------------------- #

    def get_option_contracts(
        self,
        ticker: str,
        contract_type: str,          # "call" or "put"
        min_days: int = 7,
        max_days: int = 60,
        current_price: float | None = None,
        max_strikes_per_expiry: int = 8,
    ) -> list[OptionContractInfo]:
        """Return active option contracts near the current price.

        Filters to the closest *max_strikes_per_expiry* strikes around
        *current_price* for each expiration date found in the window.
        """
        if self._api is None:
            return []
        try:
            import datetime
            from alpaca.trading.requests import GetOptionContractsRequest
            from alpaca.trading.enums import ContractType, AssetStatus

            today = datetime.date.today()
            ct = ContractType.CALL if contract_type.lower() == "call" else ContractType.PUT
            req = GetOptionContractsRequest(
                underlying_symbols=[ticker],
                status=AssetStatus.ACTIVE,
                expiration_date_gte=str(today + datetime.timedelta(days=min_days)),
                expiration_date_lte=str(today + datetime.timedelta(days=max_days)),
                type=ct,
                limit=200,
            )
            raw = self._api.get_option_contracts(req)
            contracts_raw = raw.option_contracts if hasattr(raw, "option_contracts") else raw

            result = [
                OptionContractInfo(
                    symbol=str(c.symbol),
                    underlying=str(c.underlying_symbol),
                    expiration=str(c.expiration_date),
                    strike=float(c.strike_price),
                    contract_type=c.type.value if hasattr(c.type, "value") else str(c.type),
                    last_price=float(c.close_price) if c.close_price else None,
                    open_interest=int(c.open_interest) if c.open_interest else None,
                )
                for c in contracts_raw
            ]

            if current_price and result:
                by_expiry: dict[str, list[OptionContractInfo]] = {}
                for c in result:
                    by_expiry.setdefault(c.expiration, []).append(c)
                filtered = []
                for exp_contracts in by_expiry.values():
                    closest = sorted(exp_contracts, key=lambda x: abs(x.strike - current_price))
                    filtered.extend(closest[:max_strikes_per_expiry])
                result = sorted(filtered, key=lambda x: (x.expiration, x.strike))

            return result
        except Exception as exc:
            logger.error(f"get_option_contracts({ticker}, {contract_type}): {exc}")
            return []

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _order_to_result(self, o) -> OrderResult:
        side_val = o.side.value if hasattr(o.side, "value") else str(o.side)
        type_val = o.order_type.value if hasattr(o.order_type, "value") else str(o.order_type)
        status_val = o.status.value if hasattr(o.status, "value") else str(o.status)
        return OrderResult(
            order_id=str(o.id),
            ticker=o.symbol,
            qty=float(o.qty),
            side=side_val,
            order_type=type_val,
            status=status_val,
            submitted_at=str(o.submitted_at),
            filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
        )

    @property
    def is_live(self) -> bool:
        return self._live

    @property
    def connected(self) -> bool:
        return self._api is not None
