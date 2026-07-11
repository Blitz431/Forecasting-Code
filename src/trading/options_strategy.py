from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Automated options trading strategy — enters calls/puts when flow, vol, and composite signals align; manages a fixed capital pool.

Connections:
  - src/options/options_data.py: reads flow_signal per ticker to screen entries
  - src/options/implied_vol.py: reads vol_signal to confirm IV-backed entries
  - src/ranking/ranker.py: reads composite_score as third confirmation signal
  - src/trading/alpaca_client.py: places options orders via Alpaca API
  - config/settings.py: options_capital, options_max_dte, options_min_dte, options_exit_dte, profit_target, stop_loss

In:  flow_signal, vol_signal, composite_score per ticker + live options chain via yfinance
Out: options order placements; state persisted to data/options_positions.json
"""

_STATE_FILE = "options_positions.json"


# ---------------------------------------------------------------------------#
# Data classes
# ---------------------------------------------------------------------------#

@dataclass
class OptionTradeSignal:
    symbol: str           # OSI contract symbol
    underlying: str
    contract_type: str    # "call" | "put"
    strike: float
    expiration: str       # YYYY-MM-DD
    qty: int
    side: str             # "buy" | "sell"
    limit_price: float | None
    reason: str


@dataclass
class OptionPositionState:
    symbol: str
    underlying: str
    contract_type: str    # "call" | "put"
    strike: float
    expiration: str       # YYYY-MM-DD
    qty: int
    entry_premium: float  # per-share premium at fill
    entry_date: str       # YYYY-MM-DD
    iv_spike_entry: bool  # True when iv_spike was set at entry


# ---------------------------------------------------------------------------#
# Strategy
# ---------------------------------------------------------------------------#

class OptionsStrategy:
    """Buy-only options strategy with defined-risk entries and automatic exits."""

    def __init__(self, settings=None):
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        self._s = settings
        self._state_path: Path = settings.data_dir / _STATE_FILE

    # ---------------------------------------------------------------------- #
    # State persistence
    # ---------------------------------------------------------------------- #

    def load_positions(self) -> list[OptionPositionState]:
        if not self._state_path.exists():
            return []
        try:
            data = json.loads(self._state_path.read_text())
            return [OptionPositionState(**p) for p in data.get("positions", [])]
        except Exception as exc:
            logger.error(f"load_positions failed: {exc}")
            return []

    def save_positions(self, positions: list[OptionPositionState]) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._state_path.write_text(
                json.dumps({"positions": [asdict(p) for p in positions]}, indent=2)
            )
        except Exception as exc:
            logger.error(f"save_positions failed: {exc}")

    def record_fill(
        self,
        symbol: str,
        underlying: str,
        contract_type: str,
        strike: float,
        expiration: str,
        qty: int,
        fill_price: float,
        iv_spike_entry: bool = False,
    ) -> None:
        positions = self.load_positions()
        if any(p.symbol == symbol for p in positions):
            return
        positions.append(OptionPositionState(
            symbol=symbol,
            underlying=underlying,
            contract_type=contract_type,
            strike=strike,
            expiration=expiration,
            qty=qty,
            entry_premium=fill_price,
            entry_date=datetime.date.today().isoformat(),
            iv_spike_entry=iv_spike_entry,
        ))
        self.save_positions(positions)
        logger.info(f"[OPTIONS] Recorded fill: {symbol} qty={qty} @ ${fill_price:.2f}")

    def record_close(self, symbol: str) -> None:
        positions = [p for p in self.load_positions() if p.symbol != symbol]
        self.save_positions(positions)
        logger.info(f"[OPTIONS] Removed closed position: {symbol}")

    # ---------------------------------------------------------------------- #
    # Budget helpers
    # ---------------------------------------------------------------------- #

    def committed_capital(self, positions: list[OptionPositionState] | None = None) -> float:
        """Sum of entry_premium × qty × 100 for all tracked positions."""
        if positions is None:
            positions = self.load_positions()
        return sum(p.entry_premium * p.qty * 100 for p in positions)

    def open_notional(self, live_positions: list) -> float:
        """Sum of current market_value for live option positions."""
        return sum(p.market_value for p in live_positions)

    # ---------------------------------------------------------------------- #
    # Exit signals
    # ---------------------------------------------------------------------- #

    def check_exits(
        self,
        live_positions: list,    # list[OptionPositionInfo]
        iv_metrics: dict,        # ticker -> compute_iv_metrics dict
    ) -> list[OptionTradeSignal]:
        """Return sell signals for positions that have hit exit rules."""
        today = datetime.date.today()
        positions = self.load_positions()
        live_map = {p.symbol: p for p in live_positions}

        signals: list[OptionTradeSignal] = []
        for state in positions:
            live = live_map.get(state.symbol)
            if live is None:
                continue

            # Days to expiration
            try:
                exp_date = datetime.date.fromisoformat(state.expiration)
                dte = (exp_date - today).days
            except Exception:
                dte = 999

            reason: str | None = None
            profit_pct = live.unrealized_plpc

            if profit_pct >= self._s.options_profit_target * 100:
                reason = f"profit_target: +{profit_pct:.1f}%"
            elif profit_pct <= -(self._s.options_stop_loss * 100):
                reason = f"stop_loss: {profit_pct:.1f}%"
            elif dte <= self._s.options_exit_dte:
                reason = f"dte_guard: {dte} DTE remaining"
            elif state.iv_spike_entry:
                iv_spike_now = iv_metrics.get(state.underlying, {}).get("iv_spike", True)
                if not iv_spike_now:
                    reason = "iv_crush: spike cleared after entry"

            if reason:
                signals.append(OptionTradeSignal(
                    symbol=state.symbol,
                    underlying=state.underlying,
                    contract_type=state.contract_type,
                    strike=state.strike,
                    expiration=state.expiration,
                    qty=int(state.qty),
                    side="sell",
                    limit_price=live.current_price,
                    reason=reason,
                ))

        return signals

    # ---------------------------------------------------------------------- #
    # Entry signals
    # ---------------------------------------------------------------------- #

    def generate_entries(
        self,
        top_picks: list,          # list[RankEntry]
        iv_metrics: dict,         # ticker -> compute_iv_metrics dict
        options_metrics: dict,    # ticker -> compute_options_metrics dict
        client,                   # AlpacaClient
    ) -> list[OptionTradeSignal]:
        """Return buy signals for new option positions that fit the budget."""
        s = self._s
        positions = self.load_positions()
        held_underlyings = {p.underlying for p in positions}

        max_per_pos = s.options_capital * 0.05
        max_positions = int(s.options_capital // max_per_pos) if max_per_pos > 0 else 0
        remaining = s.options_capital - self.committed_capital(positions)

        if len(positions) >= max_positions or remaining < max_per_pos:
            return []

        signals: list[OptionTradeSignal] = []

        for entry in top_picks:
            ticker = entry.ticker
            if ticker in held_underlyings:
                continue

            flow = options_metrics.get(ticker, {}).get("flow_signal", 0.0)
            iv_spike = iv_metrics.get(ticker, {}).get("iv_spike", False)
            score = entry.composite_score
            spot = iv_metrics.get(ticker, {}).get("spot_price")

            logger.info(
                f"[OPTIONS] {ticker}: flow={flow:.3f} iv_spike={iv_spike} score={score:.3f} "
                f"spot={spot} (need |flow|>{s.options_min_flow_signal} iv_spike=True |score|>0.5)"
            )

            if spot is None or spot <= 0:
                continue

            if (
                flow > s.options_min_flow_signal
                and iv_spike
                and score > 0.5
            ):
                direction = "call"
            elif (
                flow < -s.options_min_flow_signal
                and iv_spike
                and score < -0.5
            ):
                direction = "put"
            else:
                continue

            contracts = client.get_option_contracts(
                ticker=ticker,
                contract_type=direction,
                min_days=s.options_min_dte,
                max_days=s.options_max_dte,
                current_price=spot,
                max_strikes_per_expiry=8,
            )
            if not contracts:
                logger.debug(f"[OPTIONS] No contracts found for {ticker} {direction}")
                continue

            contract = self._select_contract(contracts, spot, direction)
            if contract is None or contract.last_price is None or contract.last_price <= 0:
                continue

            cost = contract.last_price * 100
            if cost > max_per_pos or cost > remaining:
                logger.debug(
                    f"[OPTIONS] {ticker} {direction} cost ${cost:.2f} exceeds budget "
                    f"(max_per_pos=${max_per_pos:.2f} remaining=${remaining:.2f})"
                )
                continue

            signals.append(OptionTradeSignal(
                symbol=contract.symbol,
                underlying=ticker,
                contract_type=direction,
                strike=contract.strike,
                expiration=contract.expiration,
                qty=1,
                side="buy",
                limit_price=round(contract.last_price, 2),
                reason=(
                    f"flow={flow:.2f} vol_sig={vol:.2f} "
                    f"score={score:.2f} {direction.upper()}"
                ),
            ))

            held_underlyings.add(ticker)
            remaining -= cost

            if len(positions) + len(signals) >= max_positions or remaining < max_per_pos:
                break

        return signals

    # ---------------------------------------------------------------------- #
    # Contract selection helper
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _select_contract(contracts: list, spot: float, contract_type: str):
        """Pick ATM or first OTM contract within the 2% band, or nearest strike."""
        valid = [c for c in contracts if c.last_price is not None and c.last_price > 0]
        if not valid:
            return None

        if contract_type == "call":
            band = [c for c in valid if spot <= c.strike <= spot * 1.02]
            if band:
                return min(band, key=lambda c: c.strike)
        else:
            band = [c for c in valid if spot * 0.98 <= c.strike <= spot]
            if band:
                return max(band, key=lambda c: c.strike)

        return min(valid, key=lambda c: abs(c.strike - spot))
