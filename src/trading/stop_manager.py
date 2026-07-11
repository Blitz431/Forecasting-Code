from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Reconciliation sweep that guarantees every currently-held Alpaca position has an
         active broker-side trailing stop — regardless of how or when the position was opened.

Connections:
  - src/trading/alpaca_client.py: AlpacaClient.list_positions()/list_open_orders()/place_trailing_stop()
  - config/settings.py: settings.data_dir (strategy_states.json lives here), broker_trailing_stop_pct
  - cli/trade.py: calls reconcile_trailing_stops() on every run (step 4c)
  - cli/scheduler.py: job_reconcile_stops() calls this every 5 minutes

In:  an AlpacaClient + Settings
Out: trailing-stop SELL orders placed on Alpaca for any unprotected position; updates
     data/strategy_states.json with a best-effort entry for positions it didn't already know about
"""


def _states_path(settings) -> Path:
    return settings.data_dir / "strategy_states.json"


def _load_states(settings) -> dict:
    fp = _states_path(settings)
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text())
    except Exception:
        return {}


def _save_states(settings, states: dict) -> None:
    fp = _states_path(settings)
    fp.parent.mkdir(parents=True, exist_ok=True)
    try:
        fp.write_text(json.dumps(states, indent=2))
    except Exception as exc:
        logger.warning(f"Could not save strategy states: {exc}")


def _order_type(o) -> str:
    return o.order_type.value if hasattr(o.order_type, "value") else str(o.order_type)


def _order_side(o) -> str:
    return o.side.value if hasattr(o.side, "value") else str(o.side)


def reconcile_trailing_stops(client, settings=None) -> list[str]:
    """Ensure every currently-held position has an active broker-side trailing stop.

    Returns the tickers a stop was newly attached to. Never raises — failures to
    attach a stop for an individual ticker are logged and skipped.
    """
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    positions = client.list_positions()
    if not positions:
        return []

    states = _load_states(settings)
    attached: list[str] = []
    dirty = False

    for pos in positions:
        ticker = pos.ticker
        try:
            open_orders = client.list_open_orders(ticker)
        except Exception as exc:
            logger.error(f"reconcile_trailing_stops: list_open_orders({ticker}) failed: {exc}")
            continue

        has_stop = any(
            _order_type(o) == "trailing_stop" and _order_side(o) == "sell"
            for o in open_orders
        )
        if has_stop:
            continue

        state = states.get(ticker, {})
        trail_pct = state.get("broker_trail_pct", settings.broker_trailing_stop_pct)

        result = client.place_trailing_stop(ticker, pos.qty, trail_percent=trail_pct)
        if result is None:
            logger.warning(f"reconcile_trailing_stops: failed to attach stop for {ticker}")
            continue

        attached.append(ticker)
        logger.info(f"reconcile_trailing_stops: attached {trail_pct}% trailing stop to {ticker}")

        if ticker not in states:
            states[ticker] = {
                "ticker": ticker,
                "entry_price": pos.avg_entry_price,
                "entry_date": date.today().isoformat(),
                "peak_price": pos.avg_entry_price,
                "trailing_stop_active": False,
                "trailing_stop_price": 0.0,
                "profitable_days_streak": 0,
                "last_checked_date": "",
                "holding_horizon_days": 30,
                "broker_trail_pct": trail_pct,
                "tighten_mode": "auto",
                "tighten_on_date": "",
            }
            dirty = True

    if dirty:
        _save_states(settings, states)

    return attached
