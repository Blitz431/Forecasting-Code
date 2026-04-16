"""Append-only trade audit log.

Every entry and exit is recorded with:
  - timestamp, ticker, side (BUY/SELL), price, shares
  - which signals triggered the trade
  - holding period (days) on exit
  - realized P&L on exit
  - strategy name

Writes to ``data/trade_journal/journal.parquet`` (upserted by trade_id so
re-running a day is safe).

Usage
-----
    from src.trading.trade_journal import TradeJournal

    journal = TradeJournal(settings)
    journal.log_entry("AAPL", price=180.0, shares=5,
                      signals={"indicator_score": 0.8, "ml_signal": 0.6},
                      strategy="momentum")
    journal.log_exit("AAPL", price=192.0, shares=5,
                     entry_price=180.0, entry_date="2025-01-10",
                     signals={"indicator_score": -0.4}, strategy="momentum",
                     exit_reason="trailing_stop")
    df = journal.load()
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------#
# Schema
# ---------------------------------------------------------------------------#

_COLUMNS = [
    "trade_id",       # UUID — unique per log row
    "timestamp",      # UTC ISO string
    "ticker",
    "side",           # "BUY" | "SELL"
    "price",          # execution price
    "shares",
    "dollar_value",   # price * shares
    "strategy",       # "momentum" | "value" | "mean_reversion" | "manual"
    "signals",        # JSON string — signal dict at time of trade
    "exit_reason",    # "signal_flip" | "trailing_stop" | "circuit_breaker" | "manual" | ""
    "entry_price",    # populated on SELL rows
    "entry_date",     # populated on SELL rows
    "holding_days",   # populated on SELL rows
    "realized_pnl",   # populated on SELL rows (dollar)
    "realized_pnl_pct",  # populated on SELL rows (percent)
]


class TradeJournal:
    """Append-only audit log backed by a Parquet file."""

    def __init__(self, settings=None):
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        self._dir: Path = settings.trade_journal_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path: Path = self._dir / "journal.parquet"

    # ---------------------------------------------------------------------- #
    # Log helpers
    # ---------------------------------------------------------------------- #

    def log_entry(
        self,
        ticker: str,
        price: float,
        shares: float,
        signals: dict[str, float] | None = None,
        strategy: str = "momentum",
    ) -> str:
        """Log a BUY trade. Returns the trade_id."""
        trade_id = str(uuid.uuid4())
        row = self._base_row(trade_id, ticker, "BUY", price, shares, strategy, signals)
        self._append(row)
        logger.info(f"Journal: BUY {shares} x {ticker} @ ${price:.2f} [{strategy}]")
        return trade_id

    def log_exit(
        self,
        ticker: str,
        price: float,
        shares: float,
        entry_price: float,
        entry_date: str,           # YYYY-MM-DD or ISO datetime
        signals: dict[str, float] | None = None,
        strategy: str = "momentum",
        exit_reason: str = "signal_flip",
    ) -> str:
        """Log a SELL trade with realized P&L. Returns the trade_id."""
        trade_id = str(uuid.uuid4())
        row = self._base_row(trade_id, ticker, "SELL", price, shares, strategy, signals)

        # Realized P&L
        realized_pnl = (price - entry_price) * shares
        realized_pnl_pct = ((price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0

        # Holding period
        holding_days = _days_between(entry_date, datetime.utcnow().isoformat())

        row.update({
            "exit_reason":      exit_reason,
            "entry_price":      entry_price,
            "entry_date":       entry_date,
            "holding_days":     holding_days,
            "realized_pnl":     realized_pnl,
            "realized_pnl_pct": realized_pnl_pct,
        })
        self._append(row)
        logger.info(
            f"Journal: SELL {shares} x {ticker} @ ${price:.2f} "
            f"P&L: ${realized_pnl:+.2f} ({realized_pnl_pct:+.1f}%) "
            f"reason={exit_reason}"
        )
        return trade_id

    # ---------------------------------------------------------------------- #
    # Storage
    # ---------------------------------------------------------------------- #

    def _base_row(
        self,
        trade_id: str,
        ticker: str,
        side: str,
        price: float,
        shares: float,
        strategy: str,
        signals: dict | None,
    ) -> dict:
        return {
            "trade_id":        trade_id,
            "timestamp":       datetime.utcnow().isoformat(),
            "ticker":          ticker,
            "side":            side,
            "price":           price,
            "shares":          shares,
            "dollar_value":    price * shares,
            "strategy":        strategy,
            "signals":         json.dumps(signals or {}),
            "exit_reason":     "",
            "entry_price":     None,
            "entry_date":      None,
            "holding_days":    None,
            "realized_pnl":    None,
            "realized_pnl_pct": None,
        }

    def _append(self, row: dict) -> None:
        new_df = pd.DataFrame([row])

        if self._path.exists():
            try:
                existing = pd.read_parquet(self._path)
                combined = pd.concat([existing, new_df], ignore_index=True)
                # Deduplicate by trade_id (idempotent re-runs)
                combined = combined.drop_duplicates(subset=["trade_id"], keep="last")
            except Exception as exc:
                logger.warning(f"Could not read existing journal ({exc}) — starting fresh.")
                combined = new_df
        else:
            combined = new_df

        combined.to_parquet(self._path, index=False)

    # ---------------------------------------------------------------------- #
    # Read
    # ---------------------------------------------------------------------- #

    def load(self) -> pd.DataFrame:
        """Load the full journal as a DataFrame."""
        if not self._path.exists():
            return pd.DataFrame(columns=_COLUMNS)
        try:
            df = pd.read_parquet(self._path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df.sort_values("timestamp", ascending=False).reset_index(drop=True)
        except Exception as exc:
            logger.error(f"load journal: {exc}")
            return pd.DataFrame(columns=_COLUMNS)

    def load_exits(self) -> pd.DataFrame:
        """Return only SELL rows (closed trades with realized P&L)."""
        df = self.load()
        if df.empty:
            return df
        return df[df["side"] == "SELL"].reset_index(drop=True)

    def summary_stats(self) -> dict:
        """Return high-level performance stats across all closed trades."""
        exits = self.load_exits()
        if exits.empty:
            return {}
        pnl = exits["realized_pnl"].dropna()
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        return {
            "total_trades":   len(exits),
            "win_rate":       len(wins) / len(pnl) if len(pnl) > 0 else 0.0,
            "total_pnl":      float(pnl.sum()),
            "avg_win":        float(wins.mean()) if not wins.empty else 0.0,
            "avg_loss":       float(losses.mean()) if not losses.empty else 0.0,
            "avg_holding_days": float(exits["holding_days"].dropna().mean()) if "holding_days" in exits else 0.0,
        }


# ---------------------------------------------------------------------------#
# Helper
# ---------------------------------------------------------------------------#

def _days_between(start: str, end: str) -> int | None:
    try:
        t0 = pd.Timestamp(start)
        t1 = pd.Timestamp(end)
        return int((t1 - t0).days)
    except Exception:
        return None
