"""Live position tracking synced with Alpaca.

Tracks open positions, average entry price, current P&L, and daily P&L.
Writes an end-of-day snapshot to ``data/portfolio/snapshots/YYYY-MM-DD.parquet``.

Usage
-----
    from src.trading.portfolio import PortfolioTracker

    tracker = PortfolioTracker(settings)
    snapshot = tracker.get_snapshot()          # live data from Alpaca
    tracker.save_eod_snapshot()                # persist to parquet
    history  = tracker.load_snapshot("2025-01-15")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------#
# Data classes
# ---------------------------------------------------------------------------#

@dataclass
class PositionSnapshot:
    ticker: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float      # percent
    daily_pl: float | None = None
    holding_days: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PortfolioSnapshot:
    as_of: str                          # ISO datetime
    portfolio_value: float
    cash: float
    equity: float
    daily_pl: float
    daily_plpc: float                   # percent
    positions: list[PositionSnapshot]
    position_count: int = 0

    def __post_init__(self):
        self.position_count = len(self.positions)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_dataframe(self) -> pd.DataFrame:
        if not self.positions:
            return pd.DataFrame()
        rows = [p.to_dict() for p in self.positions]
        df = pd.DataFrame(rows)
        df["as_of"] = self.as_of
        return df


# ---------------------------------------------------------------------------#
# Tracker
# ---------------------------------------------------------------------------#

class PortfolioTracker:
    """Sync live positions from Alpaca and persist EOD snapshots."""

    def __init__(self, settings=None):
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        self._settings = settings
        self._snapshots_dir = settings.portfolio_snapshots_dir
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

        from src.trading.alpaca_client import AlpacaClient
        self._client = AlpacaClient(settings)

    # ---------------------------------------------------------------------- #
    # Live snapshot
    # ---------------------------------------------------------------------- #

    def get_snapshot(self) -> PortfolioSnapshot | None:
        """Pull current account + positions from Alpaca."""
        account = self._client.get_account()
        if account is None:
            logger.warning("Could not fetch account info from Alpaca.")
            return None

        positions = self._client.list_positions()
        pos_snapshots: list[PositionSnapshot] = []
        for p in positions:
            pos_snapshots.append(PositionSnapshot(
                ticker=p.ticker,
                qty=p.qty,
                avg_entry_price=p.avg_entry_price,
                current_price=p.current_price,
                market_value=p.market_value,
                unrealized_pl=p.unrealized_pl,
                unrealized_plpc=p.unrealized_plpc,
            ))

        snap = PortfolioSnapshot(
            as_of=datetime.utcnow().isoformat(),
            portfolio_value=account.portfolio_value,
            cash=account.cash,
            equity=account.equity,
            daily_pl=account.daily_pnl,
            daily_plpc=account.daily_pnl_pct,
            positions=pos_snapshots,
        )
        return snap

    # ---------------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------------- #

    def save_eod_snapshot(self, snapshot: PortfolioSnapshot | None = None) -> Path | None:
        """Write EOD position snapshot to parquet.

        If *snapshot* is None, fetches live data from Alpaca first.
        Returns path written, or None on failure.
        """
        if snapshot is None:
            snapshot = self.get_snapshot()
        if snapshot is None:
            logger.warning("No snapshot data — skipping EOD save.")
            return None

        df = snapshot.to_dataframe()
        today = date.today().isoformat()
        path = self._snapshots_dir / f"{today}.parquet"

        if df.empty:
            # Save header row so we know it ran even with no positions
            df = pd.DataFrame([{
                "as_of":           snapshot.as_of,
                "portfolio_value": snapshot.portfolio_value,
                "cash":            snapshot.cash,
                "equity":          snapshot.equity,
                "daily_pl":        snapshot.daily_pl,
                "daily_plpc":      snapshot.daily_plpc,
            }])

        df.to_parquet(path, index=False)
        logger.info(f"EOD snapshot saved: {path}")
        return path

    def load_snapshot(self, date_str: str) -> pd.DataFrame:
        """Load a saved snapshot by date (YYYY-MM-DD)."""
        path = self._snapshots_dir / f"{date_str}.parquet"
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.error(f"load_snapshot({date_str}): {exc}")
            return pd.DataFrame()

    def list_snapshot_dates(self) -> list[str]:
        """Return sorted list of dates with saved snapshots."""
        files = sorted(self._snapshots_dir.glob("*.parquet"))
        return [f.stem for f in files]

    def equity_history(self) -> pd.DataFrame:
        """Load all snapshots and return a daily equity series."""
        dates = self.list_snapshot_dates()
        rows = []
        for d in dates:
            df = self.load_snapshot(d)
            if df.empty:
                continue
            if "portfolio_value" in df.columns:
                rows.append({
                    "date":            d,
                    "portfolio_value": float(df["portfolio_value"].iloc[0]),
                    "cash":            float(df["cash"].iloc[0]) if "cash" in df.columns else None,
                    "daily_pl":        float(df["daily_pl"].iloc[0]) if "daily_pl" in df.columns else None,
                })
        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows)
        result["date"] = pd.to_datetime(result["date"])
        return result.set_index("date")

    # ---------------------------------------------------------------------- #
    # Convenience helpers
    # ---------------------------------------------------------------------- #

    def position_market_values(self) -> dict[str, float]:
        """Return {ticker: market_value} for all open positions."""
        positions = self._client.list_positions()
        return {p.ticker: p.market_value for p in positions}

    def portfolio_value(self) -> float | None:
        """Return current total portfolio value."""
        account = self._client.get_account()
        return account.portfolio_value if account else None
