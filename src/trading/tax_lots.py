"""Cost basis tracking with FIFO / LIFO / specific-lot selection.

Tracks every lot opened and closed per ticker.
Classifies gains as long-term (> 365 days from open) or short-term.
Provides a tax-loss harvesting scanner to flag positions with unrealized
losses that could offset realized gains elsewhere.

Writes to ``data/tax_lots/<TICKER>.parquet``.

Usage
-----
    from src.trading.tax_lots import TaxLotTracker

    tracker = TaxLotTracker(settings)
    tracker.open_lot("AAPL", qty=10, price=180.0, date="2024-01-15")
    closed = tracker.close_lots("AAPL", qty=5, price=200.0, method="fifo")
    harvest = tracker.harvest_candidates(portfolio_value=100_000, threshold_pct=5.0)
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

Method = Literal["fifo", "lifo", "specific"]

LONG_TERM_DAYS = 365


# ---------------------------------------------------------------------------#
# Data class
# ---------------------------------------------------------------------------#

@dataclass
class Lot:
    lot_id: str
    ticker: str
    qty: float
    cost_basis: float       # per share
    open_date: str          # ISO date YYYY-MM-DD
    closed: bool = False
    close_date: str | None = None
    close_price: float | None = None
    realized_pnl: float | None = None
    gain_type: str | None = None        # "long_term" | "short_term"

    @property
    def holding_days(self) -> int:
        start = pd.Timestamp(self.open_date)
        end   = pd.Timestamp(self.close_date) if self.close_date else pd.Timestamp.now()
        return int((end - start).days)

    @property
    def is_long_term(self) -> bool:
        return self.holding_days > LONG_TERM_DAYS

    @property
    def unrealized_pnl_per_share(self) -> float:
        return 0.0  # filled by tracker when current price is known

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------#
# Tracker
# ---------------------------------------------------------------------------#

class TaxLotTracker:
    """Per-ticker lot tracking with FIFO / LIFO / specific-lot methods."""

    def __init__(self, settings=None):
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        self._dir: Path = settings.tax_lots_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------- #
    # Open / close lots
    # ---------------------------------------------------------------------- #

    def open_lot(
        self,
        ticker: str,
        qty: float,
        price: float,
        open_date: str | None = None,
    ) -> Lot:
        """Record a new purchase lot."""
        if open_date is None:
            open_date = date.today().isoformat()

        lot = Lot(
            lot_id=str(uuid.uuid4()),
            ticker=ticker,
            qty=qty,
            cost_basis=price,
            open_date=open_date,
        )
        self._save_lot(lot)
        logger.info(f"TaxLots: opened lot {lot.lot_id[:8]}… {qty} x {ticker} @ ${price:.2f}")
        return lot

    def close_lots(
        self,
        ticker: str,
        qty: float,
        price: float,
        method: Method = "fifo",
        lot_ids: list[str] | None = None,
        close_date: str | None = None,
    ) -> list[Lot]:
        """Close *qty* shares of *ticker* using specified cost basis method.

        Returns list of closed Lot objects.
        """
        if close_date is None:
            close_date = date.today().isoformat()

        open_lots = self._load_open_lots(ticker)
        if not open_lots:
            logger.warning(f"TaxLots: no open lots for {ticker}")
            return []

        # Sort according to method
        if method == "fifo":
            open_lots.sort(key=lambda l: l.open_date)
        elif method == "lifo":
            open_lots.sort(key=lambda l: l.open_date, reverse=True)
        elif method == "specific" and lot_ids:
            id_set = set(lot_ids)
            open_lots = [l for l in open_lots if l.lot_id in id_set]
        else:
            open_lots.sort(key=lambda l: l.open_date)   # default FIFO

        closed: list[Lot] = []
        remaining = qty

        for lot in open_lots:
            if remaining <= 0:
                break
            shares_from_lot = min(lot.qty, remaining)
            remaining -= shares_from_lot

            # Determine gain type
            holding = _days_between(lot.open_date, close_date)
            gain_type = "long_term" if (holding or 0) > LONG_TERM_DAYS else "short_term"
            realized_pnl = (price - lot.cost_basis) * shares_from_lot

            if shares_from_lot < lot.qty:
                # Partial close — split the lot
                remainder_lot = Lot(
                    lot_id=str(uuid.uuid4()),
                    ticker=ticker,
                    qty=lot.qty - shares_from_lot,
                    cost_basis=lot.cost_basis,
                    open_date=lot.open_date,
                )
                self._save_lot(remainder_lot)

            lot.qty = shares_from_lot
            lot.closed = True
            lot.close_date = close_date
            lot.close_price = price
            lot.realized_pnl = realized_pnl
            lot.gain_type = gain_type
            self._save_lot(lot)
            closed.append(lot)

            logger.info(
                f"TaxLots: closed {shares_from_lot} x {ticker} @ ${price:.2f} "
                f"P&L=${realized_pnl:+.2f} ({gain_type})"
            )

        return closed

    # ---------------------------------------------------------------------- #
    # Tax-loss harvesting scanner
    # ---------------------------------------------------------------------- #

    def harvest_candidates(
        self,
        current_prices: dict[str, float] | None = None,
        threshold_pct: float = 5.0,
    ) -> pd.DataFrame:
        """Scan all open lots for unrealized losses >= threshold_pct.

        Parameters
        ----------
        current_prices: {ticker: price}. If None, tries to load from daily parquet.
        threshold_pct:  Minimum unrealized loss % to flag.

        Returns
        -------
        DataFrame with columns: ticker, lot_id, qty, cost_basis, current_price,
            unrealized_pnl, unrealized_pnl_pct, holding_days, gain_type.
        """
        if current_prices is None:
            current_prices = {}

        rows = []
        for ticker_file in self._dir.glob("*.parquet"):
            ticker = ticker_file.stem
            lots = self._load_open_lots(ticker)
            if not lots:
                continue

            price = current_prices.get(ticker) or self._load_current_price(ticker)
            if price is None or price <= 0:
                continue

            for lot in lots:
                unrealized_pct = (price - lot.cost_basis) / lot.cost_basis * 100
                if unrealized_pct <= -threshold_pct:
                    rows.append({
                        "ticker":           ticker,
                        "lot_id":           lot.lot_id,
                        "qty":              lot.qty,
                        "cost_basis":       lot.cost_basis,
                        "current_price":    price,
                        "unrealized_pnl":   (price - lot.cost_basis) * lot.qty,
                        "unrealized_pnl_pct": unrealized_pct,
                        "holding_days":     lot.holding_days,
                        "gain_type":        "long_term" if lot.is_long_term else "short_term",
                        "open_date":        lot.open_date,
                    })

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.sort_values("unrealized_pnl_pct").reset_index(drop=True)

    # ---------------------------------------------------------------------- #
    # Portfolio P&L summary
    # ---------------------------------------------------------------------- #

    def realized_pnl_summary(self) -> pd.DataFrame:
        """Return summary of all realized gains/losses by ticker and gain type."""
        rows = []
        for ticker_file in self._dir.glob("*.parquet"):
            ticker = ticker_file.stem
            all_lots = self._load_all_lots(ticker)
            for lot in all_lots:
                if lot.closed and lot.realized_pnl is not None:
                    rows.append({
                        "ticker":       ticker,
                        "lot_id":       lot.lot_id,
                        "qty":          lot.qty,
                        "cost_basis":   lot.cost_basis,
                        "close_price":  lot.close_price,
                        "realized_pnl": lot.realized_pnl,
                        "gain_type":    lot.gain_type,
                        "open_date":    lot.open_date,
                        "close_date":   lot.close_date,
                        "holding_days": lot.holding_days,
                    })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("close_date", ascending=False).reset_index(drop=True)

    def open_lots_summary(self) -> pd.DataFrame:
        """Return all open lots across all tickers."""
        rows = []
        for ticker_file in self._dir.glob("*.parquet"):
            ticker = ticker_file.stem
            lots = self._load_open_lots(ticker)
            for lot in lots:
                rows.append({
                    "ticker":       ticker,
                    "lot_id":       lot.lot_id,
                    "qty":          lot.qty,
                    "cost_basis":   lot.cost_basis,
                    "open_date":    lot.open_date,
                    "holding_days": lot.holding_days,
                    "gain_type":    "long_term" if lot.is_long_term else "short_term",
                })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["ticker", "open_date"]).reset_index(drop=True)

    # ---------------------------------------------------------------------- #
    # Storage helpers
    # ---------------------------------------------------------------------- #

    def _path_for(self, ticker: str) -> Path:
        return self._dir / f"{ticker}.parquet"

    def _load_all_lots(self, ticker: str) -> list[Lot]:
        path = self._path_for(ticker)
        if not path.exists():
            return []
        try:
            df = pd.read_parquet(path)
            return [Lot(**row) for row in df.to_dict("records")]
        except Exception as exc:
            logger.error(f"load_lots({ticker}): {exc}")
            return []

    def _load_open_lots(self, ticker: str) -> list[Lot]:
        return [l for l in self._load_all_lots(ticker) if not l.closed]

    def _save_lot(self, lot: Lot) -> None:
        ticker = lot.ticker
        path = self._path_for(ticker)

        existing = self._load_all_lots(ticker)
        # Replace or append
        id_map = {l.lot_id: l for l in existing}
        id_map[lot.lot_id] = lot

        df = pd.DataFrame([l.to_dict() for l in id_map.values()])
        df.to_parquet(path, index=False)

    def _load_current_price(self, ticker: str) -> float | None:
        try:
            from config.settings import get_settings
            settings = get_settings()
            path = settings.raw_daily_dir / f"{ticker}.parquet"
            if not path.exists():
                return None
            df = pd.read_parquet(path)
            if "Close" not in df.columns or df.empty:
                return None
            return float(df["Close"].dropna().iloc[-1])
        except Exception:
            return None


# ---------------------------------------------------------------------------#
# Helper
# ---------------------------------------------------------------------------#

def _days_between(start: str, end: str) -> int | None:
    try:
        return int((pd.Timestamp(end) - pd.Timestamp(start)).days)
    except Exception:
        return None
