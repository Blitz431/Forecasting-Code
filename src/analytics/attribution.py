"""Per-stock P&L attribution.

Given a set of positions (entry price, current/exit price, shares held),
compute how much each position contributed to total portfolio P&L.

Usage
-----
    from src.analytics.attribution import compute_attribution, AttributionEntry

    positions = {
        "AAPL": {"shares": 10, "entry_price": 150.0, "current_price": 180.0},
        "JPM":  {"shares": 5,  "entry_price": 140.0, "current_price": 130.0},
    }
    results = compute_attribution(positions, portfolio_value=30000.0)
    for r in results:
        print(r.ticker, r.pnl, r.contribution_pct)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class AttributionEntry:
    ticker: str
    shares: float
    entry_price: float
    current_price: float
    cost_basis: float           # shares * entry_price
    market_value: float         # shares * current_price
    pnl: float                  # market_value - cost_basis
    pnl_pct: float              # pnl / cost_basis
    portfolio_weight: float     # market_value / total_portfolio_value
    contribution_pct: float     # pnl / total_portfolio_value (pnl attribution)
    sector: str = "Unknown"


def compute_attribution(
    positions: dict[str, dict],
    portfolio_value: float | None = None,
    sector_map: dict[str, str] | None = None,
) -> list[AttributionEntry]:
    """Compute P&L attribution for a set of positions.

    Parameters
    ----------
    positions:
        Dict mapping ticker -> {shares, entry_price, current_price}.
    portfolio_value:
        Total portfolio value (cash + market value).  If None, computed as
        sum of position market values.
    sector_map:
        Optional {ticker: sector} mapping for sector breakdown.

    Returns
    -------
    List of AttributionEntry, sorted by |pnl| descending.
    """
    if not positions:
        return []

    entries: list[AttributionEntry] = []
    total_market_value = sum(
        v["shares"] * v["current_price"] for v in positions.values()
    )
    if portfolio_value is None:
        portfolio_value = total_market_value
    if portfolio_value <= 0:
        portfolio_value = max(total_market_value, 1.0)

    for ticker, pos in positions.items():
        shares = float(pos.get("shares", 0))
        entry  = float(pos.get("entry_price", 0))
        cur    = float(pos.get("current_price", 0))

        if shares <= 0 or entry <= 0:
            continue

        cost   = shares * entry
        mval   = shares * cur
        pnl    = mval - cost
        pnl_pct = pnl / cost if cost > 0 else 0.0
        weight  = mval / portfolio_value
        contrib = pnl / portfolio_value

        sector = (sector_map or {}).get(ticker, "Unknown")

        entries.append(AttributionEntry(
            ticker=ticker,
            shares=shares,
            entry_price=entry,
            current_price=cur,
            cost_basis=cost,
            market_value=mval,
            pnl=pnl,
            pnl_pct=pnl_pct,
            portfolio_weight=weight,
            contribution_pct=contrib,
            sector=sector,
        ))

    return sorted(entries, key=lambda e: -abs(e.pnl))


def attribution_dataframe(entries: list[AttributionEntry]) -> pd.DataFrame:
    """Convert attribution entries to a styled DataFrame for display."""
    if not entries:
        return pd.DataFrame()

    rows = []
    for e in entries:
        rows.append({
            "Ticker":        e.ticker,
            "Sector":        e.sector,
            "Shares":        e.shares,
            "Entry Price":   round(e.entry_price, 2),
            "Current Price": round(e.current_price, 2),
            "Cost Basis":    round(e.cost_basis, 2),
            "Market Value":  round(e.market_value, 2),
            "P&L ($)":       round(e.pnl, 2),
            "P&L %":         round(e.pnl_pct * 100, 2),
            "Weight %":      round(e.portfolio_weight * 100, 2),
            "Contribution %": round(e.contribution_pct * 100, 3),
        })

    return pd.DataFrame(rows)


def sector_attribution(entries: list[AttributionEntry]) -> pd.DataFrame:
    """Aggregate attribution by sector."""
    if not entries:
        return pd.DataFrame()

    from collections import defaultdict
    bucket: dict[str, dict] = defaultdict(lambda: {"pnl": 0.0, "contribution": 0.0,
                                                     "value": 0.0, "tickers": []})
    for e in entries:
        s = e.sector
        bucket[s]["pnl"] += e.pnl
        bucket[s]["contribution"] += e.contribution_pct
        bucket[s]["value"] += e.market_value
        bucket[s]["tickers"].append(e.ticker)

    rows = []
    for sector, data in sorted(bucket.items(), key=lambda x: -abs(x[1]["pnl"])):
        rows.append({
            "Sector":         sector,
            "P&L ($)":        round(data["pnl"], 2),
            "Contribution %": round(data["contribution"] * 100, 3),
            "Market Value":   round(data["value"], 2),
            "Tickers":        ", ".join(data["tickers"]),
        })

    return pd.DataFrame(rows)
