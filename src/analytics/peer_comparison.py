"""Relative strength comparison of a stock vs its sector peers.

Usage
-----
    from src.analytics.peer_comparison import PeerComparison

    pc = PeerComparison(settings)
    df = pc.compare("AAPL", lookback_days=63)
    # Returns DataFrame: ticker, sector, period_return, vs_peer_avg, percentile_rank
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analytics.sector_analysis import _STATIC_SECTORS


# ---------------------------------------------------------------------------#
# Sector → peer tickers mapping (S&P 500 sample, top 15 per sector)
# ---------------------------------------------------------------------------#

_SECTOR_PEERS: dict[str, list[str]] = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "INTC", "AMD",
        "QCOM", "TXN", "AMAT", "LRCX", "ADI", "MU", "KLAC",
    ],
    "Communication Services": [
        "META", "GOOGL", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
        "CHTR", "EA", "TTWO", "OMC", "IPG",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX",
        "BKNG", "CMG", "YUM", "MAR", "HLT", "RCL", "CCL",
    ],
    "Consumer Staples": [
        "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ",
        "CL", "EL", "STZ", "KR", "SYY",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C",
        "USB", "PNC", "V", "MA", "COF", "TFC", "SCHW",
    ],
    "Health Care": [
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "ABT", "TMO", "DHR",
        "PFE", "BMY", "AMGN", "MDT", "SYK", "ISRG", "ELV",
    ],
    "Industrials": [
        "GE", "CAT", "HON", "UPS", "RTX", "LMT", "DE", "BA",
        "MMM", "GD", "FDX", "NSC", "UNP", "ETN", "EMR",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO",
        "PXD", "OXY", "DVN", "HAL", "BKR",
    ],
    "Real Estate": [
        "AMT", "PLD", "CCI", "EQIX", "PSA", "O", "WELL", "DLR",
        "SPG", "EQR", "AVB", "VTR",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "PCG",
        "ED", "ETR", "FE",
    ],
    "Materials": [
        "LIN", "APD", "SHW", "FCX", "NEM", "ECL", "PPG", "ALB",
        "VMC", "MLM", "IP",
    ],
}


@dataclass
class PeerEntry:
    ticker: str
    sector: str
    period_return: float        # e.g. 0.12 = +12%
    vs_peer_avg: float          # outperformance vs peer average return
    percentile_rank: float      # 0-100, higher is better
    is_subject: bool = False    # True for the ticker being analysed


class PeerComparison:
    """Compare a stock against its sector peers over a rolling lookback window."""

    def __init__(self, settings=None):
        self._settings = settings

    def _load_prices(self, ticker: str, data_dir: Path | None) -> pd.Series | None:
        """Return daily Close series from parquet, or None."""
        if data_dir is None:
            return None
        fp = data_dir / f"{ticker}.parquet"
        if not fp.exists():
            return None
        try:
            df = pd.read_parquet(fp)
            return df["Close"].dropna() if "Close" in df.columns else None
        except Exception:
            return None

    def _get_sector(self, ticker: str) -> str:
        return _STATIC_SECTORS.get(ticker, "Unknown")

    def _get_peers(self, ticker: str) -> list[str]:
        sector = self._get_sector(ticker)
        peers = _SECTOR_PEERS.get(sector, [])
        # Ensure subject ticker is included
        if ticker not in peers:
            peers = [ticker] + peers[:14]
        return peers

    def compare(
        self,
        ticker: str,
        lookback_days: int = 63,
        data_dir: Path | None = None,
        max_peers: int = 15,
    ) -> pd.DataFrame:
        """Return relative strength table for *ticker* vs sector peers.

        Parameters
        ----------
        ticker:
            Subject ticker.
        lookback_days:
            Rolling return window in calendar days (~63 = 1 quarter).
        data_dir:
            Directory of daily parquet files.  Defaults to settings path.
        max_peers:
            Maximum number of peer tickers to include.

        Returns
        -------
        DataFrame with columns: Ticker, Sector, Return%, vs Peer Avg, Percentile
        """
        if data_dir is None and self._settings is not None:
            data_dir = self._settings.raw_daily_dir

        peers = self._get_peers(ticker)[:max_peers]

        # Compute period return for each peer
        returns: dict[str, float] = {}
        for t in peers:
            series = self._load_prices(t, data_dir)
            if series is None or len(series) < lookback_days:
                continue
            ret = float(series.iloc[-1] / series.iloc[-lookback_days] - 1)
            returns[t] = ret

        if not returns:
            return pd.DataFrame()

        peer_avg = float(np.mean(list(returns.values())))
        sorted_vals = sorted(returns.values())

        rows = []
        for t, ret in returns.items():
            vs_avg = ret - peer_avg
            # Percentile: fraction of peers with lower return
            pct = float(np.searchsorted(sorted_vals, ret)) / max(len(sorted_vals) - 1, 1) * 100
            rows.append({
                "Ticker":      t,
                "Sector":      self._get_sector(t),
                "Return%":     round(ret * 100, 2),
                "vs Peer Avg": round(vs_avg * 100, 2),
                "Percentile":  round(pct, 1),
                "Subject":     t == ticker,
            })

        df = pd.DataFrame(rows).sort_values("Return%", ascending=False)
        return df.reset_index(drop=True)

    def relative_strength_score(self, ticker: str, lookback_days: int = 63,
                                 data_dir: Path | None = None) -> float:
        """Return a score in [-1, +1] for the ticker's relative strength.

        +1 = top of its sector, -1 = bottom.
        """
        df = self.compare(ticker, lookback_days=lookback_days, data_dir=data_dir)
        if df.empty:
            return 0.0
        row = df[df["Ticker"] == ticker]
        if row.empty:
            return 0.0
        pct = float(row["Percentile"].iloc[0])   # 0-100
        return (pct / 50.0) - 1.0                # map to [-1, +1]
