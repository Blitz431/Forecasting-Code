from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

"""
Purpose: Sector rotation detection and concentration warnings — identifies over-weight sectors across current holdings.

Connections:
  - src/analytics/peer_comparison.py: imports _STATIC_SECTORS for peer mapping
  - config/settings.py: raw_daily_dir for sector return calculations
  - dashboard: called to show sector allocation breakdown

In:  dict of {ticker: market_value} holdings; reads yfinance .info for sector tags (cached)
Out: sector_weights dict, rotation signals, concentration warning list
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------#
# Static fallback sector map (S&P 500 representative sample)
# Keeps the module usable even without a network connection.
# ---------------------------------------------------------------------------#

_STATIC_SECTORS: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "ORCL": "Technology", "CSCO": "Technology",
    "ACN": "Technology", "IBM": "Technology", "INTC": "Technology",
    "AMD": "Technology", "QCOM": "Technology", "TXN": "Technology",
    "AMAT": "Technology", "LRCX": "Technology", "ADI": "Technology",
    "MU": "Technology", "KLAC": "Technology",
    # Communication Services
    "META": "Communication Services", "GOOGL": "Communication Services",
    "GOOG": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "CMCSA": "Communication Services",
    "T": "Communication Services", "VZ": "Communication Services",
    "TMUS": "Communication Services", "CHTR": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "CMG": "Consumer Discretionary",
    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "WMT": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "MDLZ": "Consumer Staples", "CL": "Consumer Staples",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "BLK": "Financials",
    "AXP": "Financials", "C": "Financials", "USB": "Financials",
    "PNC": "Financials", "V": "Financials", "MA": "Financials",
    # Health Care
    "UNH": "Health Care", "JNJ": "Health Care", "LLY": "Health Care",
    "ABBV": "Health Care", "MRK": "Health Care", "ABT": "Health Care",
    "TMO": "Health Care", "DHR": "Health Care", "PFE": "Health Care",
    "BMY": "Health Care", "AMGN": "Health Care", "MDT": "Health Care",
    # Industrials
    "GE": "Industrials", "CAT": "Industrials", "HON": "Industrials",
    "UPS": "Industrials", "RTX": "Industrials", "LMT": "Industrials",
    "DE": "Industrials", "BA": "Industrials", "MMM": "Industrials",
    "GD": "Industrials", "FDX": "Industrials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "EOG": "Energy",
    "SLB": "Energy", "MPC": "Energy", "PSX": "Energy", "VLO": "Energy",
    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "PSA": "Real Estate", "O": "Real Estate",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "AEP": "Utilities", "EXC": "Utilities",
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "FCX": "Materials", "NEM": "Materials", "ECL": "Materials",
    # Broad market ETFs
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "GLD": "Commodities", "SLV": "Commodities",
}

CONCENTRATION_WARN_PCT = 35.0   # warn if one sector exceeds this % of portfolio


@dataclass
class SectorWeight:
    sector: str
    value: float        # dollar value
    weight: float       # fraction 0-1
    tickers: list[str]


class SectorAnalyzer:
    """Sector allocation, rotation, and concentration analysis."""

    def __init__(self, settings=None):
        self._settings = settings
        self._cache_path = (
            settings.data_dir / "sector_cache.json" if settings else None
        )
        self._sector_map: dict[str, str] = dict(_STATIC_SECTORS)
        if self._cache_path and self._cache_path.exists():
            try:
                saved = json.loads(self._cache_path.read_text())
                self._sector_map.update(saved)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Sector lookup
    # ------------------------------------------------------------------

    def get_sector(self, ticker: str) -> str:
        """Return sector for a ticker. Falls back to yfinance if not cached."""
        if ticker in self._sector_map:
            return self._sector_map[ticker]
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown")
            self._sector_map[ticker] = sector
            self._save_cache()
            return sector
        except Exception:
            return "Unknown"

    def _save_cache(self) -> None:
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._sector_map))

    # ------------------------------------------------------------------
    # Portfolio sector weights
    # ------------------------------------------------------------------

    def sector_weights(
        self,
        positions: dict[str, float],   # {ticker: dollar_value}
    ) -> list[SectorWeight]:
        """Break down portfolio value by sector."""
        total = sum(positions.values()) or 1.0

        bucket: dict[str, list[str]] = {}
        value_map: dict[str, float] = {}

        for ticker, val in positions.items():
            sector = self.get_sector(ticker)
            bucket.setdefault(sector, []).append(ticker)
            value_map[sector] = value_map.get(sector, 0.0) + val

        result = []
        for sector, tickers in sorted(bucket.items()):
            v = value_map[sector]
            result.append(SectorWeight(
                sector=sector,
                value=v,
                weight=v / total,
                tickers=tickers,
            ))
        return sorted(result, key=lambda x: -x.weight)

    def sector_weights_series(self, positions: dict[str, float]) -> pd.Series:
        """Return sector weights as a Series (sector -> weight)."""
        sw = self.sector_weights(positions)
        return pd.Series({s.sector: s.weight for s in sw})

    # ------------------------------------------------------------------
    # Concentration warnings
    # ------------------------------------------------------------------

    def concentration_warnings(self, positions: dict[str, float]) -> list[str]:
        """Return list of warning strings if any sector exceeds threshold."""
        warnings = []
        for sw in self.sector_weights(positions):
            pct = sw.weight * 100
            if sw.sector not in ("ETF", "Unknown") and pct > CONCENTRATION_WARN_PCT:
                warnings.append(
                    f"{sw.sector}: {pct:.1f}% of portfolio "
                    f"(threshold {CONCENTRATION_WARN_PCT:.0f}%) — "
                    f"tickers: {', '.join(sw.tickers)}"
                )
        return warnings

    # ------------------------------------------------------------------
    # Sector rotation detection
    # ------------------------------------------------------------------

    def detect_rotation(
        self,
        data_dir: Path | None = None,
        lookback_days: int = 63,
    ) -> pd.DataFrame:
        """Compute recent vs prior-period returns for each sector ETF to detect rotation.

        Uses SPDR sector ETFs as proxies.  Returns a DataFrame with
        columns: sector, recent_return, prior_return, momentum (recent - prior).
        """
        if data_dir is None and self._settings is not None:
            data_dir = self._settings.raw_daily_dir

        SECTOR_ETFS: dict[str, str] = {
            "Technology":              "XLK",
            "Financials":              "XLF",
            "Health Care":             "XLV",
            "Consumer Discretionary":  "XLY",
            "Consumer Staples":        "XLP",
            "Industrials":             "XLI",
            "Energy":                  "XLE",
            "Utilities":               "XLU",
            "Real Estate":             "XLRE",
            "Materials":               "XLB",
            "Communication Services":  "XLC",
        }

        rows = []
        for sector, etf in SECTOR_ETFS.items():
            try:
                fp = data_dir / f"{etf}.parquet" if data_dir else None
                if fp and fp.exists():
                    df = pd.read_parquet(fp)
                else:
                    import yfinance as yf
                    df = yf.download(etf, period="1y", progress=False)

                if df.empty or "Close" not in df.columns:
                    continue

                closes = df["Close"].dropna()
                if len(closes) < lookback_days * 2:
                    continue

                recent_ret = float(closes.iloc[-1] / closes.iloc[-lookback_days] - 1)
                prior_ret  = float(closes.iloc[-lookback_days] / closes.iloc[-(lookback_days * 2)] - 1)
                rows.append({
                    "Sector": sector,
                    "ETF": etf,
                    "Recent Return": recent_ret,
                    "Prior Return": prior_ret,
                    "Momentum": recent_ret - prior_ret,
                })
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        df_rot = pd.DataFrame(rows).sort_values("Momentum", ascending=False)
        return df_rot.reset_index(drop=True)
