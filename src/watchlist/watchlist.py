"""Manual watchlist manager with multi-timeframe analysis.

Persists the watchlist as a JSON file under data/watchlist.json.
Computes daily + weekly composite signals for each watched ticker.

Usage
-----
    from src.watchlist.watchlist import WatchlistManager

    wm = WatchlistManager(settings)
    wm.add("TSLA")
    wm.add("NVDA")

    df = wm.get_signals(timeframe="daily")   # or "weekly"
    print(df)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_WATCHLIST_FILE = "watchlist.json"


class WatchlistManager:
    """Persistent watchlist with multi-timeframe signal computation."""

    def __init__(self, settings=None, watchlist_path: Path | None = None):
        self._settings = settings
        if watchlist_path is not None:
            self._path = watchlist_path
        elif settings is not None:
            self._path = settings.data_dir / _DEFAULT_WATCHLIST_FILE
        else:
            self._path = Path(_DEFAULT_WATCHLIST_FILE)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> list[str]:
        """Return list of tickers currently in the watchlist."""
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text())
            if isinstance(raw, list):
                return [str(t).upper().strip() for t in raw if t]
            return []
        except Exception:
            return []

    def save(self, tickers: list[str]) -> None:
        """Persist the watchlist."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        clean = sorted(set(t.upper().strip() for t in tickers if t))
        self._path.write_text(json.dumps(clean, indent=2))

    def add(self, ticker: str) -> bool:
        """Add a ticker.  Returns True if added, False if already present."""
        tickers = self.load()
        t = ticker.upper().strip()
        if t in tickers:
            return False
        tickers.append(t)
        self.save(tickers)
        logger.info(f"Watchlist: added {t}")
        return True

    def remove(self, ticker: str) -> bool:
        """Remove a ticker.  Returns True if removed, False if not found."""
        tickers = self.load()
        t = ticker.upper().strip()
        if t not in tickers:
            return False
        tickers.remove(t)
        self.save(tickers)
        logger.info(f"Watchlist: removed {t}")
        return True

    def clear(self) -> None:
        self.save([])

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _load_daily(self, ticker: str) -> pd.DataFrame | None:
        if self._settings is None:
            return None
        fp = self._settings.raw_daily_dir / f"{ticker}.parquet"
        if not fp.exists():
            return None
        try:
            return pd.read_parquet(fp).sort_index()
        except Exception:
            return None

    def _resample_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample daily OHLCV to weekly (Mon-Fri) bars."""
        agg = {
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
        }
        if "Volume" in df.columns:
            agg["Volume"] = "sum"
        return df.resample("W-FRI").agg(agg).dropna(subset=["Close"])

    def _simple_signals(self, df: pd.DataFrame) -> dict:
        """Compute a quick signal summary from price data.

        Returns dict with: rsi, sma_cross, momentum, trend, composite_score, signal_label.
        """
        if df is None or len(df) < 50:
            return {}

        closes = df["Close"].dropna()

        result: dict = {}

        # RSI(14)
        delta = closes.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        ll = float(loss.iloc[-1]) if len(loss) > 0 else 0.001
        lg = float(gain.iloc[-1]) if len(gain) > 0 else 0.0
        rs = lg / ll if ll > 0 else 1.0
        rsi = 100 - (100 / (1 + rs))
        result["RSI"] = round(rsi, 1)

        # SMA 20 vs 50 crossover
        sma20 = float(closes.tail(20).mean())
        sma50 = float(closes.tail(50).mean())
        result["SMA20"] = round(sma20, 2)
        result["SMA50"] = round(sma50, 2)
        result["SMA Cross"] = "Bullish" if sma20 > sma50 else "Bearish"

        # Price vs 200d MA
        if len(closes) >= 200:
            sma200 = float(closes.tail(200).mean())
            cur    = float(closes.iloc[-1])
            result["SMA200"] = round(sma200, 2)
            result["Above 200d MA"] = cur > sma200
        else:
            result["Above 200d MA"] = None

        # 20-day momentum
        if len(closes) >= 20:
            mom = (float(closes.iloc[-1]) / float(closes.iloc[-20]) - 1) * 100
            result["20d Momentum %"] = round(mom, 2)
        else:
            result["20d Momentum %"] = None

        # 52-week high/low
        if len(closes) >= 252:
            hi52  = float(closes.tail(252).max())
            lo52  = float(closes.tail(252).min())
            cur52 = float(closes.iloc[-1])
            result["52w High"] = round(hi52, 2)
            result["52w Low"]  = round(lo52, 2)
            result["52w High %"] = round((cur52 / hi52 - 1) * 100, 1)

        # Composite score
        score = 0.0
        if rsi > 50:
            score += 1.0
        if rsi > 60:
            score += 0.5
        if sma20 > sma50:
            score += 1.5
        if result.get("Above 200d MA"):
            score += 1.0
        mom_val = result.get("20d Momentum %", 0) or 0
        if mom_val > 5:
            score += 1.0
        elif mom_val < -5:
            score -= 1.0
        if rsi > 70:
            score -= 0.5   # overbought penalty
        if rsi < 30:
            score -= 0.5

        result["Score"] = round(score, 2)
        if score >= 3:
            result["Signal"] = "Strong Buy"
        elif score >= 1.5:
            result["Signal"] = "Buy"
        elif score <= -1.5:
            result["Signal"] = "Sell"
        elif score <= -0.5:
            result["Signal"] = "Weak Sell"
        else:
            result["Signal"] = "Neutral"

        result["Current Price"] = round(float(closes.iloc[-1]), 2)
        return result

    def get_signals(
        self,
        tickers: list[str] | None = None,
        timeframe: str = "daily",
    ) -> pd.DataFrame:
        """Compute signals for all (or specified) watchlist tickers.

        Parameters
        ----------
        tickers:
            Subset of watchlist tickers.  Defaults to all.
        timeframe:
            "daily" or "weekly".

        Returns
        -------
        DataFrame with one row per ticker.
        """
        if tickers is None:
            tickers = self.load()

        rows = []
        for ticker in tickers:
            df = self._load_daily(ticker)
            if df is None or df.empty:
                rows.append({"Ticker": ticker, "Signal": "No Data", "Score": None})
                continue
            if timeframe == "weekly":
                df = self._resample_weekly(df)
            sig = self._simple_signals(df)
            if not sig:
                rows.append({"Ticker": ticker, "Signal": "Insufficient Data", "Score": None})
                continue
            row = {"Ticker": ticker}
            row.update(sig)
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df_out = pd.DataFrame(rows)
        # Sort by Score desc, put no-data tickers at bottom
        df_out["_sort"] = pd.to_numeric(df_out.get("Score"), errors="coerce").fillna(-999)
        df_out = df_out.sort_values("_sort", ascending=False).drop(columns=["_sort"])
        return df_out.reset_index(drop=True)

    def get_multi_timeframe(self, ticker: str) -> dict[str, dict]:
        """Return signal dict for both daily and weekly timeframes for one ticker."""
        df = self._load_daily(ticker)
        if df is None or df.empty:
            return {"daily": {}, "weekly": {}}
        df_weekly = self._resample_weekly(df)
        return {
            "daily":  self._simple_signals(df),
            "weekly": self._simple_signals(df_weekly),
        }
