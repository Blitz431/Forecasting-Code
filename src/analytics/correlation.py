from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

"""
Purpose: Rolling correlation matrix across held tickers — identifies over-correlated positions for diversification.

Connections:
  - config/settings.py: raw_daily_dir for loading price data
  - src/trading/risk.py: reads highly correlated pairs to adjust position sizing
  - dashboard: called to show correlation heatmap panels

In:  list of ticker symbols + lookback_days; reads data/raw/daily/{ticker}.parquet
Out: correlation pd.DataFrame; highly_correlated() returns list[CorrelatedPair] above threshold
"""


@dataclass
class CorrelatedPair:
    ticker_a: str
    ticker_b: str
    correlation: float
    warning: str = ""


class CorrelationAnalyzer:
    """Compute return correlations across a list of tickers."""

    def __init__(self, settings=None):
        self._settings = settings

    def _load_closes(self, tickers: list[str], lookback_days: int,
                     data_dir: Path | None) -> pd.DataFrame:
        """Load Close prices for all tickers, aligned on common dates."""
        frames: list[pd.Series] = []
        for t in tickers:
            if data_dir is None:
                continue
            fp = data_dir / f"{t}.parquet"
            if not fp.exists():
                continue
            try:
                df = pd.read_parquet(fp)
                if "Close" not in df.columns:
                    continue
                s = df["Close"].dropna().tail(lookback_days + 1).rename(t)
                frames.append(s)
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).dropna()

    def compute(
        self,
        tickers: list[str],
        lookback_days: int = 63,
        data_dir: Path | None = None,
    ) -> pd.DataFrame:
        """Return a symmetric correlation matrix as a DataFrame.

        Parameters
        ----------
        tickers:
            List of holdings to correlate.
        lookback_days:
            Number of daily return observations to use (~63 = 1 quarter).
        data_dir:
            Directory of daily parquet files.

        Returns
        -------
        Square DataFrame: index=tickers, columns=tickers, values=correlation.
        Empty DataFrame if fewer than 2 tickers have data.
        """
        if data_dir is None and self._settings is not None:
            data_dir = self._settings.raw_daily_dir

        price_df = self._load_closes(tickers, lookback_days, data_dir)
        if price_df.shape[1] < 2:
            return pd.DataFrame()

        returns = price_df.pct_change().dropna()
        return returns.corr()

    def highly_correlated(
        self,
        corr_df: pd.DataFrame,
        threshold: float = 0.80,
    ) -> list[CorrelatedPair]:
        """Find pairs with |correlation| >= threshold.

        Returns list of CorrelatedPair objects, each with a warning message.
        """
        if corr_df.empty:
            return []

        tickers = list(corr_df.columns)
        pairs: list[CorrelatedPair] = []

        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                a, b = tickers[i], tickers[j]
                corr = float(corr_df.loc[a, b])
                if abs(corr) >= threshold:
                    if corr > 0:
                        warning = (
                            f"{a} and {b} are {corr:.0%} correlated — "
                            "holding both provides limited diversification."
                        )
                    else:
                        warning = (
                            f"{a} and {b} are {corr:.0%} negatively correlated — "
                            "natural hedge, but review whether both align with strategy."
                        )
                    pairs.append(CorrelatedPair(a, b, corr, warning))

        return sorted(pairs, key=lambda p: -abs(p.correlation))

    def average_correlation(self, corr_df: pd.DataFrame) -> float:
        """Mean pairwise correlation (excluding diagonal)."""
        if corr_df.empty or corr_df.shape[0] < 2:
            return 0.0
        mask = ~np.eye(len(corr_df), dtype=bool)
        return float(corr_df.values[mask].mean())

    def diversification_ratio(self, corr_df: pd.DataFrame,
                               weights: pd.Series | None = None) -> float:
        """Diversification ratio: weighted vol sum / portfolio vol.

        Higher is better (>1 means diversification is working).
        Equal weights assumed if not provided.
        """
        if corr_df.empty or corr_df.shape[0] < 2:
            return 1.0

        n = len(corr_df)
        if weights is None:
            weights = pd.Series(1 / n, index=corr_df.index)
        else:
            weights = weights.reindex(corr_df.index).fillna(0)
            total = weights.sum()
            if total > 0:
                weights = weights / total

        # Assume unit volatility per asset for ratio computation
        w = weights.values
        corr = corr_df.values

        portfolio_var = float(w @ corr @ w)
        if portfolio_var <= 0:
            return 1.0

        weighted_vol_sum = float(w.sum())   # sum of weights * 1.0 vol
        return weighted_vol_sum / np.sqrt(portfolio_var)
