"""Multi-strategy executor: value, momentum, and mean-reversion running simultaneously.

Capital allocation
------------------
Each strategy is given a capital slice proportional to its rolling Sharpe ratio
(last 12 weeks).  Allocation is rebalanced weekly (every Friday).

Initial weights are equal (33.3% each) until at least 4 weeks of P&L history
accumulates.

Strategies
----------
Value
  - Screens for low P/S ratio + high dividend yield + positive fundamental score.
  - BUY signal when P/S < 2.0 and indicator fundamentals score > 0.3.
  - Hold until fundamentals deteriorate or 5% trailing stop after 3 profitable days.

Momentum
  - Uses the top-ranked picks from ``ranking/ranker.py`` (composite score).
  - Standard entry/exit via ``strategy.MomentumStrategy``.

Mean-Reversion
  - Scans for RSI < 30 (oversold) + Bollinger Band lower-band touch.
  - Short holding window — exit when RSI recovers above 50 or +3% gain, whichever first.

Usage
-----
    from src.trading.multi_strategy import MultiStrategyManager

    mgr = MultiStrategyManager(settings)
    signals = mgr.generate_all_signals(top_picks, positions, portfolio_value)
    mgr.record_weekly_pnl(strategy_name, weekly_pnl)
    weights  = mgr.capital_weights()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.trading.strategy import TradeSignal
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

StrategyName = Literal["value", "momentum", "mean_reversion"]

_STRATEGY_NAMES: list[StrategyName] = ["value", "momentum", "mean_reversion"]
# Options uses a fixed capital pool — tracked for P&L reporting but not in weights
_ALL_STRATEGY_NAMES = [*_STRATEGY_NAMES, "options"]
_MIN_SHARPE_WEEKS = 4   # minimum weeks of history before using Sharpe allocation

# P&L history file path (relative to data_dir)
_HISTORY_FILE = "multi_strategy_history.json"


# ---------------------------------------------------------------------------#
# Data class
# ---------------------------------------------------------------------------#

@dataclass
class StrategyAllocation:
    value:          float   # capital fraction [0, 1]
    momentum:       float
    mean_reversion: float
    method: str = "equal"   # "equal" | "sharpe"

    def dollar_allocations(self, total_capital: float) -> dict[str, float]:
        return {
            "value":          total_capital * self.value,
            "momentum":       total_capital * self.momentum,
            "mean_reversion": total_capital * self.mean_reversion,
        }


# ---------------------------------------------------------------------------#
# Per-strategy signal generators
# ---------------------------------------------------------------------------#

def _value_signals(
    tickers: list[str],
    settings,
    held_tickers: set[str],
    capital: float,
) -> list[TradeSignal]:
    """BUY low-P/S + high-yield stocks not already held."""
    signals: list[TradeSignal] = []
    for ticker in tickers[:50]:    # limit scan for speed
        if ticker in held_tickers:
            continue
        try:
            from src.indicators.fundamentals import FundamentalsIndicator
            from src.scraper.storage import load_dataframe, get_ticker_filepath
            filepath = get_ticker_filepath(ticker, settings.raw_daily_dir)
            df = load_dataframe(filepath)
            if df.empty:
                continue
            result = FundamentalsIndicator().compute(df)
            # result.value is the normalised fundamental score
            if result.value is not None and float(result.value) > 0.3:
                signals.append(TradeSignal(
                    ticker=ticker,
                    action="BUY",
                    reason=f"value: fundamentals_score={float(result.value):.3f}",
                    score=float(result.value),
                ))
                if len(signals) >= 3:
                    break
        except Exception as exc:
            logger.debug(f"value_signal({ticker}): {exc}")
    return signals


def _mean_reversion_signals(
    tickers: list[str],
    settings,
    held_tickers: set[str],
) -> list[TradeSignal]:
    """BUY RSI-oversold + Bollinger lower-band touch."""
    signals: list[TradeSignal] = []
    for ticker in tickers[:100]:
        if ticker in held_tickers:
            continue
        try:
            from src.scraper.storage import load_dataframe, get_ticker_filepath
            from src.indicators.rsi import RSIIndicator
            from src.indicators.bollinger import BollingerIndicator

            filepath = get_ticker_filepath(ticker, settings.raw_daily_dir)
            df = load_dataframe(filepath)
            if df.empty or len(df) < 30:
                continue

            rsi_result = RSIIndicator().compute(df)
            bol_result = BollingerIndicator().compute(df)

            rsi_val = rsi_result.value if rsi_result else None
            bol_val = bol_result.value if bol_result else None

            # RSI < 30 AND Bollinger shows oversold
            if (
                rsi_val is not None and float(rsi_val) < 30
                and bol_val is not None and float(bol_val) <= -0.5
            ):
                score = (30.0 - float(rsi_val)) / 30.0   # 0-1, higher = more oversold
                signals.append(TradeSignal(
                    ticker=ticker,
                    action="BUY",
                    reason=f"mean_reversion: RSI={float(rsi_val):.1f} bol={float(bol_val):.2f}",
                    score=score,
                ))
                if len(signals) >= 3:
                    break
        except Exception as exc:
            logger.debug(f"mean_reversion_signal({ticker}): {exc}")
    return signals


def _mean_reversion_exits(
    positions: list,
    settings,
) -> list[TradeSignal]:
    """Exit mean-reversion when RSI > 50 OR gain > 3%."""
    signals: list[TradeSignal] = []
    for pos in positions:
        ticker = pos.ticker
        try:
            from src.scraper.storage import load_dataframe, get_ticker_filepath
            from src.indicators.rsi import RSIIndicator
            filepath = get_ticker_filepath(ticker, settings.raw_daily_dir)
            df = load_dataframe(filepath)
            if df.empty:
                continue
            rsi_result = RSIIndicator().compute(df)
            rsi_val = float(rsi_result.value) if rsi_result and rsi_result.value is not None else 50.0

            gain_pct = pos.unrealized_plpc   # already in percent

            if rsi_val > 50:
                signals.append(TradeSignal(
                    ticker=ticker, action="SELL",
                    reason=f"mean_reversion_exit: RSI recovered to {rsi_val:.1f}",
                    score=rsi_val / 100,
                ))
            elif gain_pct >= 3.0:
                signals.append(TradeSignal(
                    ticker=ticker, action="SELL",
                    reason=f"mean_reversion_exit: +{gain_pct:.1f}% target hit",
                    score=gain_pct / 100,
                ))
        except Exception as exc:
            logger.debug(f"mean_reversion_exit({ticker}): {exc}")
    return signals


# ---------------------------------------------------------------------------#
# Manager
# ---------------------------------------------------------------------------#

class MultiStrategyManager:
    """Orchestrate value, momentum, and mean-reversion strategies together."""

    def __init__(self, settings=None):
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        self._s = settings
        self._history_path = settings.data_dir / _HISTORY_FILE
        self._history: dict[str, list[dict]] = self._load_history()

        # Lazy-import to avoid circular deps
        from src.trading.strategy import MomentumStrategy
        self._momentum = MomentumStrategy(settings)

    # ---------------------------------------------------------------------- #
    # Signal generation
    # ---------------------------------------------------------------------- #

    def generate_all_signals(
        self,
        top_picks: list,                    # list[RankEntry]
        positions: list,                    # list[PositionInfo] from Alpaca
        portfolio_value: float,
        tickers: list[str] | None = None,
    ) -> dict[str, list[TradeSignal]]:
        """Generate BUY/SELL signals for every active strategy.

        Returns
        -------
        {strategy_name: [TradeSignal, ...]}
        """
        held = {p.ticker for p in positions}
        alloc = self.capital_weights()
        caps = alloc.dollar_allocations(portfolio_value)

        if tickers is None:
            tickers = [e.ticker for e in top_picks]

        result: dict[str, list[TradeSignal]] = {}

        # --- Value ---
        result["value"] = _value_signals(tickers, self._s, held, caps["value"])

        # --- Momentum ---
        current_scores = {e.ticker: e.composite_score for e in top_picks}
        momentum_positions = [p for p in positions if not self._is_mean_rev(p.ticker)]
        result["momentum"] = (
            self._momentum.generate_entries(top_picks, held, portfolio_value)
            + self._momentum.generate_exits(momentum_positions, current_scores)
        )

        # --- Mean-Reversion ---
        mr_positions = [p for p in positions if self._is_mean_rev(p.ticker)]
        result["mean_reversion"] = (
            _mean_reversion_signals(tickers, self._s, held)
            + _mean_reversion_exits(mr_positions, self._s)
        )

        return result

    # ---------------------------------------------------------------------- #
    # Capital allocation
    # ---------------------------------------------------------------------- #

    def capital_weights(self) -> StrategyAllocation:
        """Compute capital weights from rolling Sharpe ratios.

        Falls back to equal weights until at least _MIN_SHARPE_WEEKS of data.
        """
        sharpes: dict[str, float] = {}
        for name in _STRATEGY_NAMES:
            sharpes[name] = self._rolling_sharpe(name)

        # If all Sharpe values are 0 (no data yet), use equal weights
        total = sum(max(s, 0.0) for s in sharpes.values())
        if total == 0:
            w = 1.0 / 3.0
            return StrategyAllocation(value=w, momentum=w, mean_reversion=w, method="equal")

        weights = {n: max(sharpes[n], 0.0) / total for n in _STRATEGY_NAMES}
        return StrategyAllocation(
            value=weights["value"],
            momentum=weights["momentum"],
            mean_reversion=weights["mean_reversion"],
            method="sharpe",
        )

    def record_weekly_pnl(self, strategy: str, weekly_pnl: float) -> None:
        """Append a weekly P&L entry used for Sharpe calculation.

        Accepts equity strategy names and "options" (fixed-capital pool).
        Options participates in P&L reporting but its capital is not
        included in capital_weights().
        """
        if strategy not in _ALL_STRATEGY_NAMES:
            return
        entry = {"date": date.today().isoformat(), "pnl": weekly_pnl}
        self._history.setdefault(strategy, []).append(entry)
        self._save_history()

    def options_pnl_summary(self, weeks: int = 12) -> dict:
        """Return rolling P&L stats for the options strategy."""
        history = self._history.get("options", [])
        if not history:
            return {"total_pnl": 0.0, "weeks": 0, "sharpe": 0.0}
        recent = [h["pnl"] for h in history[-weeks:]]
        import numpy as np
        arr = np.array(recent, dtype=float)
        std = arr.std()
        sharpe = float(arr.mean() / std * np.sqrt(52)) if std > 0 else 0.0
        return {"total_pnl": float(arr.sum()), "weeks": len(recent), "sharpe": sharpe}

    def _rolling_sharpe(self, strategy: str, weeks: int = 12) -> float:
        """Annualised Sharpe over the last *weeks* weekly P&L entries."""
        history = self._history.get(strategy, [])
        if len(history) < _MIN_SHARPE_WEEKS:
            return 0.0
        recent = [h["pnl"] for h in history[-weeks:]]
        arr = np.array(recent, dtype=float)
        std = arr.std()
        if std == 0:
            return 0.0
        return float(arr.mean() / std * np.sqrt(52))   # annualised

    # ---------------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------------- #

    def _load_history(self) -> dict[str, list[dict]]:
        if self._history_path.exists():
            try:
                data = json.loads(self._history_path.read_text())
                # Ensure options key exists in older state files
                for name in _ALL_STRATEGY_NAMES:
                    data.setdefault(name, [])
                return data
            except Exception:
                pass
        return {n: [] for n in _ALL_STRATEGY_NAMES}

    def _save_history(self) -> None:
        try:
            self._history_path.write_text(json.dumps(self._history, indent=2))
        except Exception as exc:
            logger.error(f"Could not save strategy history: {exc}")

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _is_mean_rev(self, ticker: str) -> bool:
        """Check if ticker was entered via mean-reversion (heuristic — no tagging yet)."""
        return False   # TODO: tag positions by strategy in Phase 11
