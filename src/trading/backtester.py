"""Paper-trading backtester: replay 2020-2026 day-by-day with no future data leakage.

Strategy
--------
- Train window  : configurable (default 2015-01-01 → 2019-12-31)
- Test window   : configurable (default 2020-01-01 → today)
- Signals       : technical indicators computed on only past data
- Rebalance     : weekly (every Friday close)
- Position size : equal-weight, max_position_pct of portfolio per stock
- Slippage      : flat BPS cost applied to every trade
- Spread        : half-spread added to every buy, subtracted from every sell
- Circuit breaker: halt all trading for the day if portfolio drops > circuit_pct
- Trailing stop : exit when position falls trailing_stop_pct below its peak

Output
------
BacktestResult with:
  - equity_curve  : daily portfolio value (pd.Series)
  - trades        : list of Trade
  - metrics       : PortfolioMetrics
  - spy_curve     : SPY buy-and-hold comparison
  - trade_log_df  : DataFrame summary of all trades
  - regime_overlay: daily regime labels (if available)

CLI
---
    python cli/backtest.py --mode full-sim --train 2015-2020 --test 2020-2026

Usage from Python
-----------------
    from src.trading.backtester import Backtester, BacktestConfig

    config = BacktestConfig()
    bt = Backtester(config, settings)
    result = bt.run()
    print(result.metrics.to_dict())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.analytics.portfolio_metrics import compute_metrics, PortfolioMetrics, drawdown_series

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------#
# Configuration
# ---------------------------------------------------------------------------#

@dataclass
class BacktestConfig:
    train_start: str = "2015-01-01"
    train_end: str   = "2019-12-31"
    test_start: str  = "2020-01-01"
    test_end: str    = "2026-12-31"

    initial_capital: float = 100_000.0
    max_position_pct: float = 0.05         # max 5% of portfolio per stock
    top_n: int = 20                         # hold top-N ranked stocks

    slippage_bps: float = 5.0              # one-way transaction cost (basis points)
    spread_bps: float = 2.0               # half-spread per side

    trailing_stop_pct: float = 0.05        # 5% trailing stop on each position
    circuit_breaker_pct: float = 0.10      # halt if portfolio drops 10% in a day

    rebalance_freq: str = "W-FRI"          # pandas offset alias for rebalancing

    spy_ticker: str = "SPY"


# ---------------------------------------------------------------------------#
# Data containers
# ---------------------------------------------------------------------------#

@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None
    entry_price: float
    exit_price: float | None
    shares: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""         # "rebalance", "trailing_stop", "circuit_breaker", "end_of_sim"
    peak_price: float = 0.0       # for trailing stop tracking


@dataclass
class BacktestResult:
    equity_curve: pd.Series = field(default_factory=pd.Series)
    spy_curve: pd.Series = field(default_factory=pd.Series)
    trades: list[Trade] = field(default_factory=list)
    metrics: PortfolioMetrics = field(default_factory=PortfolioMetrics)
    spy_metrics: PortfolioMetrics = field(default_factory=PortfolioMetrics)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)
    trade_log_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_overlay: pd.DataFrame = field(default_factory=pd.DataFrame)
    config: BacktestConfig = field(default_factory=BacktestConfig)

    # Summary stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    circuit_breaker_hits: int = 0


# ---------------------------------------------------------------------------#
# Signal engine (fast, no future leakage)
# ---------------------------------------------------------------------------#

def _compute_score(df_up_to_today: pd.DataFrame) -> float:
    """Compute a simple composite score from [0 .. ~5] using past data only.

    Uses:
      - 20-day vs 50-day SMA crossover
      - RSI(14) distance from 50
      - 10-day momentum
      - Price vs 200-day MA
    """
    if df_up_to_today is None or len(df_up_to_today) < 50:
        return 0.0

    closes = df_up_to_today["Close"].dropna()
    if len(closes) < 50:
        return 0.0

    score = 0.0

    # 1. SMA crossover
    sma20 = float(closes.iloc[-20:].mean())
    sma50 = float(closes.iloc[-50:].mean())
    if sma20 > sma50:
        score += 1.0 * ((sma20 - sma50) / sma50)  # proportional to gap

    # 2. RSI(14)
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    last_loss = float(loss.iloc[-1]) if len(loss) > 0 else 0.001
    last_gain = float(gain.iloc[-1]) if len(gain) > 0 else 0.0
    rs = last_gain / last_loss if last_loss > 0 else 1.0
    rsi = 100 - (100 / (1 + rs))
    score += (rsi - 50) / 100.0   # -0.5 to +0.5

    # 3. 10-day momentum
    if len(closes) >= 10:
        mom = (float(closes.iloc[-1]) / float(closes.iloc[-10])) - 1
        score += mom * 5           # amplify small momentum signal

    # 4. Price vs 200-day MA
    if len(closes) >= 200:
        ma200 = float(closes.iloc[-200:].mean())
        cur   = float(closes.iloc[-1])
        score += (cur - ma200) / ma200

    return score


def _apply_cost(price: float, config: BacktestConfig, side: str = "buy") -> float:
    """Return effective price after slippage and spread."""
    total_bps = config.slippage_bps + config.spread_bps
    multiplier = 1 + total_bps / 10_000
    if side == "buy":
        return price * multiplier
    else:
        return price * (2 - multiplier)   # sell at slight discount


# ---------------------------------------------------------------------------#
# Main Backtester
# ---------------------------------------------------------------------------#

class Backtester:
    """Replay the trading strategy day-by-day over the test window."""

    def __init__(self, config: BacktestConfig | None = None, settings=None):
        self.config   = config or BacktestConfig()
        self._settings = settings

    def _data_dir(self) -> Path | None:
        return self._settings.raw_daily_dir if self._settings else None

    def _load_price_data(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """Load Close/OHLCV data for all tickers into memory."""
        data_dir = self._data_dir()
        price_data: dict[str, pd.DataFrame] = {}
        for t in tickers:
            if data_dir is None:
                break
            fp = data_dir / f"{t}.parquet"
            if not fp.exists():
                continue
            try:
                df = pd.read_parquet(fp)
                if "Close" in df.columns:
                    price_data[t] = df.sort_index()
            except Exception:
                continue
        return price_data

    def _get_universe(self) -> list[str]:
        """Return ticker universe (tickers with data in raw_daily_dir)."""
        data_dir = self._data_dir()
        if data_dir is None or not data_dir.exists():
            return []
        files = list(data_dir.glob("*.parquet"))
        # Exclude sector ETFs and macro instruments
        excluded = {"SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "XLK", "XLF",
                    "XLV", "XLY", "XLP", "XLI", "XLE", "XLU", "XLRE", "XLB", "XLC"}
        return [f.stem for f in files if f.stem not in excluded]

    def _get_trading_days(self) -> pd.DatetimeIndex:
        """Return trading days in the test window using SPY data."""
        data_dir = self._data_dir()
        spy_fp = data_dir / f"{self.config.spy_ticker}.parquet" if data_dir else None
        if spy_fp and spy_fp.exists():
            spy = pd.read_parquet(spy_fp)
            mask = (spy.index >= self.config.test_start) & (spy.index <= self.config.test_end)
            return spy.index[mask]
        # Fallback: business days
        return pd.bdate_range(self.config.test_start, self.config.test_end)

    def _rebalance_dates(self, trading_days: pd.DatetimeIndex) -> set:
        """Return set of dates on which to rebalance (weekly Fridays)."""
        freq = self.config.rebalance_freq
        rebalance = pd.date_range(
            trading_days[0], trading_days[-1], freq=freq
        )
        # Snap to nearest trading day on or before the rebalance date
        snapped = set()
        td_set = pd.DatetimeIndex(sorted(trading_days))
        for rb in rebalance:
            # Find the last trading day <= rb
            candidates = td_set[td_set <= rb]
            if len(candidates) > 0:
                snapped.add(candidates[-1])
        return snapped

    def run(self, progress_callback: Callable[[int, int], None] | None = None
            ) -> BacktestResult:
        """Execute the full simulation and return BacktestResult."""

        cfg = self.config
        logger.info("Backtester: loading universe …")
        universe = self._get_universe()
        if not universe:
            logger.warning("No ticker data found — cannot run backtest.")
            return BacktestResult(config=cfg)

        logger.info(f"Backtester: loaded {len(universe)} tickers")
        price_data = self._load_price_data(universe + [cfg.spy_ticker])

        trading_days = self._get_trading_days()
        if len(trading_days) == 0:
            logger.warning("No trading days in test window.")
            return BacktestResult(config=cfg)

        rebalance_dates = self._rebalance_dates(trading_days)

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        cash = cfg.initial_capital
        positions: dict[str, float] = {}       # ticker -> shares held
        entry_prices: dict[str, float] = {}    # ticker -> average entry price
        peak_prices: dict[str, float] = {}     # ticker -> peak price (for trailing stop)

        equity_list:  list[tuple[pd.Timestamp, float]] = []
        spy_list:     list[tuple[pd.Timestamp, float]] = []
        completed_trades: list[Trade] = []
        circuit_breaker_hits = 0

        # SPY buy-and-hold tracking
        spy_data = price_data.get(cfg.spy_ticker)
        spy_initial = None

        logger.info(f"Backtester: simulating {len(trading_days)} trading days …")

        for day_idx, date in enumerate(trading_days):
            if progress_callback:
                progress_callback(day_idx, len(trading_days))

            # ------------------------------------------------------------------
            # Get current prices (using only today's open/close — no peeking)
            # ------------------------------------------------------------------
            cur_prices: dict[str, float] = {}
            for t, df in price_data.items():
                if date in df.index:
                    cur_prices[t] = float(df.loc[date, "Close"])

            if not cur_prices:
                continue

            # ------------------------------------------------------------------
            # SPY reference
            # ------------------------------------------------------------------
            spy_price = cur_prices.get(cfg.spy_ticker)
            if spy_price is not None:
                if spy_initial is None:
                    spy_initial = spy_price
                spy_list.append((date, spy_price))

            # ------------------------------------------------------------------
            # Compute portfolio value
            # ------------------------------------------------------------------
            portfolio_value = cash + sum(
                cur_prices.get(t, entry_prices[t]) * sh
                for t, sh in positions.items()
            )

            # ------------------------------------------------------------------
            # Circuit breaker check
            # ------------------------------------------------------------------
            cb_triggered = False
            if equity_list:
                prev_val = equity_list[-1][1]
                if prev_val > 0 and (portfolio_value - prev_val) / prev_val <= -cfg.circuit_breaker_pct:
                    logger.warning(f"Circuit breaker triggered on {date.date()}")
                    cb_triggered = True
                    circuit_breaker_hits += 1

            equity_list.append((date, portfolio_value))

            if cb_triggered:
                # Liquidate all positions
                for t, sh in list(positions.items()):
                    sell_price = _apply_cost(cur_prices.get(t, entry_prices[t]), cfg, "sell")
                    pnl = (sell_price - entry_prices[t]) * sh
                    cash += sell_price * sh
                    completed_trades.append(Trade(
                        ticker=t,
                        entry_date=date,    # approximate
                        exit_date=date,
                        entry_price=entry_prices[t],
                        exit_price=sell_price,
                        shares=sh,
                        pnl=pnl,
                        pnl_pct=pnl / (entry_prices[t] * sh),
                        exit_reason="circuit_breaker",
                        peak_price=peak_prices.get(t, entry_prices[t]),
                    ))
                positions.clear()
                entry_prices.clear()
                peak_prices.clear()
                continue

            # ------------------------------------------------------------------
            # Update peak prices for trailing stop
            # ------------------------------------------------------------------
            for t, sh in positions.items():
                cp = cur_prices.get(t, entry_prices.get(t, 0.0))
                if cp > peak_prices.get(t, 0.0):
                    peak_prices[t] = cp

            # ------------------------------------------------------------------
            # Trailing stop exits
            # ------------------------------------------------------------------
            to_exit: list[str] = []
            for t, sh in positions.items():
                cp    = cur_prices.get(t, entry_prices.get(t, 0.0))
                peak  = peak_prices.get(t, cp)
                if peak > 0 and (cp - peak) / peak <= -cfg.trailing_stop_pct:
                    to_exit.append(t)

            for t in to_exit:
                sh = positions.pop(t)
                ep = entry_prices.pop(t)
                pp = peak_prices.pop(t, ep)
                sell_price = _apply_cost(cur_prices.get(t, ep), cfg, "sell")
                pnl = (sell_price - ep) * sh
                completed_trades.append(Trade(
                    ticker=t,
                    entry_date=date,
                    exit_date=date,
                    entry_price=ep,
                    exit_price=sell_price,
                    shares=sh,
                    pnl=pnl,
                    pnl_pct=pnl / (ep * sh) if ep * sh > 0 else 0.0,
                    exit_reason="trailing_stop",
                    peak_price=pp,
                ))
                cash += sell_price * sh

            # ------------------------------------------------------------------
            # Weekly rebalance
            # ------------------------------------------------------------------
            if date in rebalance_dates:
                # Score all tickers using only data up to today
                scores: dict[str, float] = {}
                for t in universe:
                    df = price_data.get(t)
                    if df is None:
                        continue
                    df_past = df[df.index <= date]
                    # Need at least train_end of history before test_start
                    if len(df_past) < 60:
                        continue
                    scores[t] = _compute_score(df_past)

                if not scores:
                    continue

                # Select top-N
                top_tickers = sorted(scores, key=lambda x: -scores[x])[:cfg.top_n]

                # Sell positions no longer in top-N
                for t in list(positions.keys()):
                    if t not in top_tickers:
                        sh = positions.pop(t)
                        ep = entry_prices.pop(t)
                        peak_prices.pop(t, None)
                        sell_price = _apply_cost(cur_prices.get(t, ep), cfg, "sell")
                        pnl = (sell_price - ep) * sh
                        cash += sell_price * sh
                        completed_trades.append(Trade(
                            ticker=t,
                            entry_date=date,
                            exit_date=date,
                            entry_price=ep,
                            exit_price=sell_price,
                            shares=sh,
                            pnl=pnl,
                            pnl_pct=pnl / (ep * sh) if ep * sh > 0 else 0.0,
                            exit_reason="rebalance",
                            peak_price=peak_prices.get(t, ep),
                        ))

                # Recompute portfolio value after selling
                portfolio_value = cash + sum(
                    cur_prices.get(t, entry_prices[t]) * sh
                    for t, sh in positions.items()
                )

                # Buy top-N positions (equal weight, respect max_position_pct)
                target_per_stock = min(
                    portfolio_value / max(len(top_tickers), 1),
                    portfolio_value * cfg.max_position_pct,
                )

                for t in top_tickers:
                    if t in positions:
                        continue    # already held
                    cp = cur_prices.get(t)
                    if cp is None or cp <= 0:
                        continue
                    buy_price = _apply_cost(cp, cfg, "buy")
                    alloc = min(target_per_stock, cash * 0.99)
                    if alloc < buy_price:
                        continue
                    shares = alloc // buy_price
                    if shares <= 0:
                        continue
                    cost = shares * buy_price
                    cash -= cost
                    positions[t] = shares
                    entry_prices[t] = buy_price
                    peak_prices[t]  = cp

        # ------------------------------------------------------------------
        # Close all remaining positions at end of simulation
        # ------------------------------------------------------------------
        last_date = trading_days[-1] if len(trading_days) > 0 else pd.Timestamp.today()
        for t, sh in list(positions.items()):
            df = price_data.get(t)
            ep = entry_prices.get(t, 0.0)
            cp = ep
            if df is not None and last_date in df.index:
                cp = float(df.loc[last_date, "Close"])
            sell_price = _apply_cost(cp, cfg, "sell")
            pnl = (sell_price - ep) * sh
            cash += sell_price * sh
            completed_trades.append(Trade(
                ticker=t,
                entry_date=last_date,
                exit_date=last_date,
                entry_price=ep,
                exit_price=sell_price,
                shares=sh,
                pnl=pnl,
                pnl_pct=pnl / (ep * sh) if ep * sh > 0 else 0.0,
                exit_reason="end_of_sim",
                peak_price=peak_prices.get(t, ep),
            ))
        positions.clear()
        entry_prices.clear()

        # ------------------------------------------------------------------
        # Build result series
        # ------------------------------------------------------------------
        equity_curve = pd.Series(
            {d: v for d, v in equity_list},
            name="Portfolio Value",
        )
        equity_curve.index = pd.DatetimeIndex(equity_curve.index)

        if spy_initial and spy_list:
            spy_curve = pd.Series(
                {d: cfg.initial_capital * (p / spy_initial)
                 for d, p in spy_list},
                name="SPY Buy & Hold",
            )
            spy_curve.index = pd.DatetimeIndex(spy_curve.index)
        else:
            spy_curve = pd.Series(dtype=float)

        # Compute drawdown curve
        dd_curve = drawdown_series(equity_curve)

        # Compute metrics
        spy_bench = spy_curve if not spy_curve.empty else None
        metrics = compute_metrics(equity_curve, benchmark=spy_bench)
        spy_metrics = compute_metrics(spy_curve) if not spy_curve.empty else PortfolioMetrics()

        # Trade log DataFrame
        trade_rows = []
        winning = losing = 0
        for tr in completed_trades:
            if tr.pnl > 0:
                winning += 1
            elif tr.pnl < 0:
                losing += 1
            trade_rows.append({
                "Ticker":      tr.ticker,
                "Entry Date":  tr.entry_date,
                "Exit Date":   tr.exit_date,
                "Entry Price": round(tr.entry_price, 2),
                "Exit Price":  round(tr.exit_price, 2) if tr.exit_price else None,
                "Shares":      tr.shares,
                "P&L ($)":     round(tr.pnl, 2),
                "P&L %":       round(tr.pnl_pct * 100, 2),
                "Exit Reason": tr.exit_reason,
            })

        trade_log_df = pd.DataFrame(trade_rows)

        # Regime overlay (best-effort)
        try:
            from src.analytics.market_regime import MarketRegimeAnalyzer
            mra = MarketRegimeAnalyzer(self._settings)
            regime_overlay = mra.historical_regimes()
        except Exception:
            regime_overlay = pd.DataFrame()

        result = BacktestResult(
            equity_curve=equity_curve,
            spy_curve=spy_curve,
            trades=completed_trades,
            metrics=metrics,
            spy_metrics=spy_metrics,
            drawdown_curve=dd_curve,
            trade_log_df=trade_log_df,
            regime_overlay=regime_overlay,
            config=cfg,
            total_trades=len(completed_trades),
            winning_trades=winning,
            losing_trades=losing,
            circuit_breaker_hits=circuit_breaker_hits,
        )

        logger.info(
            f"Backtester complete: {len(completed_trades)} trades, "
            f"total return {metrics.total_return*100:.1f}%, "
            f"Sharpe {metrics.sharpe:.2f}, max DD {metrics.max_drawdown*100:.1f}%"
        )
        return result
