"""Alert trigger checker — scans all signal sources and returns AlertEvents.

Trigger conditions
------------------
1. New BUY signal for a top-ranked ticker
   (composite_score in top-N from ranker, or indicator_score >= BUY threshold)

2. Trailing stop hit on any open position
   (position unrealized_plpc <= -settings.trailing_stop_pct)

3. News sentiment flip: positive → negative for a held ticker
   (compare today's sentiment vs yesterday's in data/news/articles/)

4. Earnings approaching within 7 days for any held ticker
   (uses src/calendar/earnings.py)

5. Congressional trade detected for any held ticker
   (uses src/political/congress_tracker.py, within last 7 days)

6. Circuit breaker tripped
   (reads data/circuit_breaker_state.json, checks `halted` flag)

Public API
----------
    checker = TriggerChecker(settings)
    events  = checker.check_all(held_tickers)   -> list[AlertEvent]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from config.settings import get_settings
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

AlertLevel = Literal["info", "warning", "critical"]


# ---------------------------------------------------------------------------#
# AlertEvent
# ---------------------------------------------------------------------------#

@dataclass
class AlertEvent:
    trigger_type: str           # e.g. "buy_signal", "trailing_stop", ...
    title: str
    body: str
    level: AlertLevel = "info"
    ticker: str | None = None   # None for portfolio-wide triggers (circuit breaker)
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------#
# TriggerChecker
# ---------------------------------------------------------------------------#

class TriggerChecker:
    """Check all signal sources and return a list of AlertEvent objects."""

    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        self._s = settings

    # ---------------------------------------------------------------------- #
    # Main entry point
    # ---------------------------------------------------------------------- #

    def check_all(self, held_tickers: list[str] | None = None) -> list[AlertEvent]:
        """Run all trigger checks and return combined AlertEvent list.

        Parameters
        ----------
        held_tickers:
            List of tickers currently held in the portfolio.
            If None, the checker will try to load positions from the portfolio
            tracker or fall back to an empty list.
        """
        if held_tickers is None:
            held_tickers = self._load_held_tickers()

        events: list[AlertEvent] = []

        checkers = [
            ("circuit_breaker",   self._check_circuit_breaker,  []),
            ("buy_signal",        self._check_buy_signals,       [held_tickers]),
            ("trailing_stop",     self._check_trailing_stops,    [held_tickers]),
            ("sentiment_flip",    self._check_sentiment_flip,    [held_tickers]),
            ("earnings_alert",    self._check_earnings,          [held_tickers]),
            ("congress_trade",    self._check_congress_trades,   [held_tickers]),
        ]

        for name, fn, args in checkers:
            try:
                result = fn(*args)
                events.extend(result)
            except Exception as exc:
                logger.error(f"[triggers] {name} check failed: {exc}")

        logger.info(f"[triggers] {len(events)} alert(s) generated from {len(checkers)} checks")
        return events

    # ---------------------------------------------------------------------- #
    # 1. Circuit breaker
    # ---------------------------------------------------------------------- #

    def _check_circuit_breaker(self) -> list[AlertEvent]:
        state_file: Path = self._s.circuit_breaker_state_file
        if not state_file.exists():
            return []

        try:
            state = json.loads(state_file.read_text())
        except Exception as exc:
            logger.warning(f"[circuit_breaker] Could not read state: {exc}")
            return []

        if not state.get("halted", False):
            return []

        reason = state.get("halt_reason", "Unknown reason")
        halt_time = state.get("halt_time", "")
        return [AlertEvent(
            trigger_type="circuit_breaker",
            ticker=None,
            title="Circuit Breaker Tripped",
            body=f"Trading has been halted.\nReason: {reason}\nTime: {halt_time}",
            level="critical",
        )]

    # ---------------------------------------------------------------------- #
    # 2. Buy signals
    # ---------------------------------------------------------------------- #

    def _check_buy_signals(self, held_tickers: list[str]) -> list[AlertEvent]:
        """Fire an alert when a non-held ticker enters the top-N ranked picks."""
        events: list[AlertEvent] = []

        try:
            from src.ranking.ranker import top_picks, to_dataframe
            picks = top_picks(self._s.top_n_picks, self._s)
        except Exception as exc:
            logger.warning(f"[buy_signal] Could not load top picks: {exc}")
            return []

        for entry in picks:
            ticker = getattr(entry, "ticker", None)
            if ticker is None:
                continue
            if ticker in held_tickers:
                continue   # already held — skip

            score = getattr(entry, "composite_score", 0.0)
            rank = getattr(entry, "rank", "?")

            # Only alert on strong buy signals (score > 0.5 on [-1,1] scale)
            if score < 0.5:
                continue

            events.append(AlertEvent(
                trigger_type="buy_signal",
                ticker=ticker,
                title=f"BUY Signal: {ticker}",
                body=(
                    f"Ticker: {ticker}\n"
                    f"Rank: #{rank} | Composite Score: {score:.3f}\n"
                    f"This ticker has entered the top {self._s.top_n_picks} ranked picks."
                ),
                level="info",
            ))

        return events

    # ---------------------------------------------------------------------- #
    # 3. Trailing stop
    # ---------------------------------------------------------------------- #

    def _check_trailing_stops(self, held_tickers: list[str]) -> list[AlertEvent]:
        """Alert when any held position is down more than trailing_stop_pct."""
        events: list[AlertEvent] = []
        threshold = -abs(self._s.trailing_stop_pct)

        try:
            from src.trading.portfolio import PortfolioTracker
            tracker = PortfolioTracker(self._s)
            snapshot = tracker.get_snapshot()
        except Exception as exc:
            logger.warning(f"[trailing_stop] Could not load portfolio: {exc}")
            return []

        if snapshot is None:
            return []

        for pos in snapshot.positions:
            if pos.unrealized_plpc <= threshold:
                events.append(AlertEvent(
                    trigger_type="trailing_stop",
                    ticker=pos.ticker,
                    title=f"Trailing Stop Hit: {pos.ticker}",
                    body=(
                        f"Ticker: {pos.ticker}\n"
                        f"Unrealized P&L: {pos.unrealized_plpc:+.1f}%  "
                        f"(${pos.unrealized_pl:+,.2f})\n"
                        f"Threshold: {threshold:.1f}%\n"
                        f"Current Price: ${pos.current_price:.2f} | "
                        f"Avg Entry: ${pos.avg_entry_price:.2f}"
                    ),
                    level="warning",
                ))

        return events

    # ---------------------------------------------------------------------- #
    # 4. Sentiment flip
    # ---------------------------------------------------------------------- #

    def _check_sentiment_flip(self, held_tickers: list[str]) -> list[AlertEvent]:
        """Alert when news sentiment flips from positive to negative for a held ticker."""
        events: list[AlertEvent] = []
        articles_dir: Path = self._s.news_articles_dir

        today = datetime.now(tz=timezone.utc).date()
        yesterday = today - timedelta(days=1)

        for ticker in held_tickers:
            parquet = articles_dir / f"{ticker}.parquet"
            if not parquet.exists():
                continue

            try:
                df = pd.read_parquet(parquet)
                if df.empty or "sentiment_score" not in df.columns:
                    continue

                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")

                today_ts = pd.Timestamp(today, tz="UTC")
                yest_ts = pd.Timestamp(yesterday, tz="UTC")

                today_rows = df[df.index.date == today]
                yest_rows = df[df.index.date == yesterday]

                if today_rows.empty or yest_rows.empty:
                    continue

                today_score = float(today_rows["sentiment_score"].mean())
                yest_score = float(yest_rows["sentiment_score"].mean())

                # Flip: yesterday positive (>0.1), today negative (<-0.1)
                if yest_score > 0.1 and today_score < -0.1:
                    events.append(AlertEvent(
                        trigger_type="sentiment_flip",
                        ticker=ticker,
                        title=f"Sentiment Flip: {ticker}",
                        body=(
                            f"Ticker: {ticker}\n"
                            f"Yesterday sentiment: {yest_score:+.3f} (positive)\n"
                            f"Today sentiment:     {today_score:+.3f} (negative)\n"
                            f"News may have turned bearish for this held position."
                        ),
                        level="warning",
                    ))

            except Exception as exc:
                logger.debug(f"[sentiment_flip] {ticker}: {exc}")

        return events

    # ---------------------------------------------------------------------- #
    # 5. Earnings approaching
    # ---------------------------------------------------------------------- #

    def _check_earnings(self, held_tickers: list[str]) -> list[AlertEvent]:
        """Alert when earnings are within 7 days for any held ticker."""
        events: list[AlertEvent] = []
        calendar_dir: Path = self._s.calendar_dir
        window_days = 7

        for ticker in held_tickers:
            try:
                from src.calendar.earnings import get_earnings_signal
                signal = get_earnings_signal(ticker, calendar_dir)
            except Exception as exc:
                logger.debug(f"[earnings] {ticker}: {exc}")
                continue

            days = signal.get("days_to_earnings")
            if days is None:
                continue

            if 0 <= days <= window_days:
                next_date = signal.get("next_earnings_date", "Unknown")
                beat_rate = signal.get("beat_rate")
                beat_str = f"{beat_rate:.0%}" if beat_rate is not None else "N/A"

                events.append(AlertEvent(
                    trigger_type="earnings_alert",
                    ticker=ticker,
                    title=f"Earnings in {days}d: {ticker}",
                    body=(
                        f"Ticker: {ticker}\n"
                        f"Earnings Date: {next_date}\n"
                        f"Days Away: {days}\n"
                        f"Historical Beat Rate: {beat_str}"
                    ),
                    level="warning" if days <= 3 else "info",
                ))

        return events

    # ---------------------------------------------------------------------- #
    # 6. Congressional trades
    # ---------------------------------------------------------------------- #

    def _check_congress_trades(self, held_tickers: list[str]) -> list[AlertEvent]:
        """Alert when a congressional trade is detected for a held ticker in last 7d."""
        events: list[AlertEvent] = []
        congress_dir: Path = self._s.political_congress_dir
        window_days = 7
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)

        for ticker in held_tickers:
            try:
                from src.political.congress_tracker import load_congress_trades
                df = load_congress_trades(ticker, congress_dir)
            except Exception as exc:
                logger.debug(f"[congress] {ticker}: {exc}")
                continue

            if df.empty:
                continue

            recent = df[df.index >= cutoff]
            if recent.empty:
                continue

            # Summarise recent activity
            buys = recent[recent["transaction"].str.lower().str.startswith("p") |
                          recent["transaction"].str.lower().str.startswith("b")]
            sells = recent[recent["transaction"].str.lower().str.startswith("s")]

            lines = [f"Ticker: {ticker}", f"Last {window_days} days:"]
            for _, row in recent.iterrows():
                lines.append(
                    f"  {row.get('representative', '?')} "
                    f"({row.get('party', '?')}/{row.get('chamber', '?')}) — "
                    f"{row.get('transaction', '?')}"
                )

            level: AlertLevel = "info" if len(buys) >= len(sells) else "warning"
            events.append(AlertEvent(
                trigger_type="congress_trade",
                ticker=ticker,
                title=f"Congressional Trade: {ticker}",
                body="\n".join(lines),
                level=level,
            ))

        return events

    # ---------------------------------------------------------------------- #
    # Helper: load held tickers from portfolio snapshot
    # ---------------------------------------------------------------------- #

    def _load_held_tickers(self) -> list[str]:
        """Try to load currently held tickers from the latest portfolio snapshot."""
        snapshots_dir: Path = self._s.portfolio_snapshots_dir
        if not snapshots_dir.exists():
            return []

        files = sorted(snapshots_dir.glob("*.parquet"))
        if not files:
            return []

        try:
            df = pd.read_parquet(files[-1])
            if "ticker" in df.columns:
                return df["ticker"].dropna().unique().tolist()
        except Exception as exc:
            logger.warning(f"[triggers] Could not load held tickers: {exc}")

        return []
