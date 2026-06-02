"""Full trading loop CLI.

Usage
-----
    python cli/trade.py --mode paper
    python cli/trade.py --mode live   # requires ALPACA_LIVE_TRADING=true in .env

Loop
----
1. Connect to Alpaca and fetch account / positions.
2. Arm circuit breaker with opening portfolio value.
3. Check circuit breaker — halt immediately if already tripped.
4. Force-sell any position breaching the single-stock loss threshold.
5. Detect current market regime (Bull/Bear/Sideways/High-Vol).
6. Run the ranker to get top-N picks.
7. Generate entry + exit signals via MultiStrategyManager.
8. Apply risk sizing to each BUY signal.
9. Check circuit breaker pre-order.
10. Place orders via AlpacaClient.
11. Log every trade to TradeJournal and open a TaxLot for each BUY.
12. Save EOD portfolio snapshot.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.utils.logging import setup_logger

logger = setup_logger("cli.trade")


# ---------------------------------------------------------------------------#
# Helpers
# ---------------------------------------------------------------------------#

def _abort(msg: str) -> None:
    logger.critical(msg)
    print(f"\n[ABORT] {msg}\n", flush=True)
    sys.exit(1)


def _banner(mode: str, account) -> None:
    mode_str = "LIVE TRADING" if mode == "live" else "PAPER TRADING"
    print("=" * 60)
    print(f"  AutoStockAnalyzer — {mode_str}")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    if account:
        print(f"  Portfolio: ${account.portfolio_value:,.2f}  "
              f"Cash: ${account.cash:,.2f}  "
              f"Daily P&L: ${account.daily_pnl:+,.2f} ({account.daily_pnl_pct:+.2f}%)")
    print("=" * 60)


# ---------------------------------------------------------------------------#
# States file (persist trailing-stop states between runs)
# ---------------------------------------------------------------------------#

_STATES_FILE = Path(__file__).parent.parent / "data" / "strategy_states.json"


def _load_states() -> dict:
    if _STATES_FILE.exists():
        try:
            return json.loads(_STATES_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_states(states: dict) -> None:
    _STATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        _STATES_FILE.write_text(json.dumps(states, indent=2))
    except Exception as exc:
        logger.warning(f"Could not save strategy states: {exc}")


# ---------------------------------------------------------------------------#
# Main loop
# ---------------------------------------------------------------------------#

def run(mode: str, dry_run: bool = False) -> None:
    settings = get_settings()

    if mode == "live":
        import os
        if os.getenv("ALPACA_LIVE_TRADING", "").strip().lower() != "true":
            _abort(
                "Live mode requested but ALPACA_LIVE_TRADING is not set to 'true' in .env. "
                "Aborting for safety."
            )

    # ------------------------------------------------------------------ #
    # 1. Connect
    # ------------------------------------------------------------------ #
    from src.trading.alpaca_client import AlpacaClient
    client = AlpacaClient(settings)

    if not client.connected:
        _abort("Could not connect to Alpaca. Check API keys in .env.")

    account = client.get_account()
    if account is None:
        _abort("Could not retrieve Alpaca account info.")

    if account.trading_blocked or account.account_blocked:
        _abort("Alpaca account is blocked. Resolve before trading.")

    _banner(mode, account)

    portfolio_value = account.portfolio_value
    positions       = client.list_positions()
    held_tickers    = {p.ticker for p in positions}

    # ------------------------------------------------------------------ #
    # 2. Arm circuit breaker
    # ------------------------------------------------------------------ #
    from src.trading.circuit_breaker import CircuitBreaker, CircuitBreakerTripped
    cb = CircuitBreaker(settings)
    if cb._state.portfolio_open_value == 0.0:
        cb.arm(portfolio_value)

    # ------------------------------------------------------------------ #
    # 3. Circuit breaker portfolio check
    # ------------------------------------------------------------------ #
    try:
        cb.check_portfolio(portfolio_value)
    except CircuitBreakerTripped as exc:
        print(f"\n[HALT] {exc}\n")
        logger.critical(f"Circuit breaker halted trading: {exc}")
        return

    # ------------------------------------------------------------------ #
    # 4. Force-sell single-stock losers
    # ------------------------------------------------------------------ #
    force_sold = cb.check_positions(client)
    if force_sold:
        print(f"[CIRCUIT BREAKER] Force-sold {len(force_sold)} positions: {force_sold}")
        # Refresh positions after force-sells
        positions    = client.list_positions()
        held_tickers = {p.ticker for p in positions}

    # ------------------------------------------------------------------ #
    # 5. Market regime
    # ------------------------------------------------------------------ #
    from src.analytics.market_regime import MarketRegimeAnalyzer
    regime_analyzer = MarketRegimeAnalyzer(settings)
    regime_snap = regime_analyzer.current_snapshot(compute_breadth=False)
    regime = regime_snap.regime.value
    print(f"\n[REGIME] {regime}  (score={regime_snap.score:.2f})")
    if regime_snap.vix:
        print(f"         VIX={regime_snap.vix:.1f}  "
              f"yield_spread={regime_snap.yield_spread or 'N/A'}")

    # ------------------------------------------------------------------ #
    # 6. Rank tickers
    # ------------------------------------------------------------------ #
    print("\n[RANKING] Running ranker …")
    from src.ranking.ranker import top_picks
    from src.utils.tickers import get_tickers
    tickers = get_tickers(settings.ticker_source)
    picks = top_picks(n=settings.top_n_picks, settings=settings,
                      include_ml=False, include_indicators=True,
                      tickers=tickers)

    if not picks:
        print("[RANKING] No picks returned — skipping order placement.")
    else:
        top_tickers = [e.ticker for e in picks]
        print(f"[RANKING] Top {len(picks)} picks: {top_tickers[:10]} …")

    # ------------------------------------------------------------------ #
    # 7. Generate signals
    # ------------------------------------------------------------------ #
    from src.trading.multi_strategy import MultiStrategyManager
    mgr = MultiStrategyManager(settings)
    mgr._momentum.seed_states(_load_states())

    all_signals = mgr.generate_all_signals(picks, positions, portfolio_value, tickers=tickers)

    # ------------------------------------------------------------------ #
    # 8 + 9. Risk sizing + circuit breaker pre-order
    # ------------------------------------------------------------------ #
    from src.trading.risk import RiskManager
    from src.trading.trade_journal import TradeJournal
    from src.trading.tax_lots import TaxLotTracker

    risk_mgr = RiskManager(settings)
    journal  = TradeJournal(settings)
    tax_lots = TaxLotTracker(settings)

    orders_placed = 0
    orders_rejected = 0

    # ------------------------------------------------------------------ #
    # 8b. Automated options trading
    # ------------------------------------------------------------------ #
    if settings.options_capital > 0 and picks:
        from src.trading.options_strategy import OptionsStrategy
        from src.options.implied_vol import compute_iv_metrics
        from src.options.options_data import compute_options_metrics

        opt_strategy = OptionsStrategy(settings)

        # Compute IV + options-flow metrics for top picks (limit for speed)
        opt_tickers = [e.ticker for e in picks[:10]]
        iv_metrics: dict = {}
        opts_metrics: dict = {}
        for _t in opt_tickers:
            try:
                iv_metrics[_t] = compute_iv_metrics(_t)
                opts_metrics[_t] = compute_options_metrics(_t)
            except Exception as _exc:
                logger.debug(f"[OPTIONS] metrics failed for {_t}: {_exc}")

        # --- exits first ---
        opt_live_positions = client.list_option_positions()
        exit_sigs = opt_strategy.check_exits(opt_live_positions, iv_metrics)

        for sig in exit_sigs:
            try:
                cb.check_portfolio(portfolio_value)
            except CircuitBreakerTripped as exc:
                print(f"\n[HALT] {exc}")
                break
            if dry_run:
                print(f"  [DRY-RUN] OPT-CLOSE {sig.symbol} — {sig.reason}")
                continue
            result = client.close_option_position(sig.symbol)
            if result:
                orders_placed += 1
                opt_strategy.record_close(sig.symbol)
                journal.log_exit(
                    ticker=sig.symbol,
                    price=sig.limit_price or 0.0,
                    shares=sig.qty,
                    entry_price=0.0,
                    entry_date=date.today().isoformat(),
                    signals={},
                    strategy="options",
                    exit_reason=sig.reason.split(":")[0],
                )
                print(f"  [OPT-CLOSE] {sig.symbol} — {sig.reason}")

        # --- entries ---
        entry_sigs = opt_strategy.generate_entries(picks, iv_metrics, opts_metrics, client)

        for sig in entry_sigs:
            try:
                cb.check_portfolio(portfolio_value)
            except CircuitBreakerTripped as exc:
                print(f"\n[HALT] {exc}")
                break
            if dry_run:
                print(
                    f"  [DRY-RUN] OPT-BUY {sig.qty}x {sig.symbol} "
                    f"({sig.contract_type.upper()}) @ ${sig.limit_price:.2f} — {sig.reason}"
                )
                continue
            result = client.place_option_order(
                symbol=sig.symbol,
                qty=sig.qty,
                side="buy",
                order_type="limit",
                limit_price=sig.limit_price,
            )
            if result:
                orders_placed += 1
                opt_strategy.record_fill(
                    symbol=sig.symbol,
                    underlying=sig.underlying,
                    contract_type=sig.contract_type,
                    strike=sig.strike,
                    expiration=sig.expiration,
                    qty=sig.qty,
                    fill_price=sig.limit_price or 0.0,
                    iv_spike_entry=iv_metrics.get(sig.underlying, {}).get("iv_spike", False),
                )
                journal.log_entry(
                    ticker=sig.symbol,
                    price=sig.limit_price or 0.0,
                    shares=sig.qty,
                    signals={},
                    strategy="options",
                )
                print(
                    f"  [OPT-BUY] {sig.qty}x {sig.symbol} "
                    f"({sig.contract_type.upper()}) @ ${sig.limit_price:.2f} — {sig.reason}"
                )

    current_pos_values: dict[str, float] = {p.ticker: p.market_value for p in positions}
    current_scores: dict[str, float]     = {e.ticker: e.composite_score for e in picks}

    for strategy_name, signals in all_signals.items():
        for sig in signals:
            ticker = sig.ticker

            # --- SELL ---
            if sig.action == "SELL":
                if dry_run:
                    print(f"  [DRY-RUN] SELL {ticker} — {sig.reason}")
                    continue

                # Find position for entry details
                pos = next((p for p in positions if p.ticker == ticker), None)
                result = client.close_position(ticker)
                if result:
                    orders_placed += 1
                    held_tickers.discard(ticker)
                    entry_price = pos.avg_entry_price if pos else 0.0
                    entry_date  = date.today().isoformat()   # fallback
                    journal.log_exit(
                        ticker=ticker,
                        price=pos.current_price if pos else 0.0,
                        shares=pos.qty if pos else 0.0,
                        entry_price=entry_price,
                        entry_date=entry_date,
                        signals=current_scores,
                        strategy=strategy_name,
                        exit_reason=sig.reason.split(":")[0],
                    )
                    tax_lots.close_lots(ticker, qty=pos.qty if pos else 0.0,
                                        price=pos.current_price if pos else 0.0)
                    mgr._momentum.clear_state(ticker)
                    current_pos_values.pop(ticker, None)
                    print(f"  [SELL] {ticker} — {sig.reason}")
                continue

            # --- BUY ---
            # Check circuit breaker before each buy
            try:
                cb.check_portfolio(portfolio_value)
            except CircuitBreakerTripped as exc:
                print(f"\n[HALT] {exc}")
                break

            if ticker in held_tickers:
                continue

            # Fetch current price
            pos = next((p for p in picks if p.ticker == ticker), None)
            price_est = None
            try:
                from src.scraper.storage import load_dataframe, get_ticker_filepath
                fp = get_ticker_filepath(ticker, settings.raw_daily_dir)
                df = load_dataframe(fp)
                if not df.empty and "Close" in df.columns:
                    price_est = float(df["Close"].dropna().iloc[-1])
            except Exception:
                pass

            if price_est is None or price_est <= 0:
                logger.warning(f"No price data for {ticker} — skipping.")
                orders_rejected += 1
                continue

            sizing = risk_mgr.position_size(
                ticker=ticker,
                portfolio_value=portfolio_value,
                current_price=price_est,
                regime=regime,
            )
            if sizing.shares <= 0:
                logger.info(f"[{ticker}] Sized to 0 shares — skipping.")
                orders_rejected += 1
                continue

            ok, reason = risk_mgr.check_order(
                ticker=ticker,
                qty=sizing.shares,
                current_price=price_est,
                portfolio_value=portfolio_value,
                current_position_value=current_pos_values.get(ticker, 0.0),
                regime=regime,
            )
            if not ok:
                logger.warning(f"[{ticker}] Risk check failed: {reason}")
                orders_rejected += 1
                continue

            if dry_run:
                print(f"  [DRY-RUN] BUY {sizing.shares} x {ticker} @ ~${price_est:.2f} "
                      f"(${sizing.dollar_amount:,.0f}) — {strategy_name}")
                continue

            order = client.place_order(ticker, sizing.shares, "buy")
            if order:
                orders_placed += 1
                held_tickers.add(ticker)
                current_pos_values[ticker] = sizing.dollar_amount
                journal.log_entry(
                    ticker=ticker,
                    price=price_est,
                    shares=sizing.shares,
                    signals=current_scores,
                    strategy=strategy_name,
                )
                tax_lots.open_lot(ticker, qty=sizing.shares, price=price_est)
                print(f"  [BUY] {sizing.shares} x {ticker} @ ~${price_est:.2f} "
                      f"(${sizing.dollar_amount:,.0f}) [{strategy_name}]")
            else:
                orders_rejected += 1

    # ------------------------------------------------------------------ #
    # 11. Save states + EOD snapshot
    # ------------------------------------------------------------------ #
    _save_states(mgr._momentum.export_states())

    from src.trading.portfolio import PortfolioTracker
    tracker = PortfolioTracker(settings)
    tracker.save_eod_snapshot()

    print(f"\n[DONE] Orders placed: {orders_placed}  Rejected: {orders_rejected}")
    stats = journal.summary_stats()
    if stats:
        print(
            f"[JOURNAL] Total trades: {stats['total_trades']}  "
            f"Win rate: {stats['win_rate']*100:.1f}%  "
            f"Total P&L: ${stats['total_pnl']:+,.2f}"
        )


# ---------------------------------------------------------------------------#
# CLI entry
# ---------------------------------------------------------------------------#

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoStockAnalyzer trading loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli/trade.py --mode paper
  python cli/trade.py --mode paper --dry-run
  python cli/trade.py --mode live     # ALPACA_LIVE_TRADING=true required
""",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode (default: paper)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate signals but do NOT submit any orders",
    )
    args = parser.parse_args()

    run(mode=args.mode, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
