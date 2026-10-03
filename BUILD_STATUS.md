# BUILD STATUS

Status map of what is built and in what state. Not a duplicate of ARCHITECTURE.md or DOCUMENTATION.md. Statuses come only from the table in CLAUDE.md. The main session updates this file from the STATUS lines agents report.

Every module starts as Unverified. The architect checks the real files before planning around any Unverified module. A module is one row: either a single file or a package folder.

## Data layer

| Module | Files | Status | Note |
|---|---|---|---|
| config | config/settings.py | Unverified | |
| scraper | src/scraper/ (dividend_scraper, live_quotes, macro_scraper, price_scraper, storage) | Unverified | |
| calendar | src/calendar/ (earnings, economic) | Unverified | |
| options_data | src/options/ (implied_vol, options_data) | Unverified | |
| utils | src/utils/ (dates, input_sanitize, logging, tickers, validation) | Unverified | |

## Analysis

| Module | Files | Status | Note |
|---|---|---|---|
| indicators | src/indicators/ (base, bollinger, _calc, correlation, fundamentals, macd, momentum, moving_averages, rsi, signal_aggregator, stochastic, volume) | Unverified | |
| analytics | src/analytics/ (attribution, correlation, market_regime, peer_comparison, portfolio_metrics, sector_analysis) | Unverified | |
| ranking | src/ranking/ranker.py | Unverified | |
| watchlist | src/watchlist/watchlist.py | Unverified | |
| alerts | src/alerts/ (notifier, triggers) | Unverified | |
| reports | src/reports/morning_report.py | Unverified | |

## Forecasting and ML

| Module | Files | Status | Note |
|---|---|---|---|
| forecasting | src/forecasting/ (auto_best, base, decomposition, exponential_smoothing, metrics, moving_average, regression, runner) | Unverified | |
| ml | src/ml/ (base, catboost_model, ensemble_model, feature_engineer, feature_selection, linear_models, lstm_model, random_forest, runner, transformer_model, tuner, walk_forward, xgboost_model) | Unverified | |

## News and political

| Module | Files | Status | Note |
|---|---|---|---|
| news | src/news/ (aggregator, runner, scraper, sentiment, short_interest) | Unverified | |
| political | src/political/ (congress_tracker, insider_tracker) | Unverified | |

## Trading

| Module | Files | Status | Note |
|---|---|---|---|
| alpaca_client | src/trading/alpaca_client.py | Unverified | |
| backtester | src/trading/backtester.py | Unverified | |
| circuit_breaker | src/trading/circuit_breaker.py | Unverified | |
| multi_strategy | src/trading/multi_strategy.py | Unverified | |
| options_strategy | src/trading/options_strategy.py | Unverified | |
| portfolio | src/trading/portfolio.py | Unverified | |
| risk | src/trading/risk.py | Unverified | |
| stop_manager | src/trading/stop_manager.py | Unverified | |
| strategy | src/trading/strategy.py | Unverified | |
| tax_lots | src/trading/tax_lots.py | Unverified | |
| trade_journal | src/trading/trade_journal.py | Unverified | |

## Dashboard

| Module | Files | Status | Note |
|---|---|---|---|
| dashboard_core | dashboard/app.py, dashboard/components/ (charts, market_clock, session_cache, tables, ticker_selector) | Unverified | |
| pages_1_to_9 | dashboard/pages/ 1_data_overview, 2_morning_report, 2a_stock_summary, 3_long_term_forecast, 4_short_term_signals, 5_ml_predictions, 6_stock_rankings, 7_news_sentiment, 8_political_insider, 9_options_flow | Unverified | |
| pages_10_to_16 | dashboard/pages/ 10_portfolio, 11_sector_analysis, 12_market_regime, 13_backtesting, 14_calendar, 15_watchlist, 16_peer_comparison | Unverified | |
| pages_17_to_20 | dashboard/pages/ 17_trading, 18_trade_journal, 19_alerts, 20_auto_trades | Unverified | |

## Change Log

| Date | Agent | Note |
|---|---|---|
| 2026-10-02 | setup | Created BUILD_STATUS.md with every module set to Unverified |
