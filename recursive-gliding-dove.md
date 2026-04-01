# AutoStockAnalyzer - Implementation Plan

## Context
Building a stock forecasting and automated trading tool based on the user's CODE Design.pdf. The system scrapes stock/macro data, runs 12 long-term forecasting methods and short-term technical indicators, ranks stocks, and executes trades via Alpaca. The user wants both CLI scripts and a Streamlit web dashboard.

**Tech stack**: Python, yfinance, FRED API, Alpaca, Streamlit, statsmodels, PyTorch, pandas
**Ticker universe**: S&P 500 (~500 stocks), auto-fetched so it stays current
**Storage**: Parquet files (5-10x smaller than Excel, fast pandas reads, ideal for ML pipelines)
**Scraping strategy**: First run backfills full history (~15-20 min). Daily incremental runs fetch only previous day's data using batched `yf.download()` (~1-2 min for 500 tickers).
**GPU**: NVIDIA 3060 Ti (8 GB VRAM) — sufficient for LSTM/Transformer training on S&P 500 data
**Estimated storage**: ~10-20 GB total (200 GB allocated, plenty of headroom)

---

## Project Structure

```
AutoStockAnalyizer Software/
├── pyproject.toml
├── .env.example                 # API keys (Alpaca, FRED)
├── .gitignore
├── config/
│   └── settings.py              # Central config: keys, tickers, thresholds, paths
├── data/                        # Runtime data (gitignored)
│   ├── raw/daily/               # Daily OHLCV per ticker
│   ├── raw/quarterly/           # Quarterly aggregated data
│   ├── raw/macro/               # FRED series
│   ├── processed/               # Cleaned/merged datasets
│   └── forecasts/               # Output files
├── src/
│   ├── scraper/
│   │   ├── price_scraper.py     # yfinance daily/weekly/quarterly
│   │   ├── macro_scraper.py     # FRED API (GDP, rates, CPI)
│   │   ├── dividend_scraper.py  # Dividend history, ex-dates, yield (yfinance)
│   │   └── storage.py           # Parquet read/write with upsert logic
│   ├── news/
│   │   ├── news_fetcher.py      # Free news APIs (Finviz RSS, etc.)
│   │   ├── short_interest.py    # Short interest tracking
│   │   └── sentiment.py         # Basic sentiment scoring
│   ├── political/                   # Congressional & Insider Trading
│   │   ├── congress_tracker.py  # Senate/House trades (Quiver Quant scraping)
│   │   └── insider_tracker.py   # SEC Form 4 filings (insider buys/sells)
│   ├── options/
│   │   ├── options_data.py      # Put/Call ratio, unusual activity
│   │   └── implied_vol.py       # IV vs historical volatility comparison
│   ├── calendar/
│   │   ├── earnings.py          # Earnings dates, historical beat/miss data
│   │   └── economic.py          # FOMC, CPI, jobs reports, market-moving events
│   ├── forecasting/
│   │   ├── base.py              # Abstract ForecastMethod + ForecastResult
│   │   ├── decomposition.py     # Methods 1-5 (Additive, Multiplicative, Min/Max/Avg)
│   │   ├── moving_average.py    # Methods 6-7 (Weighted averages)
│   │   ├── exponential_smoothing.py  # Methods 8-10 (SES, Holt, Holt-Winters)
│   │   ├── regression.py        # Method 11 (OLS + quarterly dummies)
│   │   ├── auto_best.py         # Method 12 (AutoBest optimizer)
│   │   ├── runner.py            # Run all methods, collect results
│   │   └── metrics.py           # RMSE, MAE, MAPE
│   ├── indicators/
│   │   ├── base.py              # Abstract Indicator + Signal enum + IndicatorResult
│   │   ├── rsi.py               # RSI (overbought >70, oversold <30)
│   │   ├── moving_averages.py   # SMA/EMA (50-day, 200-day)
│   │   ├── macd.py              # MACD
│   │   ├── bollinger.py         # Bollinger Bands
│   │   ├── stochastic.py        # Stochastic Oscillator
│   │   ├── volume.py            # Volume analysis
│   │   ├── momentum.py          # Slow vs Fast momentum
│   │   ├── fundamentals.py      # P/S Ratio, Risk-Reward
│   │   ├── correlation.py       # Correlation forecasting (volume, volatility)
│   │   └── signal_aggregator.py # Combine signals into composite score
│   ├── trading/
│   │   ├── alpaca_client.py     # Alpaca wrapper (paper + live)
│   │   ├── portfolio.py         # Position tracking, P&L, analytics
│   │   ├── strategy.py          # Entry/exit logic, trailing stop rules
│   │   ├── multi_strategy.py    # Run value/momentum/mean-reversion, allocate capital
│   │   ├── risk.py              # Position sizing, risk limits
│   │   ├── circuit_breaker.py   # Kill switch: halt on 5-10% daily drop, force-sell at 15%
│   │   ├── backtester.py        # Historical strategy simulation + slippage modeling
│   │   ├── tax_lots.py          # Cost basis tracking, FIFO/LIFO, tax-loss harvesting
│   │   └── trade_journal.py     # Audit log: entry/exit reasons, P&L, holding period
│   ├── analytics/
│   │   ├── portfolio_metrics.py # Sharpe, Sortino, max drawdown, beta
│   │   ├── sector_analysis.py   # Sector rotation, concentration checks
│   │   ├── peer_comparison.py   # Stock vs sector peers relative strength
│   │   ├── correlation.py       # Holding correlation matrix
│   │   ├── market_regime.py     # Bull/Bear/Sideways/High-Vol detection (VIX, breadth)
│   │   └── attribution.py       # Performance attribution per stock
│   ├── watchlist/
│   │   └── watchlist.py         # Manual watchlist + "almost top 20" tracking
│   ├── alerts/
│   │   └── notifier.py          # Email, Discord, Telegram notifications
│   ├── reports/
│   │   └── morning_report.py    # Auto-generated daily PDF/Excel report
│   ├── ml/                          # ML Forecasting Engine
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract MLModel interface
│   │   ├── feature_engineer.py  # Build feature matrix (X) from all data sources
│   │   ├── xgboost_model.py     # XGBoost regressor
│   │   ├── random_forest.py     # Random Forest regressor
│   │   ├── linear_models.py     # Ridge, Lasso, ElasticNet
│   │   ├── lstm_model.py        # LSTM/GRU (PyTorch)
│   │   ├── transformer_model.py # Temporal Fusion Transformer (PyTorch)
│   │   ├── walk_forward.py      # Walk-forward cross-validation
│   │   ├── tuner.py             # Optuna hyperparameter optimization
│   │   ├── feature_selection.py # SHAP-based + correlation feature selection
│   │   ├── runner.py            # Run all ML models, compare results
│   │   └── models/              # Saved model artifacts (.pt, .pkl)
│   ├── ranking/
│   │   └── ranker.py            # Combine all forecasts + indicators -> top 20
│   └── utils/
│       ├── logging.py           # Structured logging
│       ├── dates.py             # Market calendar helpers
│       └── validation.py        # Data quality checks
├── cli/
│   ├── scrape.py                # python cli/scrape.py --tickers AAPL,MSFT
│   ├── forecast.py              # python cli/forecast.py --ticker AAPL
│   ├── indicators.py            # python cli/indicators.py --ticker AAPL
│   ├── ml.py                    # python cli/ml.py --ticker AAPL --train/--predict
│   ├── rank.py                  # python cli/rank.py --top 20
│   ├── trade.py                 # python cli/trade.py --mode paper
│   ├── backtest.py              # python cli/backtest.py --strategy default --years 5
│   ├── news.py                  # python cli/news.py --premarket
│   ├── watchlist.py             # python cli/watchlist.py --add TSLA / --show
│   └── report.py               # python cli/report.py --morning
├── dashboard/
│   ├── app.py                   # Streamlit main entry
│   ├── pages/
│   │   ├── 1_data_overview.py
│   │   ├── 2_long_term_forecast.py
│   │   ├── 3_short_term_signals.py
│   │   ├── 4_ml_predictions.py
│   │   ├── 5_stock_rankings.py
│   │   ├── 6_news_sentiment.py
│   │   ├── 7_political_insider.py   # Congress trades + insider filings
│   │   ├── 8_options_flow.py        # Put/Call, unusual activity, IV
│   │   ├── 9_sector_analysis.py     # Sector rotation, concentration
│   │   ├── 10_portfolio.py          # Analytics, correlation matrix, attribution
│   │   ├── 11_backtesting.py        # Run/view backtests
│   │   ├── 12_calendar.py           # Earnings + economic calendar
│   │   ├── 13_market_regime.py      # Bull/Bear/Sideways + VIX dashboard
│   │   ├── 14_watchlist.py          # Manual watchlist
│   │   ├── 15_trade_journal.py      # Trade history + audit log + tax lot view
│   │   ├── 16_peer_comparison.py    # Stock vs sector relative strength
│   │   └── 17_trading.py
│   └── components/
│       ├── charts.py            # Plotly chart builders
│       └── tables.py            # Styled dataframe renderers
└── tests/
```

---

## Key Packages

| Category | Package | Purpose |
|----------|---------|---------|
| Data | yfinance, fredapi, pyarrow, pandas, numpy | Stock/macro data + Parquet storage |
| Forecasting | statsmodels, scikit-learn | Decomposition, smoothing, OLS, metrics |
| ML | xgboost, lightgbm, torch, optuna, shap | Gradient boosting, deep learning, tuning, explainability |
| Indicators | ta | Technical analysis (RSI, MACD, Bollinger, etc.) |
| Trading | alpaca-trade-api | Broker integration |
| News | requests, feedparser, beautifulsoup4 | News APIs + scraping |
| Political | requests, beautifulsoup4 | Quiver Quant scraping, SEC EDGAR parsing |
| Options | yfinance (options chains) | Put/Call ratio, IV, unusual activity |
| Alerts | smtplib, discord-webhook, python-telegram-bot | Multi-channel notifications |
| Reports | reportlab or fpdf2 | Morning PDF report generation |
| UI | streamlit, plotly | Dashboard + charts |
| Config | python-dotenv, pydantic-settings | Environment + config management |
| Testing | pytest | Tests |

---

## Architecture: Key Patterns

### Forecasting - Strategy Pattern
All 12 methods implement `ForecastMethod` ABC with `fit(series)`, `predict(horizons)`, and `evaluate(series, holdout=8)`. `ForecastResult` dataclass holds method_name, forecasts, fitted_values, RMSE, MAE. `runner.py` orchestrates running all 12 and returning a comparison DataFrame.

### Indicators - Strategy Pattern
All indicators implement `Indicator` ABC with `compute(df) -> IndicatorResult`. `Signal` enum: STRONG_BUY(2), BUY(1), NEUTRAL(0), SELL(-1), STRONG_SELL(-2). `signal_aggregator.py` produces a weighted composite score.

### AutoBest (Method 12)
Runs methods 1-11 with 8-period holdout both forward AND backward on the series. Averages RMSE from both directions. Selects the method with lowest combined RMSE, then refits on full data.

### ML Forecasting - Full Pipeline
All ML models implement `MLModel` ABC with `train(X, y)`, `predict(X)`, `evaluate(X, y)`. Feature engineering builds X from all data sources (OHLCV, indicators, macro data, news sentiment). Walk-forward CV ensures no data leakage. Optuna tunes hyperparameters per model. SHAP provides feature importance explanations.

**Models**: XGBoost, LightGBM, Random Forest, Ridge/Lasso/ElasticNet (classical) + LSTM, GRU, Temporal Fusion Transformer (deep learning, PyTorch, GPU-accelerated on 3060 Ti)

**Feature matrix (X)**: price history, volume, all technical indicators, FRED macro series, news sentiment scores, short interest, congressional trading signals, insider buy/sell, options flow (Put/Call, IV), dividend yield, market regime, peer relative strength, seasonality features (day-of-week, month, quarter)
**Target (Y)**: stock price (configurable: next-day close, next-week close, or next-quarter avg)

---

## Implementation Phases

### Phase 1: Foundation
1. Initialize project: `pyproject.toml`, `.gitignore`, `.env.example`, directory structure
2. `config/settings.py` - centralized config with API keys, ticker lists, thresholds
3. `src/utils/` - logging, date helpers, validation
4. `src/scraper/price_scraper.py` - yfinance wrapper for daily OHLCV. **Initial backfill from 2015 to present** (11 years) for ML train/test split
5. `src/scraper/macro_scraper.py` - FRED API wrapper (also from 2015+)
6. `src/scraper/dividend_scraper.py` - dividend history, ex-dates, yield via yfinance (from 2015+)
7. `src/scraper/storage.py` - Parquet read/write with upsert (load, append, deduplicate, save)
8. `cli/scrape.py` - CLI entry point (`--backfill 2015` for first run, incremental after)

### Phase 2: Long-Term Forecasting Engine
1. `src/forecasting/base.py` - abstract interface + ForecastResult
2. `src/forecasting/metrics.py` - RMSE, MAE, MAPE + holdout evaluation
3. `decomposition.py` - Methods 1-5 (statsmodels seasonal_decompose)
4. `moving_average.py` - Methods 6-7 (weighted averages with pandas/numpy)
5. `exponential_smoothing.py` - Methods 8-10 (statsmodels SimpleExpSmoothing, Holt, ExponentialSmoothing)
6. `regression.py` - Method 11 (statsmodels OLS + quarterly dummies)
7. `auto_best.py` - Method 12 (optimizer across methods 1-11)
8. `runner.py` - orchestrator + `cli/forecast.py`

### Phase 3: Short-Term Technical Indicators
1. `src/indicators/base.py` - abstract interface + Signal enum
2. Individual indicator modules (RSI, MACD, Bollinger, Stochastic, EMA/SMA, Volume, Momentum, Fundamentals) using `ta` library for computation, custom signal logic on top
3. `correlation.py` - rolling correlation between price and volume/volatility
4. `signal_aggregator.py` + `cli/indicators.py`

### Phase 4: ML Forecasting Engine
1. `src/ml/base.py` - abstract MLModel interface with train/predict/evaluate
2. `src/ml/feature_engineer.py` - build feature matrix from all data sources (OHLCV, indicators, macro, sentiment)
3. `src/ml/walk_forward.py` - walk-forward cross-validation (train on past, test on next period, slide forward). **Default split: train 2015-2020, test 2020-2026** — configurable via settings
4. `src/ml/xgboost_model.py` + `random_forest.py` + `linear_models.py` - classical ML models
5. `src/ml/lstm_model.py` + `transformer_model.py` - PyTorch deep learning models (GPU)
6. `src/ml/tuner.py` - Optuna hyperparameter optimization for all models
7. `src/ml/feature_selection.py` - SHAP values + correlation filtering to find best features
8. `src/ml/runner.py` - run all ML models, compare RMSE/MAE, output best predictions
9. `cli/ml.py` - CLI entry point for training and prediction

### Phase 5: News & Short Interest
1. `news_fetcher.py` - Finviz RSS or free news API
2. `short_interest.py` - short interest tracking via yfinance/Finviz
3. `sentiment.py` - keyword-based sentiment scoring
4. `cli/news.py`

### Phase 6: Political & Insider Trading
1. `src/political/congress_tracker.py` - scrape Quiver Quant for Senate/House trades (who bought what, when, size)
2. `src/political/insider_tracker.py` - parse SEC EDGAR Form 4 filings (CEO/CFO buys/sells)
3. Integrate as signals into the ranker (congressional buy = bullish signal)

### Phase 7: Options & Calendar Data
1. `src/options/options_data.py` - pull options chains via yfinance, compute Put/Call ratio, detect unusual volume
2. `src/options/implied_vol.py` - compare implied volatility vs 30-day historical vol
3. `src/calendar/earnings.py` - earnings dates + historical beat/miss from yfinance
4. `src/calendar/economic.py` - FOMC dates, CPI releases, jobs reports (scrape from FRED or static calendar)

### Phase 8: Ranking + Streamlit Dashboard
1. `src/ranking/ranker.py` - composite ranking -> top 20 with all signal sources (forecasts, indicators, ML, political, options, sentiment, peer comparison)
2. Build all 17 Streamlit pages with Plotly charts (added: market regime, trade journal, tax lots)
3. Shared chart/table components

### Phase 9: Backtesting & Portfolio Analytics
1. `src/trading/backtester.py` - simulate strategy on historical data with **slippage & spread modeling** for realistic results
2. **Paper trading simulation mode**: train on 2015-2020 data, then simulate live trading through 2020-2026 day-by-day
   - System replays each trading day from Jan 2020 to present as if running live
   - All signals (forecasts, indicators, ML, news) computed using only data available at that point in time (no future data leakage)
   - Tracks every simulated trade: entries, exits, trailing stops, circuit breaker triggers
   - Produces full performance report: cumulative returns, drawdowns, win rate, Sharpe, comparison vs buy-and-hold S&P 500
   - Tests through COVID crash (Mar 2020), recovery, 2022 bear market, 2023-2025 bull run — validates system across all regime types
   - Dashboard page visualizes the simulation results with equity curve, trade markers, and regime overlays
3. `src/analytics/portfolio_metrics.py` - Sharpe ratio, Sortino, max drawdown, beta vs S&P 500
4. `src/analytics/sector_analysis.py` - sector breakdown, rotation detection, concentration warnings
5. `src/analytics/peer_comparison.py` - stock vs sector peers relative strength (AAPL vs Tech sector)
6. `src/analytics/market_regime.py` - Bull/Bear/Sideways/High-Vol detection using VIX, breadth (% above 200-day MA), yield curve
7. `src/analytics/correlation.py` - holding correlation matrix visualization
8. `src/analytics/attribution.py` - per-stock P&L attribution
9. `src/watchlist/watchlist.py` - manual watchlist with multi-timeframe analysis (daily + weekly)
10. `cli/backtest.py --mode full-sim --train 2015-2020 --test 2020-2026`

### Phase 10: Trading Integration
1. `alpaca_client.py` - paper-first wrapper (requires explicit env var for live)
2. `strategy.py` - entry/exit rules including 5% trailing stop after 3 profitable days
3. `multi_strategy.py` - run value/momentum/mean-reversion simultaneously, auto-allocate capital based on performance
4. `circuit_breaker.py` - **kill switch**: halt all trading if portfolio drops 5-10% in a day, force-sell at 15% single-stock loss, manual emergency stop button
5. `portfolio.py` - position tracking synced with Alpaca
6. `tax_lots.py` - cost basis tracking per lot (FIFO/LIFO/specific), tax-loss harvesting suggestions, long-term vs short-term gains tracking
7. `trade_journal.py` - audit log: every trade logged with entry/exit reasons, which signals triggered, P&L, holding period
8. `risk.py` - position sizing limits (e.g., max 5% per stock), regime-aware (reduce size in bear markets)
9. `cli/trade.py --mode paper|live`

### Phase 11: Alerts & Morning Report
1. `src/alerts/notifier.py` - multi-channel alerts (email, Discord webhook, Telegram bot)
   - Triggers: buy/sell signals, trailing stop hit, news sentiment flip, earnings approaching, congressional trade detected
2. `src/reports/morning_report.py` - auto-generate daily PDF/Excel: top 20 picks, portfolio status, key signals, upcoming earnings/events
3. `cli/report.py` - generate on demand or schedule at 7 AM

---

## Data Flow

```
[yfinance] ──┐                                     ┌─> [Long-Term Forecast] ──┐
              │                                     ├─> [Short-Term Indicators]│
[FRED API] ──┼─> [Scraper] ─> [Parquet Storage] ──┤                          ├─> [Ranker] ─> Top 20
              │                                     ├─> [ML Engine (GPU)] ─────┘       │
[Quiver] ────┤                                     │                                   v
[SEC EDGAR] ─┤   [Political/Insider Tracker] ──────┤                          [Backtester]
[Options] ───┤   [Options/Calendar] ───────────────┤                                   │
              │                                     │                          [Trade Executor]
[News APIs] ─┘   [News Checker] ──────────────────┘                                   │
                                                                               [Alpaca API]
                  [Alerts] <── triggers from all modules                               │
                  [Morning Report] <── daily PDF at 7AM          [Streamlit Dashboard] <┘
```

**Daily cycle**:
- 6:00 AM: Incremental data scrape (stock prices, dividends, macro, options, political filings, insider trades)
- 6:15 AM: Detect market regime (Bull/Bear/Sideways) — adjusts strategy aggressiveness
- 6:30 AM: Run forecasts + indicators + ML predictions + peer comparison
- 7:00 AM: Generate morning report (PDF/Excel), send alerts for key signals
- 8:00 AM: User reviews top 20 picks + earnings calendar + congressional trades on dashboard
- 9:30 AM: Market open — multi-strategy executor runs (value/momentum/mean-reversion), circuit breaker armed
- During hours: Monitor indicators, trailing stops on 3-day winners, alert on unusual options/news, circuit breaker watching for portfolio drawdown
- 4:00 PM: EOD snapshot, portfolio analytics, correlation matrix, trade journal entries, tax lot updates
- Weekly: Tax-loss harvesting scan, strategy performance review, capital reallocation between strategies

---

## Verification Plan
1. **Phase 1**: Run `cli/scrape.py` — Parquet files in `data/raw/` with OHLCV + dividends + macro data
2. **Phase 2**: Run `cli/forecast.py --ticker AAPL` — all 12 methods produce forecasts, AutoBest selects lowest-RMSE
3. **Phase 3**: Run `cli/indicators.py --ticker AAPL` — RSI, MACD, etc. produce valid signals
4. **Phase 4**: Run `cli/ml.py --ticker AAPL --train` then `--predict` — GPU training, walk-forward CV, SHAP feature importance
5. **Phase 5**: Run `cli/news.py --premarket` — news items with sentiment scores
6. **Phase 6**: Verify congressional trades and insider filings scraped and parsed
7. **Phase 7**: Verify options (Put/Call, IV) and earnings/economic calendar populate
8. **Phase 8**: Run `streamlit run dashboard/app.py` — all 17 pages render with data
9. **Phase 9**: Run `cli/backtest.py --mode full-sim --train 2015-2020 --test 2020-2026` — full paper trading simulation through COVID crash, bear market, and recovery. Verify: equity curve, trade log, Sharpe ratio, max drawdown, comparison vs S&P 500 buy-and-hold. Also verify market regime detection, peer comparison, and correlation matrix
10. **Phase 10**: Run `cli/trade.py --mode paper` — paper orders via Alpaca; circuit breaker halts on simulated 10% drop; multi-strategy allocates capital; trade journal logs entries; tax lots tracked
11. **Phase 11**: Trigger test alert (Discord/email), generate morning report PDF
