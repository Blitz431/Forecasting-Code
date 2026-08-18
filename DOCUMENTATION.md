# AutoStockAnalyzer — Complete Documentation

> **Disclaimer:** This tool is for educational and analytical purposes only. Nothing in this
> documentation or the software itself constitutes financial advice. Always consult a qualified
> financial professional before making investment decisions.

---

## Table of Contents

1. [What Is AutoStockAnalyzer?](#1-what-is-autostockanalyzer)
2. [Motivation — Why It Was Built](#2-motivation--why-it-was-built)
3. [How It Differs From Other Tools](#3-how-it-differs-from-other-tools)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [Data Collection (Phase 1)](#5-data-collection-phase-1)
6. [Long-Term Forecasting — 12 Methods (Phase 2)](#6-long-term-forecasting--12-methods-phase-2)
7. [Short-Term Technical Indicators (Phase 3)](#7-short-term-technical-indicators-phase-3)
8. [Machine Learning Engine (Phase 4)](#8-machine-learning-engine-phase-4)
9. [News & Sentiment Analysis (Phase 5)](#9-news--sentiment-analysis-phase-5)
10. [Political & Insider Trading Signals (Phase 6)](#10-political--insider-trading-signals-phase-6)
11. [Options Flow & Calendar (Phase 7)](#11-options-flow--calendar-phase-7)
12. [Stock Ranking System (Phase 8)](#12-stock-ranking-system-phase-8)
13. [Portfolio Analytics & Backtesting (Phase 9)](#13-portfolio-analytics--backtesting-phase-9)
14. [Trading Integration (Phase 10 + Smart Trade Management)](#14-trading-integration-phase-10--smart-trade-management)
15. [Alerts & Morning Report (Phase 11)](#15-alerts--morning-report-phase-11)
16. [Automation (Phase 12)](#16-automation-phase-12)
17. [The Dashboard — All 22 Pages](#17-the-dashboard--all-22-pages)
18. [How to Install & Run](#18-how-to-install--run)
19. [Daily Workflow — How to Use It](#19-daily-workflow--how-to-use-it)
20. [The Role of AI in This System](#20-the-role-of-ai-in-this-system)

---

## 1. What Is AutoStockAnalyzer?

AutoStockAnalyzer is a fully self-contained stock analysis and research platform built in Python.
It collects financial data from multiple public sources, runs that data through a suite of
analytical methods ranging from classical statistical forecasting to modern deep learning, and
presents the results in an interactive web dashboard.

The system covers the entire S&P 500 universe (~500 stocks) and is designed to run daily,
automatically updating all signals, forecasts, and rankings before the market opens.

**What it is:**
- An analytical research tool that makes multiple layers of market data visible in one place
- A platform for applying and comparing forecasting and ML techniques on real financial data
- A learning environment where every signal and prediction is explainable, not a black box

**What it is not:**
- A financial advisor
- A guaranteed profit system
- A replacement for professional investment guidance

---

## 2. Motivation — Why It Was Built

Most retail financial tools fall into two categories:

1. **Simple apps** (Robinhood, Yahoo Finance) — clean UI but shallow analysis. They show price
   charts and basic news. No forecasting, no ML, no multi-signal ranking.

2. **Professional terminals** (Bloomberg, FactSet) — extremely powerful but cost thousands of
   dollars per month and are designed for institutional analysts, not individual learners.

There is very little in between. Someone who wants to study how RSI interacts with a Holt-Winters
forecast and a FinBERT sentiment score on the same stock has no tool that does all three and
shows the results side by side.

AutoStockAnalyzer was built to fill that gap — a single platform that aggregates every data
source and analytical method that an informed retail analyst might want, without the cost barrier
of professional software.

A second motivation was academic: applying concepts from university coursework (statistical
forecasting, machine learning, time series analysis) to a real-world system with real data. Every
forecasting method and ML model in this codebase corresponds to a concept studied in class,
applied at production scale.

---

## 3. How It Differs From Other Tools

| Feature | Yahoo Finance | Robinhood | TradingView | AutoStockAnalyzer |
|---------|--------------|-----------|-------------|-------------------|
| Price charts | Yes | Yes | Yes | Yes |
| Technical indicators | Basic | No | Yes | Yes (9 indicators) |
| Long-term forecasting | No | No | No | Yes (12 methods) |
| ML predictions | No | No | No | Yes (7 models) |
| FinBERT sentiment | No | No | No | Yes |
| Congressional trades | No | No | No | Yes |
| Insider filings (Form 4) | No | No | No | Yes |
| Options flow analysis | Limited | Limited | Limited | Yes |
| Composite ranking | No | No | No | Yes (top 20) |
| Explainable AI (SHAP) | No | No | No | Yes |
| Fully open & customizable | No | No | No | Yes |
| Cost | Free (limited) | Free | Free/Paid | Free (self-hosted) |

**The key differentiator is explainability.** Every signal in this system shows its reasoning.
The ML models output SHAP feature importance so you can see exactly which data points drove
a prediction. The 12 forecasting methods all run simultaneously so you can compare their
outputs rather than trusting a single black-box number. This is a system built for understanding,
not just output.

---

## 4. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                         │
│ yfinance │ FRED │ Quiver │ SEC EDGAR │ RSS News │ Ollama+SearXNG │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    PARQUET STORAGE                          │
│         data/raw/  │  data/processed/  │  data/forecasts/  │
└──────┬──────────────┬──────────────────┬────────────────────┘
       │              │                  │
       ▼              ▼                  ▼
┌──────────┐  ┌──────────────┐  ┌──────────────────────────┐
│ 12 Long- │  │  9 Technical │  │    ML Engine (7 models)  │
│  Term    │  │  Indicators  │  │  XGBoost │ LightGBM │ RF  │
│ Forecast │  │  RSI │ MACD  │  │  Ridge   │ LSTM │ GRU    │
│ Methods  │  │  Bollinger   │  │  Transformer             │
└────┬─────┘  └──────┬───────┘  └────────────┬─────────────┘
     │               │                        │
     └───────────────┴────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │   COMPOSITE RANKER       │
              │   Top 20 Stock Picks     │
              └──────────────┬───────────┘
                             │
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
         [Dashboard]    [Alerts]    [Morning Report]
         Streamlit       Discord        PDF/Excel
```

The system is organized into 12 build phases, each adding a new analytical layer. All layers
feed into a single composite ranking that produces the top 20 stock picks each day.

**Storage format:** All data is stored as Apache Parquet files. Parquet is a columnar binary
format that is 5–10x smaller than CSV/Excel and reads into pandas 10x faster. It is the
standard format used by data engineering teams at scale.

**Project structure:**
```
config/         Central settings and API key management
src/
  scraper/      Data collection from all external sources
  forecasting/  12 long-term forecasting methods
  indicators/   9 short-term technical indicators
  ml/           7 ML models + feature engineering + tuning
  news/         News scraping + FinBERT sentiment scoring
  political/    Congressional trades + insider SEC filings
  options/      Put/Call ratio + implied volatility
  calendar/     Earnings dates + economic events
  analytics/    Portfolio metrics + sector analysis + regime + supply-chain graph
  trading/      Alpaca broker + strategy + risk + backtesting
  ranking/      Composite scorer → top 20
  alerts/       Discord + email notifications
  reports/      Morning PDF/Excel report
  watchlist/    Manual watchlist management
  research/     Deep-research agent (local Ollama + SearXNG) for supplier discovery
  utils/        Logging, date helpers, validation
cli/            Command-line entry points for each module
dashboard/      Streamlit web application (22 pages)
tests/          Automated test suite
```

---

## 5. Data Collection (Phase 1)

**File:** `src/scraper/`

The system collects four categories of data:

### Stock Price Data (`price_scraper.py`)
- Source: Yahoo Finance via `yfinance`
- Data: Daily OHLCV (Open, High, Low, Close, Volume) for all S&P 500 tickers
- First run: Full historical backfill from 2015 to present (~15–20 minutes)
- Daily runs: Incremental update, previous day only (~1–2 minutes for 500 tickers)
- Batching: 50 tickers per request to avoid Yahoo Finance rate limits

### Macro Economic Data (`macro_scraper.py`)
- Source: Federal Reserve Economic Data (FRED API)
- Series collected:
  - GDP — Gross Domestic Product
  - FEDFUNDS — Federal Funds interest rate
  - CPIAUCSL — Consumer Price Index (inflation)
  - UNRATE — Unemployment rate
  - DGS10 — 10-Year Treasury yield
  - DGS2 — 2-Year Treasury yield
  - VIXCLS — VIX volatility index
  - T10Y2Y — Yield curve spread (recession indicator)
  - UMCSENT — Consumer sentiment
  - INDPRO — Industrial production

### Dividend Data (`dividend_scraper.py`)
- Source: yfinance
- Data: Dividend history, ex-dividend dates, current yield per ticker
- Used as a signal in the ranking system (dividend yield, consistency)

### Storage (`storage.py`)
- All data written to Parquet with upsert logic: load existing file, append new rows,
  deduplicate by date, save back
- This ensures no data loss on repeated runs

**CLI usage:**
```bash
# First run — full backfill from 2015
python cli/scrape.py --backfill 2015

# Daily incremental update
python cli/scrape.py
```

---

## 6. Long-Term Forecasting — 12 Methods (Phase 2)

**File:** `src/forecasting/`

This is the core statistical forecasting engine. All 12 methods run on each stock's quarterly
price series and produce multi-quarter price forecasts. Results are compared by RMSE (Root Mean
Square Error) so you can see which method fits each stock best.

### The 12 Methods

**Seasonal Decomposition (Methods 1–5)**
These methods decompose a time series into trend, seasonal, and residual components using
`statsmodels.seasonal_decompose`.

1. **Additive Decomposition** — assumes seasonal swings are constant in absolute size
2. **Multiplicative Decomposition** — assumes seasonal swings grow proportionally with the trend
3. **Minimum Decomposition** — uses only the trough of each seasonal cycle for conservative forecasting
4. **Maximum Decomposition** — uses only the peak for optimistic forecasting
5. **Average Decomposition** — uses the midpoint between min and max

**Moving Averages (Methods 6–7)**
6. **Weighted Moving Average** — recent quarters weighted more heavily than older ones
7. **Linear Weighted Moving Average** — linearly increasing weights so the most recent period
   has the highest influence

**Exponential Smoothing (Methods 8–10)**
These adapt more quickly to recent changes than simple averages.

8. **Simple Exponential Smoothing (SES)** — smooths level only, good for flat series
9. **Holt's Linear Trend** — smooths level + trend, good for steadily rising/falling stocks
10. **Holt-Winters Triple Smoothing** — smooths level + trend + seasonality, the most
    comprehensive classical method

**Regression (Method 11)**
11. **OLS Regression with Quarterly Dummies** — fits a linear regression model with time as
    the predictor and Q1/Q2/Q3/Q4 dummy variables to capture seasonal patterns

**AutoBest (Method 12)**
12. **AutoBest** — runs methods 1–11 on the series with an 8-period holdout evaluation, both
    forward and backward. Averages RMSE from both directions, selects the lowest-error method,
    then refits on the full series. AutoBest is the production forecast — it automatically
    selects the best technique for each individual stock.

### Evaluation
All methods are evaluated using:
- **RMSE** — Root Mean Square Error (penalizes large errors heavily)
- **MAE** — Mean Absolute Error (average absolute miss)
- **MAPE** — Mean Absolute Percentage Error (scale-independent, useful for comparing stocks)

**CLI usage:**
```bash
python cli/forecast.py --ticker AAPL
python cli/forecast.py --tickers MSFT,NVDA,JNJ
```

---

## 7. Short-Term Technical Indicators (Phase 3)

**File:** `src/indicators/`

Nine technical indicators run on daily price data. Each produces a signal on a 5-point scale:
`STRONG_BUY (+2)`, `BUY (+1)`, `NEUTRAL (0)`, `SELL (-1)`, `STRONG_SELL (-2)`.

### Indicators

| Indicator | File | Logic |
|-----------|------|-------|
| RSI | `rsi.py` | Overbought >70 (SELL), Oversold <30 (BUY) |
| MACD | `macd.py` | Signal line crossover detection |
| Bollinger Bands | `bollinger.py` | Price touching upper/lower band = signal |
| Stochastic Oscillator | `stochastic.py` | Overbought >80, Oversold <20 |
| Moving Averages | `moving_averages.py` | Price vs 50-day & 200-day EMA/SMA, Golden/Death Cross |
| Volume Analysis | `volume.py` | Volume spike relative to rolling average |
| Momentum | `momentum.py` | Fast vs slow momentum divergence |
| Fundamentals | `fundamentals.py` | P/S ratio, risk-reward ratio |
| Correlation | `correlation.py` | Rolling correlation between price and volume/volatility |

### Signal Aggregator (`signal_aggregator.py`)
Combines all 9 indicator signals into a single weighted composite score. The composite score
feeds directly into the ranking system.

**CLI usage:**
```bash
python cli/indicators.py --ticker AAPL
```

---

## 8. Machine Learning Engine (Phase 4)

**File:** `src/ml/`

The ML engine trains 7 models on a feature matrix built from all available data sources and
produces price forecasts. This is the most computationally intensive part of the system.

### Models

**Classical ML (CPU, fast):**
- **XGBoost** — gradient boosted decision trees; consistently one of the best performers on
  tabular financial data
- **LightGBM** — Microsoft's gradient boosting implementation; faster than XGBoost on large
  datasets
- **Random Forest** — ensemble of decision trees; robust to outliers
- **Ridge Regression** — linear model with L2 regularization
- **Lasso Regression** — linear model with L1 regularization (automatic feature selection)
- **ElasticNet** — combination of Ridge and Lasso

**Deep Learning (GPU-accelerated, optional):**
- **LSTM** — Long Short-Term Memory network; captures long-range temporal dependencies in
  sequential price data
- **GRU** — Gated Recurrent Unit; similar to LSTM but faster to train
- **Temporal Fusion Transformer** — attention-based architecture; state-of-the-art for
  multi-horizon time series forecasting

### Feature Engineering (`feature_engineer.py`)
The feature matrix (X) is built from every available data source:
- Price history (OHLCV, returns, log returns)
- All 9 technical indicator values and signals
- FRED macro series (GDP, rates, VIX, yield curve)
- News sentiment scores (FinBERT)
- Short interest ratio
- Congressional trading signals
- Insider buy/sell activity
- Options data (Put/Call ratio, implied volatility)
- Dividend yield
- Market regime classification
- Peer relative strength
- Seasonality features (day-of-week, month, quarter)

### Walk-Forward Cross-Validation (`walk_forward.py`)
Standard cross-validation shuffles data randomly, which causes data leakage in time series
(training on future data). Walk-forward validation avoids this:
- Train on 2015–2019 data
- Test on 2020–2026 data
- Within each window, slide forward one period at a time
- This simulates how the model would have performed in real deployment

### Hyperparameter Tuning (`tuner.py`)
Optuna runs automated Bayesian hyperparameter optimization for each model. It searches the
parameter space efficiently (not random, not exhaustive) to find settings that minimize
validation error.

### Explainability (`feature_selection.py`)
SHAP (SHapley Additive exPlanations) values are computed for each prediction. SHAP answers
the question "which features drove this prediction and by how much?" — making the ML models
interpretable rather than black-box.

### Model File Integrity (`base.py`)
Every trained model is saved with an HMAC-SHA256 signature sidecar file (`.sig`). When a model
is loaded, the signature is verified before deserialization. If the file has been modified since
it was saved — whether by a corrupted write or deliberate tampering — loading is blocked and a
clear error is raised. The signing key is `ML_MODEL_SECRET` in `.env`. This applies to both
pickle files (`.pkl`) used by classical models and PyTorch checkpoints (`.pt`) used by LSTM,
GRU, and Transformer.

**CLI usage:**
```bash
# Train all models for a ticker
python cli/ml.py --ticker AAPL --train

# Generate predictions
python cli/ml.py --ticker AAPL --predict

# Skip deep learning (CPU-only, fast)
python cli/ml.py --ticker AAPL --train --models xgboost,lightgbm,rf
```

---

## 9. News & Sentiment Analysis (Phase 5)

**File:** `src/news/`

### News Fetching (`scraper.py`)
Fetches headlines and summaries from:
- Yahoo Finance RSS feeds per ticker
- Google News RSS feeds
- yfinance news API as fallback

Up to 50 articles per ticker per run are collected and stored as Parquet.

### FinBERT Sentiment Scoring (`sentiment.py`)
**FinBERT** is a version of Google's BERT language model fine-tuned specifically on financial
text (earnings reports, analyst notes, news articles). It classifies each piece of text as:
- **Positive** — bullish language
- **Negative** — bearish language
- **Neutral** — factual / no directional lean

Each article gets a numeric score in [-1, +1] (confidence × direction). The 7-day rolling
average sentiment score per ticker feeds into the ranking system.

FinBERT understands financial language that general sentiment models get wrong. For example,
"the company missed estimates" is clearly negative in finance — a model trained on general
text might not capture that nuance.

### Short Interest (`short_interest.py`)
Short interest (the percentage of shares sold short) is a contrarian indicator. Unusually high
short interest can signal either:
- Bearish sentiment from sophisticated traders (follow the shorts)
- Short squeeze potential (stocks with high short interest can spike dramatically if forced
  to cover)

The system tracks short interest ratio and flags stocks above a configurable threshold.

**CLI usage:**
```bash
python cli/news.py --premarket
python cli/news.py --tickers MSFT,TSLA
```

---

## 10. Political & Insider Trading Signals (Phase 6)

**File:** `src/political/`

### Congressional Trading (`congress_tracker.py`)
The STOCK Act requires members of Congress to disclose stock trades within 45 days. These
disclosures are public record. Congressional trades are significant because:
- Members of Congress have access to information during committee hearings that isn't public
- Academic research shows congressional stock trades outperform the market significantly
- A congressional buy is treated as a moderately bullish signal in the ranking system

Data source: Quiver Quant API, which aggregates STOCK Act disclosures.

### Insider Trading — SEC Form 4 (`insider_tracker.py`)
Corporate insiders (CEOs, CFOs, directors, >10% shareholders) must file SEC Form 4 within
2 business days of any trade. Insider buys are particularly meaningful:
- Insiders sell for many reasons (diversification, taxes, personal expenses)
- Insiders buy for essentially one reason: they believe the stock will go up
- Cluster buying (multiple insiders buying simultaneously) is a stronger signal

Data source: SEC EDGAR, parsed directly from the public filing database.

---

## 11. Options Flow & Calendar (Phase 7)

**File:** `src/options/`, `src/calendar/`

### Options Data (`options_data.py`)
- **Put/Call Ratio** — ratio of put options volume to call options volume. High put/call = bearish
  sentiment. Low put/call = bullish sentiment.
- **Unusual Volume** — flags when options volume exceeds 2x open interest, suggesting informed
  positioning ahead of a catalyst
- Source: yfinance options chains

### Implied Volatility (`implied_vol.py`)
- **Implied Volatility (IV)** — the market's forward-looking estimate of volatility, priced into
  options premiums
- **Historical Volatility (HV)** — realized price movement over the past 30 days
- When IV >> HV (IV spike), the options market is pricing in a major expected move — earnings,
  FDA approval, legal ruling, etc.

### Earnings Calendar (`earnings.py`)
- Upcoming earnings dates for all tickers
- Historical beat/miss record: how often has this stock beaten estimates?
- Earnings date proximity is a risk flag — positions near earnings face binary event risk

### Economic Calendar (`economic.py`)
- FOMC meeting dates (Federal Reserve interest rate decisions)
- CPI release dates (inflation data)
- Jobs report dates (Non-Farm Payrolls)
- These macro events move the entire market and are flagged in the dashboard

---

## 12. Stock Ranking System (Phase 8)

**File:** `src/ranking/ranker.py`

The ranker combines signals from every module into a single composite score per stock and
produces a sorted top-20 list.

### Signal Sources
| Source | Weight Category |
|--------|----------------|
| Long-term forecast direction (AutoBest) | Forecasting |
| Technical indicator composite score | Momentum/Trend |
| ML model consensus prediction | ML |
| News sentiment (7-day average) | Sentiment |
| Short interest signal | Contrarian |
| Congressional trading activity | Political |
| Insider buy/sell activity | Insider |
| Put/Call ratio + IV signal | Options |
| Market regime adjustment | Risk |
| Peer relative strength | Relative value |
| Dividend yield | Income |

### Fast Ranking vs Full Ranking
- **Fast ranking** (`include_ml=False`) — skips ML models, runs in seconds. Useful for quick
  intraday refreshes.
- **Full ranking** (`include_ml=True`) — includes all ML predictions. Takes longer but most
  comprehensive.

**CLI usage:**
```bash
python cli/rank.py --top 20
python cli/rank.py --top 20 --no-ml  # fast mode
```

---

## 13. Portfolio Analytics & Backtesting (Phase 9)

**File:** `src/trading/backtester.py`, `src/analytics/`

### Backtesting
The backtester simulates the full strategy on historical data from 2020 to 2026:
- Uses only data available at each point in time (no future data leakage)
- Models realistic trading costs: bid-ask spread, slippage
- Tracks every simulated trade: entry price, exit price, holding period, P&L
- Tests through COVID crash (March 2020), the 2022 bear market, and the 2023–2025 bull run
- Produces a full performance report vs S&P 500 buy-and-hold benchmark

### Portfolio Metrics (`portfolio_metrics.py`)
- **Sharpe Ratio** — risk-adjusted return (return per unit of volatility)
- **Sortino Ratio** — like Sharpe but only penalizes downside volatility
- **Maximum Drawdown** — largest peak-to-trough decline (measures worst-case loss)
- **Beta** — correlation with the S&P 500 (beta > 1 = more volatile than market)
- **Win Rate** — percentage of trades that were profitable

### Market Regime Detection (`market_regime.py`)
Classifies current market conditions as one of four regimes:
- **Bull** — strong uptrend, low VIX
- **Bear** — downtrend, elevated VIX
- **Sideways** — no clear trend, moderate VIX
- **High Volatility** — regime-agnostic but risk is elevated

Detection uses VIX level, market breadth (% of stocks above 200-day moving average), and
the yield curve spread.

### Sector Analysis (`sector_analysis.py`)
- Breaks down holdings by GICS sector
- Detects sector rotation (which sectors are strengthening/weakening)
- Flags overconcentration in a single sector

### Peer Comparison (`peer_comparison.py`)
Compares each stock's performance against its sector peers, identifying relative strength
(stocks outperforming their sector) and relative weakness.

**CLI usage:**
```bash
python cli/backtest.py --mode full-sim --train 2015-2020 --test 2020-2026
```

---

## 14. Trading Integration (Phase 10 + Smart Trade Management)

**File:** `src/trading/`

**Key features added in this phase:**
- High-conviction auto-buy filter (score ≥ 0.7 bypasses regime filters)
- Broker-side GTC trailing stops placed on Alpaca immediately after every BUY
- Per-position holding horizons (14 / 30 / 60 days) derived from signal sources
- Three-mode stop tightening (auto / on_date / manual) — 5% → 1% after horizon expires
- Manual order form in the Trading dashboard page with same trailing stop support
- Upgraded Stock Summary Quick Trade panel (buy with trail, sell with position metrics)
- Page 20 "Auto Trades" dashboard for complete buy history and exit calendar

### Alpaca Integration (`alpaca_client.py`)
Connects to the Alpaca brokerage API. The system defaults to **paper trading** (simulated money,
real market prices) and requires an explicit environment variable change to enable live trading.

### Strategy (`strategy.py`)
- Entry rules based on composite ranking score exceeding a threshold
- **Software trailing stop** — activated after 3 consecutive profitable days on a position;
  checked at each 15-minute cycle as a belt-and-suspenders backup
- Position is automatically exited if price drops 5% from its recent peak

### High-Conviction Filter
Any stock whose `composite_score >= 0.7` triggers an automatic BUY signal regardless of
momentum-strategy regime filters. Lower-conviction entries (`score >= 0.1`) still require
the full regime and momentum checks. The threshold is configurable via
`high_conviction_score_threshold` in `config/settings.py`.

### Broker-Side Trailing Stops (`alpaca_client.py`)
Every BUY — whether placed by the auto loop or by a manual order through the dashboard — is
immediately followed by a **GTC (Good-Till-Cancelled) trailing-stop sell order placed on
Alpaca**. Alpaca tracks the stop tick-by-tick in real time, not just at each 15-minute cycle.

The initial trail is **5%** (`broker_trailing_stop_pct`). After the position's holding horizon
expires the trail automatically tightens to **1%** (`broker_trailing_stop_tight_pct`), letting
winners keep running while ending the trade quickly on any reversal.

Three helper methods were added to `AlpacaClient`:
- `place_trailing_stop(ticker, qty, trail_percent)` — submits the GTC trailing-stop order
- `list_open_orders(ticker)` — returns all open orders for a symbol
- `cancel_order(order_id)` — cancels a single order by UUID (used when swapping 5% → 1%)

### Per-Position Holding Horizons (`derive_horizon_days`)
Every position gets a holding horizon derived from which signal sources drove the buy.
The function `derive_horizon_days(rank_entry)` in `src/trading/strategy.py` inspects the
top-3 signals by absolute value and applies this rule:

| Dominant sources | Horizon |
|---|---|
| ≥ 2 of: `options_flow`, `iv_signal`, `earnings_signal`, `forecast_signal`, `news_sentiment` | **14 days** (short) |
| ≥ 2 of: `insider_signal`, `congress_signal`, `short_interest` | **60 days** (long) |
| Mixed / `indicator_score`, `ml_signal` | **30 days** (medium) |

### Tighten Mode — Three Options
Each position independently controls *when* the 5% → 1% tightening happens via `tighten_mode`
stored in `PositionState` and persisted in `data/strategy_states.json`:

| Mode | Behaviour |
|---|---|
| `"auto"` | Tighten after `holding_horizon_days` have elapsed (default) |
| `"on_date"` | Tighten on a specific calendar date (`tighten_on_date` ISO string) |
| `"manual"` | Never auto-tighten — user clicks the button in the dashboard |

The tighten check runs at the start of every `cli/trade.py` cycle (Step 4b) and is idempotent:
once `broker_trail_pct` has flipped to 1%, the check is skipped on all subsequent cycles.

### PositionState fields (persisted to `data/strategy_states.json`)
```
ticker, entry_price, entry_date, peak_price
trailing_stop_active, trailing_stop_price    ← software stop (backup)
profitable_days_streak, last_checked_date
holding_horizon_days                          ← 14 / 30 / 60
broker_trail_pct                              ← 5.0 → 1.0 after tighten
tighten_mode                                  ← "auto" | "on_date" | "manual"
tighten_on_date                               ← ISO date string (on_date mode)
```

### Trade Loop — `cli/trade.py`

Each 15-minute cycle executes these steps in order:

| Step | What happens |
|------|---|
| 1 | Fetch current portfolio value and cash |
| 2 | Load all existing Alpaca positions |
| 3 | Check circuit breaker (portfolio or single-stock threshold) |
| 4 | **Force-sell** any position whose software trailing stop was hit |
| **4b** | **Tighten check** — for every held position, evaluate `tighten_mode` and flip `broker_trail_pct` from 5% → 1% on Alpaca if the condition is met |
| 5 | Run the market-regime detector |
| 6 | Run the composite ranker on all S&P 500 tickers |
| **6b** | **High-conviction pass** — collect tickers with `composite_score >= 0.7`; build `hc_signals` list injected at the front of `all_signals` so they are processed first |
| 7 | Run all signal generators (momentum, ML, news, options, insider, etc.) |
| 8 | Merge signals; strategy manager decides BUY / SELL / HOLD per ticker |
| 9 | Execute orders via Alpaca; for every BUY: `place_trailing_stop()` (5% GTC) + create and persist `PositionState` with horizon/trail/tighten fields |
| 10 | Log every trade to `TradeJournal` |
| 11 | Export `strategy_states.json` so the dashboard can read live state |
| 12 | Sleep until next cycle |

### Manual Orders (page 17 — Trading)

The **📝 Place Manual Order** form in the Trading page mirrors the auto-loop buy path:
- Places the order via `AlpacaClient.place_order()`
- Optionally calls `place_trailing_stop()` if the "Attach trailing stop" checkbox is ticked
- Writes a `PositionState` entry to `data/strategy_states.json` with the same tighten/horizon fields

This ensures manual trades appear in the **Exit Calendar** and get the same automatic tightening
as system-generated buys.

### Quick Trade (page 2a — Stock Summary)

The **🟢 BUY** panel on the Stock Summary page adds:
- **Trailing stop checkbox** — attach a GTC trailing stop immediately after the fill
- **Tighten mode** — choose `auto` (after N days), `on_date` (specific calendar date), or `manual`
- **Horizon / exact date** fields that appear conditionally based on the chosen mode

The **🔴 SELL** panel adds:
- Live position metrics (avg entry, current price, unrealized P&L)
- **Sell entire position** toggle (auto-fills quantity from Alpaca position)
- **Cancel trailing-stop orders** checkbox (cancels any open GTC sell orders before submitting)

### Multi-Strategy (`multi_strategy.py`)
Runs three strategies simultaneously and allocates capital based on recent performance:
- **Value strategy** — targets undervalued stocks (low P/S, high dividend yield)
- **Momentum strategy** — targets stocks with strong price and indicator momentum
- **Mean reversion strategy** — targets stocks that have pulled back from overbought levels

### Circuit Breaker (`circuit_breaker.py`)
An automatic kill switch with three levels:
- **Portfolio -10% in one day** — halt all new trades
- **Single stock -15%** — force-sell that position immediately
- **Manual emergency stop** — button in the dashboard halts everything

### Risk Management (`risk.py`)
- Maximum 5% of portfolio in any single stock
- Position sizing scales down in Bear or High-Volatility regimes
- Optional Kelly Criterion sizing (fractional, for safety)

### Tax Lots (`tax_lots.py`)
Tracks cost basis per share lot using FIFO/LIFO/specific lot methods. Identifies tax-loss
harvesting opportunities (sell losers to offset gains). Distinguishes short-term vs long-term
capital gains.

### Trade Journal (`trade_journal.py`)
Every trade is logged with:
- Entry and exit price and date
- Which signals triggered the entry decision
- P&L and holding period
- Exit reason (signal reversal, trailing stop, circuit breaker, etc.)

**CLI usage:**
```bash
python cli/trade.py --mode paper
python cli/trade.py --mode live  # requires ALPACA_LIVE=true in .env
```

---

## 15. Alerts & Morning Report (Phase 11)

**File:** `src/alerts/`, `src/reports/`

### Alerts (`notifier.py`)
Multi-channel notifications sent automatically when:
- A new BUY or SELL signal crosses threshold
- A trailing stop is hit
- News sentiment flips from positive to negative (or vice versa)
- Earnings are approaching within 5 days
- Congressional trading is detected on a held position
- Circuit breaker is triggered

Channels supported: Discord webhook, email (SMTP).

### Morning Report (`morning_report.py`)
Generated every day at 7 AM before market open. Contains:
- Top 20 ranked picks with composite scores
- Portfolio current status and overnight P&L
- Upcoming earnings and economic events for the week
- Key signals from overnight news
- Market regime classification

Output formats: PDF (via fpdf2) and Excel (via openpyxl).

**CLI usage:**
```bash
python cli/report.py --morning
```

---

## 16. Automation (Phase 12)

**File:** `cli/scheduler.py`

APScheduler runs the full daily pipeline automatically on a wall-clock schedule:

| Time | Job |
|------|-----|
| 6:00 AM | Incremental data scrape (prices, dividends, macro, options, political, news) |
| 6:15 AM | Market regime detection |
| 6:30 AM | Forecasts + indicators + ML predictions + peer comparison |
| 7:00 AM | Generate morning report, send alerts |
| 9:30 AM | Market open — multi-strategy executor starts, circuit breaker armed |
| During hours | Monitor trailing stops, unusual options, news alerts |
| 4:00 PM | EOD snapshot, portfolio analytics, trade journal update |
| Weekly | Tax-loss harvesting scan, strategy performance review |

**CLI usage:**
```bash
python cli/scheduler.py  # starts the automated daily pipeline
```

---

## 17. The Dashboard — All 22 Pages

Run with: `streamlit run dashboard/app.py`

| # | Page | What You Can Do |
|---|------|----------------|
| Home | app.py | System status overview, quick top-10 ranking |
| 1 | Data Overview | Browse available ticker data, date ranges, data quality stats |
| 2 | Morning Report | View and generate the daily morning briefing |
| 2a | Stock Summary | Deep-dive on any single ticker: signals, ML, options, news, and a **Quick Trade** panel to buy/sell with automatic GTC trailing stop, tighten mode, and holding horizon |
| 3 | Long-Term Forecast | Compare all 12 forecast methods for any ticker, overlay chart |
| 4 | Short-Term Signals | View all 9 indicators, composite signal, heatmap across tickers |
| 5 | ML Predictions | Model results, RMSE comparison, SHAP feature importance charts |
| 6 | Stock Rankings | Full top-20 composite ranking with per-signal breakdown |
| 7 | News Sentiment | Recent headlines, FinBERT scores, sentiment trend chart |
| 8 | Political & Insider | Congressional STOCK Act trades, SEC Form 4 insider filings |
| 9 | Options Flow | Put/Call ratio, unusual volume flags, IV vs historical vol |
| 10 | Portfolio | Holdings, P&L, correlation matrix, sector breakdown, attribution |
| 11 | Sector Analysis | Sector rotation view, concentration warnings, top picks by sector |
| 12 | Market Regime | Bull/Bear/Sideways/High-Vol indicator, VIX dashboard, breadth |
| 13 | Backtesting | Run historical simulations, equity curve, vs S&P 500 benchmark |
| 14 | Calendar | Upcoming earnings, FOMC, CPI, jobs reports |
| 15 | Watchlist | Custom personal watchlist with multi-timeframe analysis |
| 16 | Peer Comparison | Stock vs sector peers relative strength chart |
| 17 | Trading | Paper/live trading controls, circuit breaker status, order log; **manual order form** with optional GTC trailing stop and tighten-mode settings |
| 18 | Trade Journal | Full trade history, audit log, tax lot view |
| 19 | Alerts | Configure and test notification channels |
| 20 | Auto Trades | **Auto-Buy Log** (every automatic purchase with strategy/score/size) + **Exit Calendar** (holding horizon progress, exact exit date, days remaining, trail status, per-position tighten controls) |
| 21 | Supply Chain | Interactive directed buyer→supplier network (arrows point supplier→buyer), nodes sized by market cap and coloured by sector; filter by neighbourhood/sector/dependency, add relationships manually, and run the **Research suppliers** agent with an inline review-and-approve list |
| 22 | Deep Research | Standalone control surface for the supplier-discovery agent — pick a target ticker, endpoint/model, and round budget; live progress; review discovered suppliers with their **source links** before promoting them to the graph |

---

### Supply Chain Map & Deep Research (pages 21–22)

The **Supply Chain Map** renders buyer-supplier relationships as a directed graph
(`src/analytics/supply_chain.py` + `dashboard/components/charts.py`). An arrow **supplier → buyer**
means the buyer depends on that supplier (`TSM → AAPL` = "AAPL buys from TSM"). Nodes are sized by
market cap and coloured by sector; `dependency_pct` is an *estimate* of the share of the supplier's
revenue that comes from that buyer. The graph seeds from a curated set checked into
`src/analytics/supply_chain_seed.py` (because `data/` is gitignored) and grows through the manual
"Add relationship" form or the research agent.

**Deep Research** grows the map automatically. A local **Ollama** model plans web searches, a local
**SearXNG** instance runs them, and the agent reads results and extracts candidate suppliers over
several rounds — the number of rounds scales with how hard the target is (the "Auto" setting).
Every candidate must carry a **source link**; findings land in a per-ticker review queue
(`data/research/<ticker>.json`) and only reach the graph when you approve them, at which point they
are written with `source="research"` and the evidence link attached. Configure the endpoints with
`OLLAMA_BASE_URL` (default `http://localhost:11434`) and `SEARX_BASE_URL` (default
`http://localhost:8080`). CLI: `python cli/research.py --ticker AAPL` (or `--list-models`).

---

## 18. How to Install & Run

### Prerequisites
- Python 3.11 or higher
- FRED API key (free at fred.stlouisfed.org)
- Alpaca paper trading account (free at alpaca.markets) — optional, only needed for trading

### Step 1 — Clone and set up environment
```bash
git clone <repo>
cd AutoStockAnalyzer
```

### Step 2 — Install dependencies
```bash
pip install yfinance fredapi pyarrow pandas numpy openpyxl statsmodels scikit-learn xgboost lightgbm optuna shap ta alpaca-trade-api requests feedparser beautifulsoup4 transformers sentencepiece discord-webhook fpdf2 python-docx streamlit plotly python-dotenv pydantic-settings pydantic apscheduler defusedxml
```

PyTorch (CPU, no GPU required):
```bash
pip install torch torchvision torchaudio
```

### Step 3 — Configure API keys
Create a `.env` file in the project root (copy `.env.example` as a starting point):
```
FRED_API_KEY=your_fred_key_here
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
DISCORD_WEBHOOK_URL=your_discord_webhook_here  # optional

# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
ML_MODEL_SECRET=your_generated_hex_key_here
```

`ML_MODEL_SECRET` is used to sign saved ML model files with HMAC-SHA256. Without it, model
integrity checking is skipped with a warning. Generate once and keep it in `.env` (which is
gitignored and never committed).

### Step 4 — Run the initial data backfill
This downloads 11 years of data for all S&P 500 stocks. Takes 15–20 minutes on first run.
```bash
python cli/scrape.py --backfill 2015
```

### Step 5 — Launch the dashboard
```bash
streamlit run dashboard/app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 19. Daily Workflow — How to Use It

### Morning (before market open)
1. The scheduler has already run at 6 AM — all data is fresh
2. Open dashboard → **Morning Report** page for the day's briefing
3. Review **Stock Rankings** page — top 20 picks with signal breakdown
4. Check **Calendar** page — any earnings or macro events today?
5. Review **Market Regime** — are we in Bull/Bear/Sideways? Adjust conviction accordingly
6. Check **News Sentiment** — any overnight sentiment flips on held positions?
7. Check **Political & Insider** — any congressional or insider activity?

### Midday
- **Short-Term Signals** page for intraday indicator updates
- **Options Flow** page for unusual activity that might indicate an upcoming move

### End of Day
- **Portfolio** page for daily P&L and position review
- **Trade Journal** — automatic entry logged for any trades executed
- **Sector Analysis** — are positions still diversified appropriately?

### Weekly
- **Backtesting** page to review strategy performance vs benchmark
- **Peer Comparison** — are holdings still outperforming their sector peers?
- **Portfolio → Attribution** — which positions are contributing/detracting?

---

## 20. The Role of AI in This System

AutoStockAnalyzer uses AI at four distinct layers:

### Layer 1 — Classical Statistical AI
The 12 forecasting methods (exponential smoothing, decomposition, regression) are established
statistical learning techniques. They are interpretable by design — you can see exactly what
the trend and seasonal components look like and understand why the forecast goes where it does.

### Layer 2 — Ensemble Machine Learning
XGBoost, LightGBM, and Random Forest are gradient boosting and ensemble methods that learn
non-linear patterns in the feature matrix. These models can find interactions between variables
that humans wouldn't think to look for — e.g., the combination of high short interest + positive
congressional buying + rising momentum might be a stronger signal than any one factor alone.

### Layer 3 — Deep Learning (Sequential)
LSTM, GRU, and the Temporal Fusion Transformer process price data as sequences, maintaining
memory of past patterns over long windows. These models are particularly suited to detecting
regime changes and multi-step dependencies in financial time series.

### Layer 4 — Natural Language AI (FinBERT)
FinBERT applies transformer-based natural language understanding to financial text. It has been
fine-tuned on financial corpora and understands the specific vocabulary and framing of financial
news in a way that general sentiment models do not.

### Why Multiple AI Methods?
No single AI method dominates across all market conditions. The AutoBest system explicitly
acknowledges this: it runs all 12 statistical methods and selects the best one for each stock.
The same philosophy applies to the ML layer — 7 models run simultaneously and their predictions
can be compared or ensembled.

This system treats AI as a **set of analytical lenses**, not an oracle. Each method illuminates
a different aspect of the data. The human analyst's job is to look across all lenses and make
an informed judgment — not to follow any single output blindly.

---

*AutoStockAnalyzer — Built as an application of university coursework in forecasting, machine
learning, and data engineering to real-world financial analysis.*

*This document is for educational purposes. Nothing herein constitutes financial advice.*
