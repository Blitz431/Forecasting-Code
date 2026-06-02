# AutoStockAnalyzer — Architecture & Module Dependency Map

This document shows how every module in the project connects to every other module,
where data is read and written, and how the phases wire together at runtime.

For the build plan and what each phase implements, see `recursive-gliding-dove.md`.
For installation instructions, see `INSTALL.md`.

---

## High-Level Data Flow

```
[yfinance] ──────────────────────────────────────────────────────┐
[FRED API] ──> scraper/ ──> data/raw/ (Parquet) ──> forecasting/ ─┤
[yfinance] ──────────────────────────────────────────────────────┤         ┌──> ranker/ ──> Top 20
                                                                  ├──> ml/ ─┤
[RSS/News] ──> news/scraper ──> data/news/ (Parquet) ─────────────┤         └──> dashboard/
                                                                  │
[yfinance] ──> indicators/ ───────────────────────────────────────┤
                                                                  │
[Quiver/SEC] ──> political/ ──────────────────────────────────────┤
[yfinance]  ──> options/    ──────────────────────────────────────┘

config/settings.py ──> everything (all modules read settings)
src/utils/         ──> everything (logging, tickers, validation)
```

---

## Daily Runtime Cycle

```
6:00 AM  cli/scrape.py         ──> scraper/ ──> data/raw/
6:15 AM  cli/indicators.py     ──> indicators/ (reads data/raw/daily/)
6:30 AM  cli/forecast.py       ──> forecasting/ (reads data/raw/quarterly/)
6:30 AM  cli/ml.py --predict   ──> ml/ (reads data/raw/ + data/processed/)
6:30 AM  cli/news.py           ──> news/ (RSS feeds + yfinance.news)
7:00 AM  cli/report.py         ──> reports/ (reads all of the above)
9:30 AM  cli/trade.py          ──> trading/ (reads ranking output + live signals)
4:00 PM  EOD snapshot          ──> analytics/ (reads trading/ + data/raw/)
```

---

## Module Dependency Graph

Arrows mean "imports from". Only internal `src/` dependencies are shown.

```
config/settings.py
    └── (no src/ imports — root of dependency tree)

src/utils/logging.py
    └── config.settings

src/utils/dates.py
    └── (no src/ imports)

src/utils/tickers.py
    └── src.utils.logging

src/utils/validation.py
    └── src.utils.logging

src/scraper/storage.py
    ├── src.utils.logging
    └── src.utils.validation

src/scraper/price_scraper.py
    ├── config.settings
    ├── src.scraper.storage
    ├── src.utils.logging
    └── src.utils.validation

src/scraper/dividend_scraper.py
    ├── config.settings
    ├── src.scraper.storage
    └── src.utils.logging

src/scraper/macro_scraper.py
    ├── config.settings
    ├── src.scraper.storage
    └── src.utils.logging

src/indicators/base.py
    └── (no src/ imports)

src/indicators/_calc.py
    └── (no src/ imports — pure math)

src/indicators/rsi.py
    ├── config.settings
    ├── src.indicators._calc
    └── src.indicators.base

src/indicators/macd.py
    ├── src.indicators._calc
    └── src.indicators.base

src/indicators/bollinger.py
    ├── src.indicators._calc
    └── src.indicators.base

src/indicators/stochastic.py
    ├── config.settings
    ├── src.indicators._calc
    └── src.indicators.base

src/indicators/moving_averages.py
    ├── src.indicators._calc
    └── src.indicators.base

src/indicators/momentum.py
    ├── src.indicators._calc
    └── src.indicators.base

src/indicators/volume.py
    ├── src.indicators._calc
    └── src.indicators.base

src/indicators/correlation.py
    ├── src.indicators._calc
    └── src.indicators.base

src/indicators/fundamentals.py
    ├── src.indicators._calc
    └── src.indicators.base

src/indicators/signal_aggregator.py
    ├── ALL indicator classes above
    ├── src.indicators.base
    └── src.utils.logging

src/forecasting/base.py
    └── (no src/ imports)

src/forecasting/metrics.py
    └── (no src/ imports)

src/forecasting/decomposition.py
    └── src.forecasting.base

src/forecasting/moving_average.py
    └── src.forecasting.base

src/forecasting/exponential_smoothing.py
    └── src.forecasting.base

src/forecasting/regression.py
    └── src.forecasting.base

src/forecasting/auto_best.py
    ├── ALL method classes above
    ├── src.forecasting.metrics
    └── src.utils.logging

src/forecasting/runner.py
    ├── config.settings
    ├── ALL method classes
    ├── src.forecasting.metrics
    ├── src.scraper.storage
    └── src.utils.logging

src/ml/base.py
    └── (no src/ imports — stdlib only: os, hashlib, hmac for model integrity)

src/ml/feature_engineer.py
    ├── config.settings
    ├── src.scraper.storage
    └── src.utils.logging

src/ml/feature_selection.py
    ├── src.ml.xgboost_model
    ├── src.ml.random_forest
    └── src.utils.logging

src/ml/walk_forward.py
    ├── config.settings
    └── src.utils.logging

src/ml/linear_models.py
    └── src.utils.logging

src/ml/random_forest.py
    └── src.utils.logging

src/ml/xgboost_model.py
    ├── src.ml.base
    └── src.utils.logging

src/ml/lstm_model.py
    ├── src.ml.base
    └── src.utils.logging

src/ml/transformer_model.py
    ├── src.ml.base
    └── src.utils.logging

src/ml/tuner.py
    ├── src.ml.xgboost_model
    ├── src.ml.random_forest
    └── src.utils.logging

src/ml/runner.py
    ├── config.settings
    ├── ALL model classes
    ├── src.ml.feature_engineer
    ├── src.ml.walk_forward
    ├── src.ml.feature_selection
    ├── src.scraper.storage
    └── src.utils.logging

src/news/scraper.py
    └── src.utils.logging

src/news/sentiment.py
    └── src.utils.logging

src/news/short_interest.py
    ├── config.settings
    ├── src.scraper.storage
    └── src.utils.logging

src/news/aggregator.py
    ├── src.scraper.storage
    └── src.utils.logging

src/news/runner.py
    ├── config.settings
    ├── src.news.scraper
    ├── src.news.sentiment
    ├── src.news.short_interest
    ├── src.news.aggregator
    └── src.utils.logging
```

---

## CLI to Module Mapping

| CLI Script | Calls Into | Reads From | Writes To |
|------------|-----------|------------|-----------|
| `cli/scrape.py` | scraper/price, dividend, macro | External APIs | data/raw/ |
| `cli/indicators.py` | indicators/signal_aggregator | data/raw/daily/ | (in-memory only) |
| `cli/forecast.py` | forecasting/runner | data/raw/quarterly/ | data/forecasts/ |
| `cli/ml.py` | ml/runner | data/raw/, data/processed/ | data/processed/, model files |
| `cli/news.py` | news/runner | RSS feeds, yfinance.news | data/news/ |

---

## Parquet Storage Map

All persistent data lives under `data/`. Each ticker gets its own `.parquet` file.

```
data/
├── raw/
│   ├── daily/          ── written by: scraper/price_scraper.py
│   │   └── AAPL.parquet   read by:    indicators/, ml/feature_engineer.py
│   │       Columns: Open, High, Low, Close, Volume
│   │       Index:   DatetimeIndex (daily, UTC)
│   │
│   ├── quarterly/      ── written by: scraper/price_scraper.py (aggregate_to_quarterly)
│   │   └── AAPL.parquet   read by:    forecasting/runner.py
│   │       Columns: Open, High, Low, Close, Volume
│   │       Index:   DatetimeIndex (QE, quarter-end aligned)
│   │
│   ├── macro/          ── written by: scraper/macro_scraper.py
│   │   └── GDP.parquet    read by:    ml/feature_engineer.py
│   │       Columns: GDP (one column named after the series)
│   │       Index:   DatetimeIndex
│   │
│   └── dividends/      ── written by: scraper/dividend_scraper.py
│       └── AAPL.parquet   read by:    ml/feature_engineer.py
│           Columns: Dividends
│           Index:   DatetimeIndex (ex-dividend dates)
│
├── forecasts/          ── written by: forecasting/runner.py
│   └── AAPL_forecasts.parquet
│       Columns: Ticker, Method_Number, Method_Name, Forecast_Date,
│                Forecast_Price, RMSE, MAE, MAPE
│
├── processed/          ── written by: ml/runner.py
│   └── AAPL_ml_results.parquet + trained model files (.pkl, .pt)
│       Each saved model also gets a .sig sidecar (HMAC-SHA256 of the
│       model file, keyed by ML_MODEL_SECRET from .env). load_from_disk()
│       verifies the sig before deserializing — tampered files are rejected.
│       Columns: model_name, ticker, rmse, mae, mape, train_start/end,
│                test_start/end, feature_importance (JSON)
│
└── news/
    ├── articles/       ── written by: news/aggregator.py
    │   └── AAPL.parquet   read by:    news/aggregator.py (sentiment queries)
    │       Columns: headline, summary, url, source,
    │                sentiment_label, sentiment_score, confidence
    │       Index:   DatetimeIndex (UTC publish time)
    │
    └── short_interest/ ── written by: news/short_interest.py
        └── AAPL.parquet   read by:    news/short_interest.py (signal queries)
            Columns: short_ratio, short_pct_float, shares_short, shares_float,
                     high_short_interest
            Index:   DatetimeIndex (daily, date of fetch)
```

---

## Key Design Patterns

### 1. Strategy Pattern
Every indicator, forecast method, and ML model implements an abstract base class.
The runner calls them all the same way without knowing which one it's running.

```
Indicator ABC (base.py)         ──> RSIIndicator, MACDIndicator, etc.
ForecastMethod ABC (base.py)    ──> AdditiveDecomposition, HoltWinters, etc.
MLModel ABC (base.py)           ──> XGBoostModel, LSTMModel, etc.
```

### 2. Upsert Storage
`scraper/storage.py:upsert_dataframe()` is used everywhere data is saved.
It loads the existing Parquet, concatenates new rows, deduplicates by index,
sorts, and saves. This means every scrape is safe to re-run — no duplicate rows.

### 3. Settings as Single Source of Truth
`config/settings.py` holds all thresholds, paths, and parameters.
No magic numbers in module code — everything references `get_settings()`.
New phases add their settings fields here.

### 4. Exception-Safe Compute
Every indicator has a `safe_compute(df)` wrapper that catches exceptions and
returns an error `IndicatorResult` instead of crashing the whole pipeline.
The same pattern is used in forecasting `evaluate()` and ML `runner.py`.

### 5. Walk-Forward Validation (ML)
`ml/walk_forward.py` ensures no future data leaks into training.
Default split: train on 2015–2019, test on 2020–2026.
Walk-forward mode slides a window forward in 63-day steps (≈1 quarter).

### 6. Lazy Imports for Heavy Dependencies
`news/sentiment.py` imports `torch` and `transformers` inside `__init__`
rather than at module level. This lets the whole package import even if
PyTorch is not installed, so tests and other modules are not blocked.

### 7. Model File Integrity (HMAC-SHA256)
Every model file written by `MLModel.save()` — whether a pickle (`.pkl`) for
classical models or a `torch.save` (`.pt`) for deep learning — gets a companion
`.sig` sidecar file written alongside it. The sidecar contains the
HMAC-SHA256 of the model bytes keyed by `ML_MODEL_SECRET` from `.env`.

`MLModel.load_from_disk()` calls `_verify_sig()` before deserializing. If the
sig file is missing the load proceeds with a warning (backward compatibility for
models saved before this feature). If the sig exists but doesn't match, a
`ValueError` is raised and deserialization is blocked.

Utility functions live in `src/ml/base.py`: `_write_sig()`, `_verify_sig()`,
`_model_secret()`, `_sig_path()`.

### 8. Defusedxml for External XML
`src/political/insider_tracker.py` parses SEC Form 4 filings (EDGAR XML).
It uses `defusedxml.ElementTree` instead of the stdlib `xml.etree.ElementTree`
to guard against XML bomb and related attacks on the parser itself. The API is
identical — only the import changed.

---

## Where Future Phases Plug In

```
Phase 6 — Political/Insider
  src/political/congress_tracker.py  ──> reads: Quiver Quant (HTTP)
  src/political/insider_tracker.py   ──> reads: SEC EDGAR (HTTP)
  Both write to: data/political/
  Feed into: ml/feature_engineer.py (congressional/insider signals as features)
             ranking/ranker.py (buy signal boost)

Phase 7 — Options & Calendar
  src/options/options_data.py        ──> reads: yfinance options chains
  src/options/implied_vol.py         ──> reads: yfinance options chains
  src/calendar/earnings.py           ──> reads: yfinance
  src/calendar/economic.py           ──> reads: FRED + static calendar
  All write to: data/options/, data/calendar/
  Feed into: ml/feature_engineer.py, ranking/ranker.py

Phase 8 — Ranking + Dashboard
  src/ranking/ranker.py
    reads from: data/forecasts/, data/processed/, data/news/,
                data/political/, data/options/ (Phases 1-7 outputs)
    produces:   top 20 picks with composite score
  dashboard/app.py + pages/
    reads from: ALL data/ directories
    runs as:    streamlit run dashboard/app.py

Phase 9 — Backtesting & Analytics
  src/trading/backtester.py
    reads from: data/raw/daily/ (replay 2020-2026 day by day)
                ranking output (signals computed with no-future-data constraint)
  src/analytics/*
    reads from: trading/portfolio.py, data/raw/

Phase 10 — Trading
  src/trading/alpaca_client.py       ──> reads: Alpaca API (live/paper)
  src/trading/strategy.py            ──> reads: ranking/ranker.py output
  src/trading/circuit_breaker.py     ──> reads: trading/portfolio.py
  All feed into: trading/trade_journal.py, trading/tax_lots.py

Phase 11 — Alerts & Reports
  src/alerts/notifier.py             ──> triggered by: all phases
  src/reports/morning_report.py      ──> reads: ALL data/ + ranking output
```

---

## Signal Flow into the Ranker (Phase 8)

When `ranking/ranker.py` runs, it will pull from every upstream module:

| Signal Source | Data Location | Weight (planned) |
|---------------|---------------|-----------------|
| Long-term forecasts (Phase 2) | data/forecasts/ | High |
| Technical indicators (Phase 3) | in-memory from data/raw/daily/ | High |
| ML predictions (Phase 4) | data/processed/ | High |
| News sentiment (Phase 5) | data/news/articles/ | Medium |
| Short interest (Phase 5) | data/news/short_interest/ | Medium |
| Congressional trades (Phase 6) | data/political/ | Medium |
| Insider filings (Phase 6) | data/political/ | Medium |
| Options flow — Put/Call (Phase 7) | data/options/ | Medium |
| Implied volatility (Phase 7) | data/options/ | Low |
| Earnings calendar (Phase 7) | data/calendar/ | Low |
| Peer relative strength (Phase 9) | in-memory | Medium |
| Market regime (Phase 9) | in-memory from macro | Medium |
