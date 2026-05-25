# AutoStockAnalyzer

Automated stock analysis, forecasting, and ranking system for the S&P 500.
Provides mid-to-long term buy/hold suggestions using forecasting, ML, technical indicators, news sentiment, political/insider signals, and options flow.

## Quick Start

Double-click **`Launch App.bat`** in the project folder, or run from the terminal:

```
streamlit run dashboard/app.py
```

## Dashboard Pages

| # | Page | Description |
|---|------|-------------|
| 1 | Data Overview | Browse available ticker data, date ranges, and quality stats |
| 2 | Long-Term Forecast | All 12 forecast methods per ticker with RMSE comparison |
| 3 | Short-Term Signals | 9 technical indicators, composite signal, heatmap |
| 4 | ML Predictions | XGBoost / LightGBM / LSTM / Transformer results |
| 5 | Stock Rankings | Full top-20 composite ranking with per-signal breakdown |
| 6 | News & Sentiment | Headlines, FinBERT sentiment scores, short interest |
| 7 | Political & Insider | Congressional STOCK Act trades, SEC Form 4 filings |
| 8 | Options Flow | Put/Call ratio, unusual volume, implied vs historical vol |
| 9 | Sector Analysis | Sector rotation, concentration, top picks by sector |
| 10 | Portfolio | Portfolio analytics, correlation matrix, attribution |
| 11 | Backtesting | Strategy simulation 2020–2026 with equity curve |
| 12 | Calendar | Upcoming earnings dates and economic events |
| 13 | Market Regime | Bull/Bear/Sideways detection, VIX, breadth indicators |
| 14 | Watchlist | Custom watchlist with multi-timeframe analysis |
| 15 | Trade Journal | Trade history, audit log, tax lot view |
| 16 | Peer Comparison | Stock vs sector peers, relative strength |
| 17 | Trading | Live/paper trading controls and circuit breaker |
| 18 | Alerts | Alert configuration and notification channels |
| 19 | Morning Report | Daily pre-market summary with top picks |
| 20 | Settings | API keys, data scraper launcher, thresholds, data paths |

## Data Scraper

Run from the **Settings** page (page 20 in the sidebar) or via CLI:

```
# Daily incremental update (~2–5 min)
python cli/scrape.py

# Full backfill from 2015 for all ~500 S&P 500 tickers (~15–30 min)
python cli/scrape.py --backfill
```

## Setup

See [INSTALL.md](INSTALL.md) for full installation instructions and API key setup.
See [ARCHITECTURE.md](ARCHITECTURE.md) for module dependency maps and data flow.

