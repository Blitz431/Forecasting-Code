# AutoStockAnalyzer — Installation Guide

Install everything below before running any part of the program.
Run each `pip install` command in your terminal.

---

## Launching the App

Double-click **`Launch App.bat`** in the project root, or run from the terminal:

```
streamlit run dashboard/app.py
```

---

## Quick Install (all at once)

```
pip install yfinance fredapi pyarrow pandas numpy openpyxl statsmodels scikit-learn xgboost lightgbm optuna shap ta alpaca-trade-api requests feedparser beautifulsoup4 transformers sentencepiece discord-webhook fpdf2 python-docx streamlit plotly python-dotenv pydantic-settings pydantic apscheduler
```

For PyTorch (GPU version for your NVIDIA 3060 Ti):
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For development/testing only:
```
pip install pytest pytest-cov ruff
```

---

## Package-by-Package Breakdown

### Data (Phase 1)
| Package | Install | Why |
|---------|---------|-----|
| `yfinance` | `pip install yfinance` | Downloads stock price history, dividends, fundamentals, and news from Yahoo Finance |
| `fredapi` | `pip install fredapi` | Pulls macro data from the Federal Reserve (GDP, CPI, interest rates, VIX, etc.) |
| `pyarrow` | `pip install pyarrow` | Engine for reading/writing Parquet files — all price and indicator data is stored in Parquet format |
| `pandas` | `pip install pandas` | Core data manipulation — DataFrames are used everywhere |
| `numpy` | `pip install numpy` | Numerical arrays — used by forecasting, ML, and indicators |
| `openpyxl` | `pip install openpyxl` | Exports DataFrames to Excel files for manual review |

### Forecasting (Phase 2)
| Package | Install | Why |
|---------|---------|-----|
| `statsmodels` | `pip install statsmodels` | Powers all 12 long-term forecast methods: seasonal decomposition, Holt-Winters, exponential smoothing, OLS regression |
| `scikit-learn` | `pip install scikit-learn` | Linear models (Ridge, Lasso, ElasticNet), metrics (RMSE, MAE), and preprocessing |

### Technical Indicators (Phase 3)
| Package | Install | Why |
|---------|---------|-----|
| `ta` | `pip install ta` | Computes all technical indicators: RSI, MACD, Bollinger Bands, Stochastic Oscillator, EMA/SMA, volume analysis |

### Machine Learning (Phase 4)
| Package | Install | Why |
|---------|---------|-----|
| `torch` | See GPU install above | Deep learning framework — powers the LSTM, GRU, and Transformer models. GPU version required for training speed on your 3060 Ti |
| `xgboost` | `pip install xgboost` | XGBoost gradient boosting model — one of the best performing ML models for tabular stock data |
| `lightgbm` | `pip install lightgbm` | LightGBM gradient boosting — faster than XGBoost on large datasets |
| `optuna` | `pip install optuna` | Hyperparameter tuning — automatically finds the best settings for each ML model |
| `shap` | `pip install shap` | SHAP feature importance — explains which indicators/signals matter most for each prediction |

### News & Sentiment (Phase 5)
| Package | Install | Why |
|---------|---------|-----|
| `requests` | `pip install requests` | Makes HTTP requests to news APIs and external data sources |
| `feedparser` | `pip install feedparser` | Parses Yahoo Finance and Google News RSS feeds to pull headlines |
| `beautifulsoup4` | `pip install beautifulsoup4` | Strips HTML tags from news article summaries |
| `transformers` | `pip install transformers` | Loads FinBERT — a BERT model fine-tuned on financial text for sentiment scoring. First run downloads ~400 MB model from HuggingFace and caches it locally |
| `sentencepiece` | `pip install sentencepiece` | Required by the FinBERT tokenizer to process text |

### Trading (Phase 10)
| Package | Install | Why |
|---------|---------|-----|
| `alpaca-trade-api` | `pip install alpaca-trade-api` | Connects to Alpaca broker for paper and live trading — placing orders, checking positions, getting account status |

### Alerts (Phase 11)
| Package | Install | Why |
|---------|---------|-----|
| `discord-webhook` | `pip install discord-webhook` | Sends trade alerts and morning report summaries to a Discord channel |

### Automation (Phase 12)
| Package | Install | Why |
|---------|---------|-----|
| `apscheduler` | `pip install apscheduler` | Schedules the daily pipeline jobs by wall-clock time — scrape at 6 AM, indicators at 6:15, forecasts + ML at 6:30, morning report at 7 AM, EOD snapshot at 4 PM |

### Reports (Phase 11)
| Package | Install | Why |
|---------|---------|-----|
| `fpdf2` | `pip install fpdf2` | Generates the daily morning PDF report with top picks, signals, and portfolio status |

### Dashboard (Phase 8)
| Package | Install | Why |
|---------|---------|-----|
| `streamlit` | `pip install streamlit` | Runs the web dashboard — `streamlit run dashboard/app.py` |
| `plotly` | `pip install plotly` | Interactive charts used throughout all 20 dashboard pages |

### Config (All Phases)
| Package | Install | Why |
|---------|---------|-----|
| `python-dotenv` | `pip install python-dotenv` | Loads API keys from the `.env` file (Alpaca, FRED, Discord, etc.) |
| `pydantic-settings` | `pip install pydantic-settings` | Validates and manages all settings in `config/settings.py` |
| `pydantic` | `pip install pydantic` | Data validation library used by pydantic-settings |

### Development & Testing (Optional)
| Package | Install | Why |
|---------|---------|-----|
| `pytest` | `pip install pytest` | Runs the test suite (`python -m pytest tests/`) |
| `pytest-cov` | `pip install pytest-cov` | Test coverage reports |
| `ruff` | `pip install ruff` | Fast Python linter to catch errors and enforce style |

---

## API Keys Required

Create a `.env` file in the project root with the following (fill in your actual keys):

```
FRED_API_KEY=your_fred_key_here
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here
```

- **FRED API key** — free at https://fred.stlouisfed.org/docs/api/api_key.html
- **Alpaca API key** — free paper trading account at 

Discord webhook URL is optional — only needed if you want Discord alert notifications.

---

## Phase Build Order

Install only what you need for the phase you're currently running:

| Phase | Packages Needed |
|-------|----------------|
| 1 — Data Scraper | yfinance, fredapi, pyarrow, pandas, numpy, openpyxl, python-dotenv, pydantic-settings, pydantic |
| 2 — Forecasting | + statsmodels, scikit-learn |
| 3 — Indicators | + ta |
| 4 — ML Engine | + torch (GPU), xgboost, lightgbm, optuna, shap |
| 5 — News & Sentiment | + requests, feedparser, beautifulsoup4, transformers, sentencepiece |
| 6 — Political/Insider | no new packages |
| 7 — Options/Calendar | no new packages |
| 8 — Ranking + Dashboard | + streamlit, plotly (already listed above) |
| 9 — Backtesting | no new packages |
| 10 — Trading | + alpaca-trade-api |
| 11 — Alerts & Reports | + discord-webhook, fpdf2, python-docx |
| 12 — Automation | + apscheduler |
