# BUILD_STATUS.md

Living status tracker for AutoStockAnalyzer, mapped to the phases in
`recursive-gliding-dove.md`. Every agent (architect, coder, reviewer,
tester, security-auditor, debugger, api-integration, data-flow-checker)
updates this file when it finishes work on a module or finds a problem --
see each agent's own "Updating BUILD_STATUS.md" instructions.

**Status key:** Not Started | In Progress | Done | Needs Review | Broken

**First task for architect:** the project is fully built already — this
file was seeded from the plan doc, not from auditing the actual repo, so
every status below is a placeholder. On first use, audit the real codebase
against this list and mark everything that exists and works as "Done."
Only modules that are genuinely missing, incomplete, or broken should get
anything other than Done. From that point on, this file exists to track
FUTURE changes and known issues, not initial construction.

---

## Phase 1: Foundation
| Module | Status | Notes |
|---|---|---|
| config/settings.py | Unverified | |
| src/utils/ (logging, dates, validation) | Unverified | |
| src/scraper/price_scraper.py | Unverified | |
| src/scraper/macro_scraper.py | Unverified | |
| src/scraper/dividend_scraper.py | Unverified | |
| src/scraper/storage.py | Unverified | |
| cli/scrape.py | Unverified | |

## Phase 2: Long-Term Forecasting Engine
| Module | Status | Notes |
|---|---|---|
| src/forecasting/base.py | Unverified | |
| src/forecasting/metrics.py | Unverified | |
| decomposition.py (Methods 1-5) | Unverified | |
| moving_average.py (Methods 6-7) | Unverified | |
| exponential_smoothing.py (Methods 8-10) | Unverified | |
| regression.py (Method 11) | Unverified | |
| auto_best.py (Method 12) | Unverified | |
| runner.py + cli/forecast.py | Unverified | |

## Phase 3: Short-Term Technical Indicators
| Module | Status | Notes |
|---|---|---|
| src/indicators/base.py | Unverified | |
| RSI, MACD, Bollinger, Stochastic, EMA/SMA, Volume, Momentum, Fundamentals | Unverified | |
| correlation.py | Unverified | |
| signal_aggregator.py + cli/indicators.py | Unverified | |

## Phase 4: ML Forecasting Engine
| Module | Status | Notes |
|---|---|---|
| src/ml/base.py | Unverified | |
| feature_engineer.py | Unverified | |
| walk_forward.py | Unverified | |
| xgboost_model.py, random_forest.py, linear_models.py | Unverified | |
| lstm_model.py, transformer_model.py (GPU) | Unverified | |
| tuner.py (Optuna) | Unverified | |
| feature_selection.py (SHAP) | Unverified | |
| runner.py + cli/ml.py | Unverified | |

## Phase 5: News & Short Interest
| Module | Status | Notes |
|---|---|---|
| news_fetcher.py | Unverified | |
| short_interest.py | Unverified | |
| sentiment.py | Unverified | |
| cli/news.py | Unverified | |

## Phase 6: Political & Insider Trading
| Module | Status | Notes |
|---|---|---|
| congress_tracker.py | Unverified | |
| insider_tracker.py | Unverified | |
| ranker integration (congressional signal) | Unverified | |

## Phase 7: Options & Calendar Data
| Module | Status | Notes |
|---|---|---|
| options_data.py | Unverified | |
| implied_vol.py | Unverified | |
| calendar/earnings.py | Unverified | |
| calendar/economic.py | Unverified | |

## Phase 8: Ranking + Streamlit Dashboard
| Module | Status | Notes |
|---|---|---|
| src/ranking/ranker.py | Unverified | |
| dashboard/app.py | Unverified | |
| 17 dashboard pages (1-17) | Unverified | list which pages exist once audited |
| components/charts.py, tables.py | Unverified | |

## Phase 9: Backtesting & Portfolio Analytics
| Module | Status | Notes |
|---|---|---|
| backtester.py (slippage/spread modeling) | Unverified | |
| Full paper-trading simulation (2015-2020 train / 2020-2026 test) | Unverified | |
| portfolio_metrics.py | Unverified | |
| sector_analysis.py | Unverified | |
| peer_comparison.py | Unverified | |
| market_regime.py | Unverified | |
| analytics/correlation.py | Unverified | |
| attribution.py | Unverified | |
| watchlist.py | Unverified | |
| cli/backtest.py | Unverified | |

## Phase 10: Trading Integration
| Module | Status | Notes |
|---|---|---|
| alpaca_client.py | Unverified | |
| strategy.py | Unverified | |
| multi_strategy.py | Unverified | |
| circuit_breaker.py | Unverified | |
| portfolio.py | Unverified | |
| tax_lots.py | Unverified | |
| trade_journal.py | Unverified | |
| risk.py | Unverified | |
| cli/trade.py | Unverified | |

## Phase 11: Alerts & Morning Report
| Module | Status | Notes |
|---|---|---|
| notifier.py (email/Discord/Telegram) | Unverified | |
| morning_report.py | Unverified | |
| cli/report.py | Unverified | |

---

## Change Log
Newest entries at top. One line per architect session: date, what was
audited or built, what changed.

<!-- new entries go below this line -->
