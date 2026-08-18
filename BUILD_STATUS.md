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
| config/settings.py | Done | |
| src/utils/ (logging, dates, validation) | Done | plus input_sanitize.py, tickers.py beyond original plan |
| src/scraper/price_scraper.py | Done | |
| src/scraper/macro_scraper.py | Done | |
| src/scraper/dividend_scraper.py | Done | |
| src/scraper/storage.py | Done | |
| cli/scrape.py | Done | |

## Phase 2: Long-Term Forecasting Engine
| Module | Status | Notes |
|---|---|---|
| src/forecasting/base.py | Done | |
| src/forecasting/metrics.py | Done | |
| decomposition.py (Methods 1-5) | Done | |
| moving_average.py (Methods 6-7) | Done | |
| exponential_smoothing.py (Methods 8-10) | Done | |
| regression.py (Method 11) | Done | |
| auto_best.py (Method 12) | Done | |
| runner.py + cli/forecast.py | Done | |

## Phase 3: Short-Term Technical Indicators
| Module | Status | Notes |
|---|---|---|
| src/indicators/base.py | Done | |
| RSI, MACD, Bollinger, Stochastic, EMA/SMA, Volume, Momentum, Fundamentals | Done | moving_averages.py covers EMA/SMA |
| correlation.py | Done | |
| signal_aggregator.py + cli/indicators.py | Done | |

## Phase 4: ML Forecasting Engine
| Module | Status | Notes |
|---|---|---|
| src/ml/base.py | Done | |
| feature_engineer.py | Done | |
| walk_forward.py | Done | |
| xgboost_model.py, random_forest.py, linear_models.py | Done | plus catboost_model.py, ensemble_model.py beyond original plan |
| lstm_model.py, transformer_model.py (GPU) | Done | GPU availability not verified in this pass (no code execution) |
| tuner.py (Optuna) | Done | |
| feature_selection.py (SHAP) | Done | |
| runner.py + cli/ml.py | Done | plus src/ml/timing.py beyond original plan |

## Phase 5: News & Short Interest
| Module | Status | Notes |
|---|---|---|
| news_fetcher.py | Done | implemented as src/news/scraper.py |
| short_interest.py | Done | |
| sentiment.py | Done | |
| cli/news.py | Done | plus src/news/aggregator.py, src/news/runner.py beyond original plan |

## Phase 6: Political & Insider Trading
| Module | Status | Notes |
|---|---|---|
| congress_tracker.py | Done | |
| insider_tracker.py | Done | |
| ranker integration (congressional signal) | Done | confirmed wired: src/ranking/ranker.py imports get_congress_signal/get_insider_signal, weight 1.0 each |

## Phase 7: Options & Calendar Data
| Module | Status | Notes |
|---|---|---|
| options_data.py | Done | |
| implied_vol.py | Done | |
| calendar/earnings.py | Done | |
| calendar/economic.py | Done | |

## Phase 8: Ranking + Streamlit Dashboard
| Module | Status | Notes |
|---|---|---|
| src/ranking/ranker.py | Done | |
| dashboard/app.py | Done | serves as Settings/API-key landing page; confirmed .env write path is gitignored |
| dashboard pages | Done | 23 pages exist (1, 1_data_overview, 2, 2a, 3-20), more than the originally planned 17 — doc count was stale, not a bug |
| components/charts.py, tables.py | Done | plus market_clock.py, session_cache.py, ticker_selector.py, research_panel.py beyond original plan |

## Phase 9: Backtesting & Portfolio Analytics
| Module | Status | Notes |
|---|---|---|
| backtester.py (slippage/spread modeling) | Done | |
| Full paper-trading simulation (2015-2020 train / 2020-2026 test) | Done | data/backtest_results/ contains real equity_curve, drawdown_curve, trade_log, metrics.json — sim has actually been run |
| portfolio_metrics.py | Done | |
| sector_analysis.py | Done | |
| peer_comparison.py | Done | |
| market_regime.py | Done | |
| analytics/correlation.py | Done | |
| attribution.py | Done | |
| watchlist.py | Done | |
| cli/backtest.py | Done | |

## Phase 10: Trading Integration
| Module | Status | Notes |
|---|---|---|
| alpaca_client.py | Done | TRADING_MODE=PAPER, ALPACA_BASE_URL points at paper-api — confirmed in env, not live |
| strategy.py | Done | |
| multi_strategy.py | Needs Review | Fixed: `_is_mean_rev` stub (always False, so mean-reversion exits never fired) replaced with `_mean_rev_tickers()`, backed by new `TradeJournal.latest_open_strategy()`. Logic traced and syntax-checked; could not run pytest in this environment (no package access in this WSL session — project runs on Windows Python) so status stays Needs Review pending an actual test run, not Done. New tests added at tests/test_trading/test_multi_strategy.py, unrun. |
| circuit_breaker.py | Done | confirmed wired into cli/trade.py (CircuitBreaker/CircuitBreakerTripped imported and used at 4 call sites) |
| portfolio.py | Done | |
| tax_lots.py | Done | |
| trade_journal.py | Done | |
| risk.py | Done | |
| cli/trade.py | Done | |

## Phase 11: Alerts & Morning Report
| Module | Status | Notes |
|---|---|---|
| notifier.py (email/Discord/Telegram) | Done | |
| morning_report.py | Done | |
| cli/report.py | Done | |

## Phase 12: Research & Supply Chain (uncommitted)
Not in the original plan doc. Fully-built, working-tree-only (untracked in
git as of this audit) attempt at the FUTURE_FEATURES.md "Supply Chain /
Company Relationship Map" item plus a new LLM-backed research feature.
| Module | Status | Notes |
|---|---|---|
| src/research/ (ollama_client, search_client, page_fetch, cache, sources, entity_resolve, research_store, runner) | Done | ollama_client.py fails with a clear error ("Start Ollama or set OLLAMA_BASE_URL") if the local Ollama service isn't running — external runtime dependency, not a code defect |
| src/analytics/supply_chain.py, supply_chain_seed.py | Done | uses networkx (present in both pyproject.toml and requirements.txt) |
| src/analytics/company_meta.py | Done | confirmed used by supply_chain.py, entity_resolve.py, supply_chain_agent.py, and dashboard/pages/21_supply_chain.py |
| dashboard/pages/21_supply_chain.py, 22_deep_research.py | Done | |
| dashboard/components/research_panel.py | Done | confirmed referenced from charts.py and the two new pages |
| cli/research.py | Done | |
| Docker setup (Dockerfile, docker-compose.yml, .dockerignore, requirements.txt) | Done | Dockerfile correctly COPYs and installs requirements.txt |
| **Action needed** | — | none of Phase 12 is committed to git; still sitting as untracked/modified working-tree files as of this audit |

## Phase 13: Extensions beyond original plan (organic, not in recursive-gliding-dove.md)
Built during real usage, confirmed to exist and be wired in, not individually
deep-audited beyond existence + call-site confirmation.
| Module | Status | Notes |
|---|---|---|
| src/trading/stop_manager.py | Done | confirmed wired into cli/scheduler.py and cli/trade.py — this is the trailing-stop-order fix referenced in recent commit history |
| src/trading/options_strategy.py | Done | confirmed wired into cli/trade.py and dashboard/pages/17_trading.py |
| src/scraper/live_quotes.py | Done | confirmed wired into cli/scheduler.py, dashboard/pages/15_watchlist.py, dashboard/pages/2a_stock_summary.py |
| src/alerts/triggers.py | Done | |
| cli/pipeline.py, cli/rank.py | Done | |
| dashboard/pages/1_data_overview.py, 10_portfolio.py, 20_auto_trades.py | Done | |

---

## Known gaps outside the phase tables
- CLAUDE.md does not exist in this repo, despite every agent definition in
  .claude/agents/ referencing a "stop-and-report rule in CLAUDE.md" and
  preflight.md listing it as a required project file. Agents are currently
  relying on an inferred version of that rule (log out-of-scope findings
  here, don't fix them inline). Recommend creating CLAUDE.md with that rule
  written down explicitly.

## Change Log
Newest entries at top. One line per architect session: date, what was
audited or built, what changed.

<!-- new entries go below this line -->
- 2026-08-17 (coder): Fixed multi_strategy.py's `_is_mean_rev` TODO — it always returned False, so mean-reversion exit signals never fired for any position. Added `TradeJournal.latest_open_strategy()` (most-recent-BUY-per-ticker lookup) and `MultiStrategyManager._mean_rev_tickers()`, computed once per generate_all_signals() call instead of the old per-position calls. Added tests/test_trading/test_multi_strategy.py (7 cases: empty journal, open position, closed position omitted, re-buy after close, manager lookup, empty input, journal-read-failure fallback). Could not execute pytest in this WSL session (no pandas/pip/venv access without sudo) — verified by syntax check and manual trace only, left as Needs Review rather than Done pending a real test run.
- 2026-08-17 (architect): First real audit since this file was seeded as a placeholder. All 11 original phases confirmed Done (files exist, non-stub, wired to their callers) — the project is substantially complete, not just planned. Only concrete issue found: src/trading/multi_strategy.py has one unfinished TODO (position-strategy tagging), marked Needs Review. Added Phase 12 to track the uncommitted research/supply-chain/Docker work (all Done, just not committed to git yet) and Phase 13 to track organic extensions (stop_manager, options_strategy, live_quotes, alerts/triggers, cli/pipeline+rank, 3 extra dashboard pages) built beyond the original plan doc but never added to this tracker. Flagged CLAUDE.md as missing despite being referenced by every agent definition. Confirmed TRADING_MODE=PAPER and ALPACA_BASE_URL point at paper-api — no live-order risk found. No code was executed as part of this audit (existence/completeness/wiring only, no correctness testing).
