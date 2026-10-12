# BUILD STATUS

Status map of what is built and in what state. Not a duplicate of ARCHITECTURE.md or DOCUMENTATION.md. Statuses come only from the table in CLAUDE.md. The main session updates this file from the STATUS lines agents report.

Every module starts as Unverified. The architect checks the real files before planning around any Unverified module. A module is one row: either a single file or a package folder.

## Data layer

| Module | Files | Status | Note |
|---|---|---|---|
| config | config/settings.py | Needs Review | auto_rank_enabled added; invalid AUTO_RANK_ENABLED now coerces to False with a warning (never raises) |
| scraper | src/scraper/ (dividend_scraper, live_quotes, macro_scraper, price_scraper, storage) | Unverified | |
| calendar | src/calendar/ (earnings, economic) | Unverified | |
| options_data | src/options/ (implied_vol, options_data) | Unverified | |
| utils | src/utils/ (dates, input_sanitize, logging, tickers, validation) | Unverified | |

## Analysis

| Module | Files | Status | Note |
|---|---|---|---|
| indicators | src/indicators/ (base, bollinger, _calc, correlation, fundamentals, macd, momentum, moving_averages, rsi, signal_aggregator, stochastic, volume) | Unverified | |
| analytics | src/analytics/ (attribution, correlation, market_regime, peer_comparison, portfolio_metrics, sector_analysis) | Unverified | |
| ranking | src/ranking/ranker.py | Reviewed | NaN/unparseable rank rows skipped; ordering of valid entries preserved (reviewer 2026-10-11) |
| watchlist | src/watchlist/watchlist.py | Unverified | |
| alerts | src/alerts/ (notifier, triggers) | Unverified | |
| reports | src/reports/morning_report.py | Unverified | |

## Forecasting and ML

| Module | Files | Status | Note |
|---|---|---|---|
| forecasting | src/forecasting/ (auto_best, base, decomposition, exponential_smoothing, metrics, moving_average, regression, runner) | Broken | SES/Holt init fix verified. Separate gap: NaN/inf input gives silent NaN forecasts; runner.best_result can return a NaN-RMSE winner (low severity, predates fix) |
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
| dashboard_core | dashboard/app.py, dashboard/components/ (charts, market_clock, session_cache, tables, ticker_selector) | Planned | Existing files checked on disk 2026-10-11; app.py pipeline-button edits planned in plans/auto_rank_background.md |
| pipeline_runner | src/pipeline_runner/ (__init__, steps, phase1_store, run_state, launcher), cli/pipeline_job.py, cli/pipeline.py (--skip-phase1 + Phase 1 artifact only), dashboard/components/pipeline_progress.py | Reviewed | Launcher spawn_error fix re-checked PASS 2026-10-11 (647 passed, 1 xfailed). Awaiting security-auditor. Known unfixed: wrapper SIGKILL orphans pipeline child (strict xfail); residual gaps A and C. |
| pages_1_to_9 | dashboard/pages/ 1_data_overview, 2_morning_report, 2a_stock_summary, 3_long_term_forecast, 4_short_term_signals, 5_ml_predictions, 6_stock_rankings, 7_news_sentiment, 8_political_insider, 9_options_flow | Reviewed | Files checked on disk; add progress-bar call to all, cache-key change on page 6 (plans/auto_rank_background.md); page 6 _compute_rankings UnboundLocalError on pd in no-cache return (data-flow-checker 2026-10-11) |
| pages_10_to_16 | dashboard/pages/ 10_portfolio, 11_sector_analysis, 12_market_regime, 13_backtesting, 14_calendar, 15_watchlist, 16_peer_comparison | Planned | Files checked on disk; add progress-bar call to all, cache-key change on page 11 (plans/auto_rank_background.md) |
| pages_17_to_20 | dashboard/pages/ 17_trading, 18_trade_journal, 19_alerts, 20_auto_trades | Planned | Files checked on disk; add progress-bar call only (plans/auto_rank_background.md) |

## Change Log

| Date | Agent | Note |
|---|---|---|
| 2026-10-02 | setup | Created BUILD_STATUS.md with every module set to Unverified |
| 2026-10-04 | debugger | forecasting/exponential_smoothing: fixed SES/Holt TypeError on statsmodels 0.15 (explicit initialization_method); Needs Review |
| 2026-10-04 | reviewer | forecasting/exponential_smoothing: PASS; explicit initialization_method="estimated" correct and minimal, 97 tests pass; Reviewed |
| 2026-10-04 | tester | forecasting: SES/Holt fix verified equivalent to statsmodels; added 36 edge tests (suite 133 passed); found NaN/inf input gap in wrappers and runner.best_result, plus predict(0) leaking a statsmodels error; Broken |
| 2026-10-11 | architect | plans/auto_rank_background.md (daily background full-pipeline auto-run + floating progress bar); pipeline_runner added, dashboard_core and pages_* set to Planned |
| 2026-10-11 | architect | plans/auto_rank_background.md revised (section 13 answered: Phase-1 fallback, Phase-1 reuse via cli/pipeline.py --skip-phase1, AUTO_RANK_ENABLED, page 6 Re-run); config set to Planned, pipeline_runner row now includes phase1_store and the cli/pipeline.py edit |
| 2026-10-11 | coder | pyproject.toml: streamlit pin bumped to >=1.37.0 (plan task 1); Needs Review (no module row) |
| 2026-10-11 | coder | config/settings.py + .env.example: AUTO_RANK_ENABLED added (plan task 2); Needs Review. Invalid value still raises, follow-up fix queued |
| 2026-10-11 | coder | src/pipeline_runner/__init__.py + steps.py created (plan task 3); Needs Review. RunMode/RunState aliases defined locally in steps.py pending run_state.py |
| 2026-10-11 | coder | config/settings.py: invalid AUTO_RANK_ENABLED now coerces to False + warning instead of raising; Needs Review |
| 2026-10-11 | coder | src/pipeline_runner/run_state.py created (plan task 4); Needs Review. decide_auto_action lives in launcher.py per plan 4.4 |
| 2026-10-11 | coder | src/pipeline_runner/phase1_store.py created (plan task 5); Needs Review. Has private _write_json_atomic; swap for run_state.write_json_atomic at review |
| 2026-10-11 | coder | cli/pipeline.py: Phase 1 result saved after Phase 1, --skip-phase1 added (plan task 6); Needs Review. Checked with stubs only |
| 2026-10-11 | coder | src/pipeline_runner/launcher.py created (plan task 7); Needs Review. Verified with fake spawn only; Phase-1 reuse flag and lock re-check not exercised |
| 2026-10-11 | coder | cli/pipeline_job.py created (plan task 8); Needs Review. Fake-child checks only; midnight rollover, lock-release ordering and Windows untested; 'Run failed' label invented by coder |
| 2026-10-11 | coder | dashboard/components/pipeline_progress.py created (plan task 9); Needs Review. AppTest smoke only, no browser check; _bar_html has extra fallback_pending kwarg |
| 2026-10-11 | coder | dashboard/app.py: buttons routed through launcher, floating bar, AUTO_RANK_ENABLED row, old in-process runner removed (plan task 10); Needs Review. AppTest smoke with stubs |
| 2026-10-11 | coder | dashboard/pages (19 pages): render_pipeline_progress call added (plan task 12); Needs Review. py_compile only |
| 2026-10-11 | coder | dashboard/pages/6_stock_rankings.py + 11_sector_analysis.py: bar call, ranking-cache mtime cache key, Re-run starts phase1 via launcher (plan task 11); Needs Review. Re-run button branch never exercised; page 11 still has live top_picks() fallback when no cache |
| 2026-10-11 | reviewer | auto_rank_background: steps, run_state, launcher, pipeline_job, __init__, cli/pipeline.py, app.py, pages (all 21), config, .env.example, pyproject, triggers.py Reviewed; once-per-day guarantee traced and holds. FAIL on phase1_store (duplicate _write_json_atomic -> Broken until swapped) and pipeline_progress docstring placement (Needs Review); recommends removing page 11 live top_picks fallback |
| 2026-10-11 | coder | phase1_store.py now uses run_state.write_json_atomic (private writer removed); pipeline_progress.py docstring moved to top; both Needs Review, awaiting reviewer re-check |
| 2026-10-11 | coder | dashboard/pages/11_sector_analysis.py: removed implicit live top_picks() fallback; empty cache shows info + stop; Needs Review |
| 2026-10-11 | reviewer | re-check PASS: phase1_store.py, pipeline_progress.py, 11_sector_analysis.py Reviewed |
| 2026-10-11 | data-flow-checker | auto_rank_background: all 6 routes PASS (argv, markers 9/8/5, phase1 artifact, cache path, status->bar, AUTO_RANK_ENABLED). Found pre-existing: ranker.py saves failed tickers as rank 0 / -inf and load_ranking_cache sorts them first (src/ranking/ranker.py Broken); page 6 _compute_rankings local `import pandas as pd` makes pd unbound on no-cache return (6_stock_rankings.py Broken). Minor race: launcher pid fill can overwrite a very fast wrapper's fresher status (self-heals) |

| 2026-10-11 | debugger | ranker.py: load_ranking_cache skips -inf/rank-0 entries; 6_stock_rankings.py: removed local pandas/get_settings imports so no-cache path shows info instead of UnboundLocalError; both Needs Review, repro scripts only (no pytest) |
| 2026-10-11 | reviewer | ranker.py and 6_stock_rankings.py fixes PASS: Reviewed. Only behaviour change: all-failed cache now makes cli/trade.py run the live ranker instead of using junk entries |
| 2026-10-11 | coder | tests/test_pipeline_runner: 505 tests written (plan task 13), all pass when run with scratchpad-installed pytest (venv still lacks pytest); no bugs found; Needs Review |
| 2026-10-11 | tester | auto_rank_background: 505 existing pass; added 141 edge tests (130 pass, 11 xfail strict documenting 9 defects). Residual gaps A/B/C each can cause one extra auto full run (B repeatedly); pid-fill race never causes a second run. pipeline_runner, ranking, page 6 set Broken |
| 2026-10-11 | coder | Tester fixes: run_state.py, launcher.py, steps.py, cli/pipeline_job.py, ranker.py, pages/6_stock_rankings.py, app.py; 9 xfails resolved (778 passed, 1 xfailed = wrapper-SIGKILL orphan, unfixed); wrapper-kill tests leaked fake child processes, cleanup fix in progress; all Needs Review |
| 2026-10-11 | reviewer | post-tester fixes: all PASS except launcher.py:238 (OSError from try_acquire_lock not converted to spawn_error on read-only data dir; regression from tolerant get_run_paths); launcher Broken |
| 2026-10-11 | reviewer | launcher.py try_acquire_lock OSError -> spawn_error re-check PASS; pipeline_runner Reviewed (note: new tests do not assert the message text) |
| 2026-10-11 | security-auditor | auto_rank_background change set: PASS (gitleaks history + tree clean, bandit -ll clean, no shell=True, run_id validated, HTML escaped, data/ and logs/ ignored). Found literal private string in plans/auto_rank_background.md (scrubbed by main session). Pre-existing, not in this change: a private drive path in launch_dashboard.bat, launch_scheduler.bat, setup_scheduler_autostart.bat, and scripts/run_trade.ps1, plus a Windows user-profile path in scripts/run_trade.ps1; also review tracked 'CODE Design.pdf' before pushing. No push done |
