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
| config/settings.py | Done | perf plan adds 4 additive tunables (scrape_max_workers, scrape_io_workers, news_max_workers, fred_max_workers) |
| src/utils/ (logging, dates, validation) | Done | plus input_sanitize.py, tickers.py beyond original plan |
| src/utils/parallel.py | Done | NEW — shared bounded ThreadPoolExecutor helper `thread_map()`; prerequisite for all scraper perf work. Order-preserving, per-item failure isolation (None + WARNING log via setup_logger), max_workers<=1 runs inline, stagger sleeps between submits/calls. Not yet imported by any scraper file. |
| src/scraper/_yf.py | Done | NEW — `yf_download()` + `extract_ticker_frame()`. Global semaphore sized to `scrape_max_workers` caps concurrent batch calls; `threads=yf_inner_threads` (int, not bare `True`) caps yfinance's own internal pool; `timeout=30` added; `YFRateLimitError` gets 30*(attempt+1)s backoff, other exceptions get 5*(attempt+1)s (matches old `download_batch` behavior). Sanity-checked live against yfinance 1.2.0 (`AAPL`/`MSFT`, Jan 2024) — non-empty batch frame, correct title-cased single-ticker extraction. `price_scraper.py` not yet updated to import from here — that's the next task. |
| src/scraper/storage.py | Done | Perf plan 2026-08-18: `get_latest_date()` now reads pyarrow row-group statistics for the index column (falls back to `pd.read_parquet(columns=[])` index-only read if stats/index-name unresolvable or unnamed `__index_level_0__`); new `get_latest_dates()` batches it via `thread_map`; `save_dataframe`/`upsert_dataframe` now serialize per-file via `_file_lock` (module-level dict of `threading.Lock` keyed by resolved path) and write atomically (`.tmp` + `os.replace`, cleaned up in `finally`); `upsert_dataframe` skips dedup/sort when already clean (`has_duplicates`/`is_monotonic_increasing` guards). To avoid deadlock, `upsert_dataframe` holds the lock for its whole read-modify-write and calls a private `_write_parquet_atomic()` directly instead of the public `save_dataframe()` (which also locks) — same lock object, so save/upsert on the same path still serialize with each other, just not nested. Public signatures/return types unchanged; all ~20 callers (verified via grep) only use the unchanged public functions. Sanity-checked with a throwaway `python.exe` script (save/load/get_latest_date incl. unnamed-index fallback path, upsert merge+dedup+sort+empty-short-circuit, get_latest_dates over existing+missing paths) — all passed, script deleted. `python.exe -c "from src.scraper import storage"` imports cleanly. |
| src/scraper/price_scraper.py | Done | Rewired 2026-08-18 to delegate to `_yf.yf_download()`/`extract_ticker_frame()` (removed the now-duplicate `_extract_ticker_from_multiindex`, confirmed no other importers via grep). `download_batch()` signature/return contract and log lines unchanged. `scrape_prices()` gained `max_workers` (→ `settings.scrape_max_workers`), replaced the per-ticker `get_latest_date()` loop with one `storage.get_latest_dates()` call, and now runs backfill+incremental `(batch, start)` units through `thread_map()` instead of sequential loops — same `{ticker: rows}` return contract, merged from per-unit results. `aggregate_to_quarterly()` gained `max_workers` (→ `settings.scrape_io_workers`) and `force` (bypasses a new skip check: if quarterly parquet exists and its mtime >= the daily parquet's mtime, that ticker is skipped entirely); non-skipped tickers now read only `["Open","High","Low","Close","Volume"]` columns and run through `thread_map()`. The `resample("QE")` aggregation spec (mean O/H/L/C, sum Volume) is byte-identical to before — confirmed by reading, not touched. `cli/scrape.py` intentionally NOT touched (its `--workers` passthrough is a separate follow-up task) — note its BUILD_STATUS row already says "Done" for that passthrough but the flag does not exist yet in the file as of this edit; flagging the mismatch here rather than fixing, per stop-and-report (out of scope for this task). Verified `python.exe -c "from src.scraper import price_scraper"` imports cleanly, and ran a live throwaway `python.exe` sanity script outside the repo (`J:\Coding\tmp_sanity_price_scraper.py`, deleted after): `scrape_prices(["AAPL","MSFT"], backfill=True)` → 2922 rows each written, parquet files created; second incremental call completed in 0.16s with 0 new rows (both tickers already current — start==end, yfinance correctly reports 0 new rows, not an error); `aggregate_to_quarterly()` → 47 quarterly rows/ticker, `High >= Low` and `Volume > 0` held for all rows; a repeat call with `force=False` was a no-op (0.01s, mtime skip fired); `force=True` re-aggregated both tickers. |
| src/scraper/macro_scraper.py | Done | Perf plan implemented: `_get_fred_client()` is now a lock-guarded singleton (double-checked, module-level `_FRED_CLIENT`/`_FRED_CLIENT_LOCK`); `download_fred_series()` takes an optional `client=` param, falling back to the singleton; `scrape_macro()` builds `(series_id, start)` pairs, fetches the singleton client once, runs downloads through `thread_map(..., max_workers=settings.fred_max_workers)`, then upserts serially in the main thread. Signatures/return contracts and the `ValueError` raised by `_get_fred_client()` when no API key is configured are all unchanged (verified `cli/scrape.py` still catches it correctly). Live-verified with a real FRED key: `scrape_macro(series_ids=["FEDFUNDS","VIXCLS"], backfill=True)` against a temp dir produced correct single-column parquets (index named "Date") with sane values, and singleton identity (`_get_fred_client() is _get_fred_client()`) was confirmed True. Unrelated pre-existing issue still open, not fixed, logged per stop-and-report: `logger.error(f"Failed to download FRED series '{series_id}': {e}")` logs the raw exception from `fredapi`, which builds request URLs with `api_key` as a query param — if the underlying `requests` exception embeds the URL, the FRED key could leak into logs. Needs a sanitized error message, not a raw `{e}` log. |
| src/scraper/dividend_scraper.py | Done | Rewired 2026-08-18 to use `_yf.yf_download()`/`extract_ticker_frame()` instead of 503 sequential `yf.Ticker().dividends` calls. New `download_dividends_batch(tickers, start)` does one batched `yf_download(..., actions=True)` per `yfinance_batch_size` chunk, extracts each ticker's `Dividends` column, filters the dense 0.0-filled column down to nonzero/non-NaN rows (confirmed empirically: AAPL/MSFT since 2015 → 47/46 sparse rows, not ~2900 daily rows), and tz-normalizes the result. Chose to `tz_localize`/`tz_convert` the batch result to `America/New_York` (not strip tz) because existing stored parquet files (confirmed by reading `data/raw/dividends/AAPL.parquet`) are tz-aware `America/New_York` from the old `yf.Ticker().dividends` path — localizing new data to match keeps `upsert_dataframe`'s index merge/dedup working with zero migration needed. `download_dividends(ticker, start)` is now a thin wrapper (`download_dividends_batch([ticker], start).get(ticker, pd.DataFrame())`) — signature/return contract unchanged. `scrape_dividends()` gained `max_workers` (→ `settings.scrape_max_workers`), replaced the per-ticker loop with one `storage.get_latest_dates()` call grouped by start-date (same shape as price_scraper's incremental grouping), chunked by `yfinance_batch_size`, run through `thread_map(..., label="dividend-batch")`; same `{ticker: rows}` return contract and same final INFO log line. `get_dividend_yield()` untouched per instructions. Verified `python.exe -c "from src.scraper import dividend_scraper"` imports cleanly; live throwaway sanity script (`python.exe`, deleted after): `download_dividends_batch(["AAPL","MSFT"], start="2024-01-01")` → 11/10 sparse rows matching known 2024-2026 AAPL/MSFT ex-div dates and amounts (e.g. AAPL 2024-02-09 $0.24, MSFT 2024-02-14 $0.75), index tz confirmed `America/New_York`; `scrape_dividends(["AAPL","MSFT"], data_dir=<tmp>, backfill=True)` wrote 47/46-row parquet files with `(Dividends != 0).all()` holding. Pre-existing open question carried forward unresolved (not this task's scope to fix): no-dividend tickers still have no stored file, so `get_latest_dates` returns `None` for them and they re-download from `backfill_start_year` on every incremental run — same behavior as the old code, just noting it's still unresolved. |
| src/scraper/live_quotes.py | Done | Perf plan implemented 2026-08-18: `_build_data_client()` caches `StockHistoricalDataClient` in a module-level dict keyed by `(api_key, secret_key)` guarded by `threading.Lock` (same None-on-failure contract preserved); `_prev_close()` reads only `columns=["Close"]` and is mtime-cached (`dict[ticker] -> (st_mtime_ns, value)`, invalidated when the file's mtime changes); new `_prev_closes()` batches it via `thread_map(..., max_workers=settings.scrape_io_workers)`; `get_live_quotes()` now calls `_prev_closes()` once per request instead of `_prev_close()` per ticker in the loop; `refresh_quote_cache()`'s `quotes.json` write is now atomic (`.tmp` + `os.replace`) to fix torn reads by the dashboard/watchlist page during the scheduler's 5-min refresh. All 5 public signatures/return shapes and the JSON cache shape unchanged (verified against `dashboard/pages/15_watchlist.py` and `2a_stock_summary.py` callers). `get_tracked_tickers`'s Alpaca `list_positions()` call untouched, as instructed. `python.exe -c "from src.scraper import live_quotes"` imports cleanly; live throwaway `python.exe` sanity script (deleted after) confirmed: correct prev-close values from fake parquet files, cache hit on unchanged mtime, recompute + cache update after mtime change, correct `_prev_closes()` batched dict incl. missing-ticker→None, Alpaca client identity reused across two `_build_data_client()` calls, and atomic-write logic (tmp file written, `os.replace`'d, tmp cleaned up, `load_quote_cache()` reads the result correctly). |
| cli/scrape.py | Done | 2026-08-18: Added `--workers` (`type=int, default=None`) CLI flag, passed through as `max_workers=args.workers` to `scrape_prices()`, `aggregate_to_quarterly()`, and `scrape_dividends()`; `scrape_macro()` call left untouched (macro scraper deliberately has no `max_workers` param). No other flags/structure/logging changed. `python.exe -m py_compile` and `--help` both verified. |
| tests/test_scraper/ | Done | Tester pass 2026-08-18: added `test_storage.py` (17 tests) and `test_price_scraper.py` (17 tests), all passing, no production bugs found. Covers fast/fallback `get_latest_date` (named index, unnamed `__index_level_0__`, missing, empty), batch `get_latest_dates`, upsert dedup semantics (confirmed newest/new-data wins on overlap), the has_duplicates/is_monotonic_increasing slow-path firing correctly on genuinely dirty merged data, atomic-write tmp cleanup on both serialize and os.replace failure, 12-thread concurrent-upsert-same-file safety (no lost rows, no leftover .tmp), both yfinance MultiIndex column orderings via `_yf.extract_ticker_frame`, `scrape_prices` batch-grouping (backfill vs incremental vs already-up-to-date skip vs shared-batch grouping), `aggregate_to_quarterly` mtime skip/recompute/force logic, and numeric correctness of the quarterly resample (mean OHLC, sum Volume). Full suite: 150/150 passed (`python.exe -m pytest tests/ -q`). Real timing measured live against production data/scraper (503 S&P 500 tickers, true steady-state incremental — see Change Log): default `scrape_max_workers=3` ≈2.8–3.0s vs `--workers 1` (serial) ≈8.7–9.8s, ~3.2x speedup, consistent with the 3x worker count. |

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
| news_fetcher.py | Done | implemented as src/news/scraper.py. Perf plan done 2026-08-18: `_fetch_feed()` now uses `requests` (per-thread `Session` via `_get_session()`, explicit `timeout=10`, `User-Agent` header) instead of bare untimed `feedparser.parse(url)`; `seen_urls` param removed (dedup moved to caller, after both feeds return, to avoid shared-set mutation from pool threads); `fetch_ticker_news()` fetches Yahoo+Google concurrently via `thread_map(..., max_workers=2)` then dedupes Yahoo-before-Google; `fetch_batch_news()` gained `max_workers` (→ `settings.news_max_workers`) and replaced the serial per-ticker `time.sleep(delay)` loop with `thread_map(..., stagger=delay/max_workers)` — `delay`'s meaning changed from "hard serialization sleep" to "submission stagger on a bounded pool" (documented in the docstring; use `max_workers=1` for the old guarantee). `src/news/runner.py` still calls `fetch_batch_news` serially per ticker in its own loop — untouched, out of scope, already logged below. |
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
- ~~CLAUDE.md does not exist in this repo~~ — resolved; CLAUDE.md now exists
  with the stop-and-report rule and the "use Windows python.exe" rule.
- `config/settings.py::get_settings()` constructs a fresh `Settings()` on every
  call, which re-reads and re-parses `.env` from disk each time. It is called
  from inside per-ticker loops across the codebase (scrapers, ranker, dashboard
  pages). An `@lru_cache` would fix it globally but would change behaviour for
  anything relying on picking up mid-process `.env` edits (the dashboard
  Settings page writes `.env`). Out of scope for the scraper perf work — the
  scraper plan only hoists `get_settings()` out of loops locally. Needs a
  deliberate decision, not a drive-by change.
- `src/news/runner.py` (`run_news_pipeline`, `run_sentiment_only`) fetches news
  strictly one ticker at a time with a `request_delay=1.0` sleep between each.
  Even after `src/news/scraper.py` is parallelised, the runner will still
  serialise everything, so most of the news-fetch win is unrealised until the
  runner prefetches articles in parallel and only serialises the FinBERT
  scoring step. Out of the scraper-perf scope; not touched.
- `src/scraper/dividend_scraper.py::get_dividend_yield()` calls `yf.Ticker().info`
  (a slow full quote-summary request) and has no callers anywhere in the repo.
  Candidate for deletion; left alone.
- Parquet writes were non-atomic everywhere (`DataFrame.to_parquet` straight
  onto the target path). A crash mid-write corrupts the file. Fixed
  2026-08-18 in `storage.py::save_dataframe`/`upsert_dataframe` (tmp file +
  `os.replace`, per-file lock), which covers most writers, but modules that
  call `df.to_parquet(...)` directly (bypassing `src/scraper/storage.py`)
  are still not covered — not audited as part of this task.

## Change Log
Newest entries at top. One line per architect session: date, what was
audited or built, what changed.

<!-- new entries go below this line -->
- 2026-08-18 (tester): Wrote `tests/test_scraper/test_storage.py` (17 tests) and `tests/test_scraper/test_price_scraper.py` (17 tests) hunting for edge cases beyond spec compliance — empty/missing files, unnamed-index fallback, overlapping-upsert dedup direction, deliberately-dirty (duplicated + unsorted) merge data to force the slow path, atomic-write tmp cleanup on both to_parquet and os.replace failure (via monkeypatch), 12-thread concurrent-upsert-same-file race test, both yfinance MultiIndex column orderings, scrape_prices batch-grouping (backfill/incremental/already-up-to-date-skip/shared-batch), aggregate_to_quarterly mtime skip/recompute/force, and numeric correctness of the quarterly resample against a hand-computed expected result. All 34 new tests pass; ran 5x in a row incl. the concurrency tests with no flakiness. Full suite `python.exe -m pytest tests/ -q` — 150/150 passed, no regressions. Found no production bugs — every edge case investigated matched correct/intended behavior. Measured real speedup live against production data: `python.exe cli/scrape.py --prices-only --no-quarterly` on the full 503-ticker S&P 500 list in true steady-state incremental mode (stored data already current through the prior trading day) — default `scrape_max_workers=3` completed in 2.8–3.0s (2 runs) vs `--workers 1` (serial) in 8.7–9.8s (2 runs), a ~3.2x speedup consistent with the 3x worker count. Caveat: this environment's system date is ahead of real market data availability, so yfinance returned "no price data found" for every ticker on every run (visible as harmless `possibly delisted` log noise) and no rows were actually written — the timing therefore reflects per-batch network round-trip/API-call overhead (exactly what the thread-pool parallelism targets) rather than payload transfer time, and the run was safely repeatable (confirmed stored `AAPL.parquet` latest date unchanged at 2026-08-17 across all 4 runs) without corrupting or advancing real production data.
- 2026-08-18 (reviewer): Reviewed the full scraper perf pass against the architect spec — `src/utils/parallel.py`, `src/scraper/_yf.py`, `config/settings.py`, `src/scraper/storage.py`, `src/scraper/price_scraper.py`, `src/scraper/dividend_scraper.py`, `src/scraper/macro_scraper.py`, `src/scraper/live_quotes.py`, `src/news/scraper.py`, `cli/scrape.py`. All contracts verified against spec (signatures, `threads=yf_inner_threads` as int not bool, `YFRateLimitError` vs generic backoff timing, dense-to-sparse dividend filtering, tz localization to `America/New_York`, FRED singleton + serial upserts, `list_positions()`/`get_dividend_yield()`/`_parse_entry()` left untouched as required, `--workers` not passed to `scrape_macro`). No closure/lambda capture bugs found (all thread_map callables close over per-call function params, not reassigned loop variables). `grep -rn "_extract_ticker_from_multiindex\|seen_urls"` — no dangling references to the removed helper/param; remaining `seen_urls` hits are the still-valid `_fetch_yfinance_news` internal dedup set, unrelated. `ast.parse` clean on all 10 files. `python.exe -m pytest tests/ -x -q` — 116 passed, 0 failed, including `tests/test_news/test_news.py` 22/22 (note: `test_returns_sorted_articles`/`test_deduplicates_by_url` now exercise a real outbound `requests.get()` before the mocked `feedparser.parse()` intercepts, so they're network-dependent in a way they weren't before — passed in this sandbox, already called out by the coder in the prior changelog entry, not a regression but worth knowing if CI ever runs offline). BUILD_STATUS.md rows for all touched files are accurate and consistent with the code as of this review. No hardcoded secrets or unsafe eval/exec/subprocess patterns found in the touched files. Verdict: PASS, no changes made.
- 2026-08-18 (coder): Implemented `src/scraper/live_quotes.py` per architect spec — cached `_build_data_client()`'s Alpaca client keyed by `(api_key, secret_key)` behind a `threading.Lock`; `_prev_close()` now reads only the `Close` column and is mtime-cached (per-ticker `(st_mtime_ns, value)` dict); new `_prev_closes()` batches it via `thread_map(..., max_workers=settings.scrape_io_workers)`; `get_live_quotes()` calls `_prev_closes()` once instead of `_prev_close()` per ticker; `refresh_quote_cache()`'s `quotes.json` write is now atomic (tmp file + `os.replace`) to stop torn reads racing with the 5-min scheduler job. All 5 public signatures/return shapes and the JSON cache shape unchanged; `get_tracked_tickers`'s Alpaca `list_positions()` call untouched per instructions. Live throwaway `python.exe` sanity script (deleted after) confirmed correctness of prev-close values, mtime cache hit/invalidation, batched `_prev_closes()`, client identity reuse, and the atomic-write + `load_quote_cache()` read-back path. No ambiguities hit — nothing needed logging beyond this entry.
- 2026-08-18 (coder): Rewired `src/scraper/dividend_scraper.py` to use `_yf.yf_download()`/`extract_ticker_frame()` instead of 503 sequential `yf.Ticker().dividends` calls — new `download_dividends_batch()`, `download_dividends()` reduced to a thin wrapper (unchanged contract), `scrape_dividends()` gained `max_workers` and now groups tickers by incremental start date (via `storage.get_latest_dates()`) and runs batches through `thread_map()`. Handled both api-integration-flagged gotchas: filtered the dense 0.0-filled `Dividends` column down to nonzero/non-NaN rows before storing, and localized the tz-naive batch index to `America/New_York` to match existing tz-aware stored parquet data (confirmed by reading an existing file, not stripping tz on the old side). Live sanity script (deleted after): AAPL/MSFT 2024 batch → 11/10 sparse rows matching known dividend dates/amounts; `scrape_dividends(backfill=True)` on a temp dir → 47/46-row parquet files, no zero rows. See dividend_scraper.py row for full detail. Pre-existing open question (no-dividend tickers re-download every incremental run) carried forward unresolved, out of scope for this task.
- 2026-08-18 (coder): Rewired `src/scraper/macro_scraper.py` per architect spec — `_get_fred_client()` is now a lock-guarded module-level singleton (double-checked pattern), `download_fred_series()` gained an optional `client=` param (falls back to the singleton), `scrape_macro()` builds `(series_id, start)` pairs first, fetches the singleton client once, fans downloads out through `thread_map(..., max_workers=settings.fred_max_workers, label="fred-series")`, then upserts serially in the main thread (only ~10 files, per architect guidance). Signatures/return contracts unchanged; confirmed `_get_fred_client()` still raises `ValueError` and `cli/scrape.py` still catches exactly that type. `python.exe -c "from src.scraper import macro_scraper"` imports cleanly. A FRED API key was present in this environment's `.env`, so did a live sanity run instead of mocking: throwaway `python.exe` script (deleted after) called `scrape_macro(series_ids=["FEDFUNDS","VIXCLS"], data_dir=<tmp>, backfill=True)` — both series downloaded via the thread pool, upserted, and read back with correct shape (single column named after series_id, index named "Date", sane values); also confirmed `_get_fred_client() is _get_fred_client()` (singleton identity). Did not touch the pre-existing FRED-key-leak-via-exception-log risk at the `logger.error(...)` line — already logged above, out of scope per stop-and-report.
- 2026-08-18 (coder): Implemented `src/scraper/storage.py` internals per architect spec — metadata-only `get_latest_date()` via pyarrow row-group statistics (with index-only-read fallback), new batched `get_latest_dates()` using `thread_map`, per-file locking (`_file_lock`) + atomic tmp/replace writes in `save_dataframe`/`upsert_dataframe`, and dedup/sort fast-path guards in `upsert_dataframe`. Public signatures/return types and the `"Upserted N new rows -> M total in <name>"` log line unchanged. Verified all ~20 existing callers only use unchanged public functions (grep). Sanity-checked with a throwaway `python.exe` script covering all new/changed behavior — passed, script deleted. See storage.py row and "non-atomic writes" gap note above for details.
- 2026-08-18 (coder): Created `src/scraper/_yf.py` per architect spec — `yf_download()` (rate-limit-aware `yf.download()` wrapper, module-level semaphore capped to `scrape_max_workers`, `threads=yf_inner_threads` int, `timeout=30`, `YFRateLimitError`-specific 30*(attempt+1)s backoff vs 5*(attempt+1)s for other exceptions) and `extract_ticker_frame()` (verbatim move of `price_scraper._extract_ticker_from_multiindex`, including the `['Ticker','Price']`/`['Price','Ticker']` level-order detection). Verified `python.exe -c "from src.scraper import _yf"` imports cleanly and ran a live throwaway sanity script (`yf_download(["AAPL","MSFT"], start="2024-01-01", end="2024-01-31")` → non-empty (20, 10) MultiIndex frame, `columns.names == ['Ticker','Price']`; `extract_ticker_frame(data, "AAPL")` → (20, 5) frame with title-cased `Open/High/Low/Close/Volume` columns) — passed, script deleted after. Create-only per instructions: `price_scraper.py` NOT edited yet, still has its own `_extract_ticker_from_multiindex`/`download_batch` — that removal/rewire is a separate follow-up task.
- 2026-08-18 (api-integration): Signed off on the two open risks from the scraper perf plan. (1) Dividends-column approach WORKS: verified empirically with `python.exe`, yfinance 1.2.0, `yf.download(["AAPL","MSFT"], actions=True, auto_adjust=True, group_by="ticker", threads=True)` over 2024 — MultiIndex columns `['Ticker','Price']` include `Dividends`/`Stock Splits` per ticker, values match `yf.Ticker().dividends` exactly on all 8 spot-checked ex-div dates/amounts, compatible with the existing `_extract_ticker_from_multiindex` helper. Caveats logged against dividend_scraper.py row: batch column is dense (0.0-filled) not sparse, and index is tz-naive vs `yf.Ticker().dividends`'s tz-aware index — both need explicit handling before merging into existing parquet files. (2) `scrape_max_workers=6` as designed is riskier than it looks: read yfinance 1.2.0 source (`multi.py`) and confirmed `threads=True` inside `yf.download()` already fans out to `min(batch_size, cpu_count()*2)` concurrent per-ticker requests on its own, uncapped by our code — so 6 concurrent batch calls could mean ~100+ simultaneous requests to Yahoo's undocumented endpoint, not 6. Recommended fix: pass an explicit small `threads=` int inside `download_batch()` and drop outer `scrape_max_workers` to 3, giving a known ~12 concurrent-request ceiling; also recommended catching `yfinance.exceptions.YFRateLimitError` (confirmed present in 1.2.0) with exponential backoff + jitter and capped retries instead of the current linear-wait bare-`Exception` retry, and adding an explicit `timeout=` to `yf.download()`. Logged full details against price_scraper.py and dividend_scraper.py rows above. In passing, flagged a pre-existing unrelated risk in macro_scraper.py (possible FRED API key leak via unsanitized exception logging) — not fixed, out of scope per stop-and-report. No production code touched — investigation/verification only.
- 2026-08-18 (coder): Implemented `src/utils/parallel.py::thread_map()` per architect spec — bounded ThreadPoolExecutor, order-preserving results, per-item try/except with None + WARNING log (setup_logger) on failure, inline loop for max_workers<=1, stagger sleep between submits/calls, empty-items short-circuit. Self-checked with a throwaway script (python.exe) covering pooled, inline, and empty-items cases — all passed, order preserved, failures replaced with None. Not yet wired into any scraper file.
- 2026-08-18 (architect): Planned a scraper performance pass (no code written). Bottlenecks found: `get_latest_date()` reads every ticker's full parquet just to get the index max and price_scraper then reads each file a second time in `upsert_dataframe`; backfill/incremental yfinance batches run strictly sequentially; `scrape_dividends` makes one `yf.Ticker().dividends` HTTP call per ticker (~503 sequential requests, the single worst offender); `aggregate_to_quarterly` recomputes every quarter back to 2015 on every run; `macro_scraper` rebuilds a FRED client per series; `live_quotes._prev_close` reads a full daily parquet per ticker on every dashboard rerun; `news/scraper.py` uses feedparser's untimed fetch and sleeps 1s per ticker. Plan adds `src/utils/parallel.py` and `src/scraper/_yf.py`, marks the 5 scraper files + `src/news/scraper.py` In Progress, and logs 5 out-of-scope findings above (uncached `get_settings()`, serial `news/runner.py`, dead `get_dividend_yield()`, non-atomic parquet writes). Concurrency defaults against yfinance/Alpaca still need api-integration sign-off before merge.
- 2026-08-18 (coder): Rewired `src/scraper/price_scraper.py` to use the already-built `_yf.py`/`storage.get_latest_dates()`/`utils.parallel.thread_map()` foundation — deleted the duplicate `_extract_ticker_from_multiindex` (confirmed unused elsewhere via grep), `download_batch()` now delegates to `_yf.yf_download()`, `scrape_prices()` gained `max_workers` and now batches `get_latest_dates()` + runs download/validate/upsert units through `thread_map()`, `aggregate_to_quarterly()` gained `max_workers`/`force` with an mtime-based skip check and column-pruned reads; `resample("QE")` spec left byte-identical. Live sanity script (AAPL/MSFT, deleted after) confirmed backfill writes rows, incremental re-run is a fast no-op, and quarterly aggregation produces sane OHLCV with working skip/force behavior. `cli/scrape.py` intentionally not touched (separate follow-up task) — flagged its BUILD_STATUS row overstating completion, see price_scraper.py notes.
- 2026-08-18 (coder): Rewired `src/news/scraper.py` per architect spec — `_parse_entry()` untouched (still directly imported by `tests/test_news/test_news.py`). `_fetch_feed()` dropped the `seen_urls` param (dedup moved to caller) and now fetches via `requests` with a per-thread `Session` (`_get_session()`, `threading.local()`), explicit `timeout=10`, and a `User-Agent` header, replacing the untimed bare `feedparser.parse(url)` call; `feedparser.parse()` is still called, just on `response.content` instead of the raw URL. `fetch_ticker_news()` fetches Yahoo+Google concurrently via `thread_map(_fetch_feed_unit, feeds, max_workers=2)` (order-preserving, so Yahoo is always index 0) then dedupes by URL Yahoo-before-Google after both return. `fetch_batch_news()` gained `max_workers: int | None = None` (→ `settings.news_max_workers`) and replaced the serial `for ticker: fetch(); time.sleep(delay)` loop with `thread_map(..., stagger=delay/max_workers)` — documented in the docstring that `delay`'s meaning changed from "hard per-ticker serialization sleep" to "stagger between task submissions on a bounded pool" (only bounds the average rate now; use `max_workers=1` for the old guarantee). Verified `python.exe -c "from src.news import scraper"` imports cleanly; `python.exe -m pytest tests/test_news/test_news.py -v` — all 22 tests pass unchanged, including the Yahoo-before-Google dedup test (confirmed this repo's test sandbox has real outbound network access, so the real `requests.get()` call succeeds before the mocked `feedparser.parse()` intercepts — no test changes were needed). Live throwaway `python.exe` sanity script (deleted after): `fetch_ticker_news("AAPL")` → 50 real articles, correct keys, sorted newest-first, no duplicate URLs; `fetch_batch_news(["AAPL","MSFT"], max_workers=2)` → both tickers returned 50 articles each. `src/news/runner.py` (`fetch_batch_news` caller) intentionally not touched — already logged above as a separate follow-up (still serializes per-ticker in its own loop). No ambiguities hit needing sign-off beyond what was already pre-approved in the task spec.
- 2026-08-17 (coder): Fixed multi_strategy.py's `_is_mean_rev` TODO — it always returned False, so mean-reversion exit signals never fired for any position. Added `TradeJournal.latest_open_strategy()` (most-recent-BUY-per-ticker lookup) and `MultiStrategyManager._mean_rev_tickers()`, computed once per generate_all_signals() call instead of the old per-position calls. Added tests/test_trading/test_multi_strategy.py (7 cases: empty journal, open position, closed position omitted, re-buy after close, manager lookup, empty input, journal-read-failure fallback). Could not execute pytest in this WSL session (no pandas/pip/venv access without sudo) — verified by syntax check and manual trace only, left as Needs Review rather than Done pending a real test run.
- 2026-08-17 (architect): First real audit since this file was seeded as a placeholder. All 11 original phases confirmed Done (files exist, non-stub, wired to their callers) — the project is substantially complete, not just planned. Only concrete issue found: src/trading/multi_strategy.py has one unfinished TODO (position-strategy tagging), marked Needs Review. Added Phase 12 to track the uncommitted research/supply-chain/Docker work (all Done, just not committed to git yet) and Phase 13 to track organic extensions (stop_manager, options_strategy, live_quotes, alerts/triggers, cli/pipeline+rank, 3 extra dashboard pages) built beyond the original plan doc but never added to this tracker. Flagged CLAUDE.md as missing despite being referenced by every agent definition. Confirmed TRADING_MODE=PAPER and ALPACA_BASE_URL point at paper-api — no live-order risk found. No code was executed as part of this audit (existence/completeness/wiring only, no correctness testing).
- 2026-08-18 (coder): Wired the final piece of the scraper perf plan into `cli/scrape.py` — added `--workers` (`type=int, default=None`) flag and passed `max_workers=args.workers` through to `scrape_prices()`, `aggregate_to_quarterly()`, and `scrape_dividends()`; `scrape_macro()` call left untouched (no `max_workers` param there by design). No other flags/structure/logging changed, `SCRAPE COMPLETE in Xs` summary line preserved verbatim. Fixed the stale BUILD_STATUS row (previously marked "Done" for this passthrough before the flag existed — flagged by prior agents). Verified `python.exe -m py_compile cli/scrape.py`, `python.exe cli/scrape.py --help` shows the new flag, and live smoke runs (`--tickers AAPL,MSFT --dividends-only --workers 2` and `--prices-only --workers 2`) completed without error (no new price rows since today's session already had current data — expected, not a failure).
