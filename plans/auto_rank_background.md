# Plan: auto_rank_background

Daily automatic ranking pipeline run in a detached background process, with a
floating progress bar on every dashboard page.

Plan author: architect. First written 2026-10-11. Revised 2026-10-11 (see Revisions).

---

## 1. Summary of approach

The ranking pipeline (`cli/pipeline.py`) runs automatically once per local
calendar day, the first time any dashboard page is opened. It runs in the
background, survives page navigation, Streamlit reruns and closed browser tabs,
and reports progress through a floating bar on every page.

Decisions (user-approved, do not reopen):
- **The daily auto-run is the FULL pipeline** (Phase 1 + Phase 2 ML training), with the
  same args as the app.py "Full Analysis Pipeline" button:
  `python cli/pipeline.py --pool 100 --top 50 --no-deep`. Progress uses the full step markers.
- **If the automatic full run fails, fall back automatically to ONE fast rank (Phase 1)
  run** so there are still rankings today. After that it is buttons only for the rest of
  the day. The full run is never auto-retried.
- **The Phase 1 "Fast Rank Only" button stays a separate, manual-only option.** Page 6
  "Re-run Ranking" also starts a manual fast rank (when no custom tickers are typed).
- **Full runs reuse today's Phase 1.** When a full run starts and a valid Phase 1 result
  from today already exists (from a fast rank, or from the Phase 1 half of a failed full
  run), the pipeline skips Phase 1 and runs only Phase 2 on the saved pool. If the saved
  result is missing or stale, it falls back to a normal full run.
- **`AUTO_RANK_ENABLED`** (env / `.env`, default `true`) turns off only the automatic
  start. Buttons always work.
- A button run is **refused** while another run is active (buttons disabled plus a
  server-side check). There is no queue.
- Background runs use lower CPU priority so the dashboard stays responsive.

Design: a new library package `src/pipeline_runner/` owns all run state on disk:
the lock, the status JSON, the daily done marker, the per-day auto-state, the Phase 1
artifact and step parsing. A new wrapper `cli/pipeline_job.py` is spawned **detached**.
It runs `cli/pipeline.py` as its own child, parses its stdout into steps, writes the
status JSON atomically, heartbeats, records success or failure, and releases the lock
last. The dashboard only reads these files, so each check is a cheap stat/read. A new
component `dashboard/components/pipeline_progress.py` calls `maybe_start_daily_run()`
on every page script run, and while a run is active or just finished it renders a
fixed-position bar from an `st.fragment(run_every="2s")`. When idle it creates no
fragment, so nothing polls.

Why this shape:
- Only a detached process plus files on disk survives reruns, page changes, multiple tabs
  and a closed browser. Streamlit threads die or get duplicated.
- The lock file (atomic `O_CREAT|O_EXCL`) is the single source of truth for "a run is in
  progress". The status file is display only.
- The full run is long, so staleness is judged by **PID liveness** of the wrapper, not by
  a short age timeout.
- `cli/pipeline.py` gets only a minimal, additive change (section 4.5): it saves the Phase 1
  result, and a new `--skip-phase1` flag reuses it. **This is the only place the plan touches
  `cli/pipeline.py`.**
- Stdlib only. No psutil, no filelock.

---

## 2. Verified state of modules this plan touches (checked on disk 2026-10-11)

All were Unverified in BUILD_STATUS.md. Checked against the real files:

- `cli/pipeline.py`: complete. Flags `--pool` (default 150), `--top` (default 50), `--no-deep`,
  `--phase1-only` and `--output`. Plain `print()` with no flush. Flow:
  - Phase 1: `get_tickers(settings.ticker_source)` filtered to tickers that have
    `raw_daily_dir/<T>.parquet`, then `rank_tickers(available, include_ml=False, include_indicators=True)`,
    then `pool_tickers = [e.ticker for e in phase1[:pool]]`.
  - Phase 2: `src.ml.runner.run_tickers(pool_tickers, ...)`, then
    `rank_tickers(pool_tickers, include_ml=True)`, which **recomputes every signal** for the pool.
  - So Phase 2 needs only the ordered `pool_tickers` list from Phase 1.
- **How Phase 1 is persisted today:** only indirectly. `--phase1-only` calls
  `save_ranking_cache(phase1, ...)`, which writes ALL Phase 1 entries to
  `data/ranking_cache.parquet` (index `ticker`; columns `rank`, `composite_score`,
  `signals_available`, `updated_at` (pandas Timestamp, tz=UTC), one float column per signal
  in `_WEIGHTS` with NaN when absent, plus optional `current_price`, `ml_target`,
  `ml_target_date`, `fcst_target`, `fcst_date`). A full run **overwrites the same file with only
  the Phase 2 pool**. The file does not record whether it came from Phase 1 or Phase 2, the pool
  size, the ticker universe, or the local run date.
  - **Conclusion: the ranking cache cannot be used reliably as the saved Phase 1 result.**
    This plan adds a dedicated Phase 1 artifact (section 5.5) and leaves the ranking cache unchanged.
- `src/ranking/ranker.py`: `RankEntry(ticker, composite_score, rank, signals, signals_available, details)`.
  Failed tickers get `composite_score = float("-inf")` and `rank = 0` and sort last. Not changed.
- `dashboard/app.py`: has `_PIPELINE_STEPS_PHASE1`, `_PIPELINE_STEPS_FULL`, `_advance_step` and
  `_run_pipeline`. Pool default `100`, top default `50`. Phase-1 args
  `["--pool", p, "--top", t, "--phase1-only"]`; full args `["--pool", p, "--top", t, "--no-deep"]`.
  It runs inline (blocking) with `PYTHONIOENCODING=utf-8` and `encoding="utf-8", errors="replace"`.
  `KEY_SPEC` is a list of `(env_var, label, sensitive, help)` written to `.env` as strings.
- `dashboard/components/market_clock.py`: uses `@st.fragment(run_every="30s")`, so the installed
  Streamlit is at least 1.37. `pyproject.toml` still pins `streamlit>=1.31.0`; section 4.9 fixes that.
- `dashboard/components/session_cache.py`: already uses `data/run_cache/`.
- `dashboard/pages/6_stock_rankings.py`: data check
  `not daily_dir.exists() or not any(daily_dir.glob("*.parquet"))`. "Re-run Ranking" currently only
  calls `st.cache_data.clear()`. `_compute_rankings` is `@st.cache_data(ttl=900)`.
- `dashboard/pages/11_sector_analysis.py`: `@st.cache_data(ttl=900) _load_top_picks(n)` reads the ranking cache.
- `config/settings.py`: `Settings` (pydantic-settings, no env prefix, `.env` at ROOT). `get_settings()`
  builds a new `Settings()` on every call (no caching), so `.env` edits are picked up without a restart.
  Existing validator pattern: `AlertSettings._coerce_smtp_port`.
- `.env.example`: exists; groups of `KEY=value` with comment headers.
- `.gitignore` ignores `data/` and `logs/`.
- All 21 pages plus `app.py` call `render_market_clock()` right after `st.set_page_config(...)`,
  before any `st.stop()`.

---

## 3. Files to create or edit

### New files

| File | Purpose |
|---|---|
| `src/pipeline_runner/__init__.py` | Package marker; re-exports the public names in section 4. |
| `src/pipeline_runner/steps.py` | Step tables (moved from app.py, plus the Phase-2-only table), stdout marker constants shared with pipeline.py, `advance_step`, `steps_for_run`, `progress_fraction`. No Streamlit. |
| `src/pipeline_runner/phase1_store.py` | Save, load and validate the dated Phase 1 artifact. Used by `cli/pipeline.py` and the launcher. No Streamlit, does not import the ranker (duck-typed entries). |
| `src/pipeline_runner/run_state.py` | On-disk run state: paths, atomic JSON IO, lock, stale-lock breaking, done marker, per-day auto-state, PID liveness, derived UI view. No Streamlit, no spawning. |
| `src/pipeline_runner/launcher.py` | `decide_auto_action` (pure), `start_pipeline_run`, `maybe_start_daily_run`, detached spawn (Windows and POSIX), child reaping. No Streamlit. |
| `cli/pipeline_job.py` | Detached wrapper. Runs `cli/pipeline.py`, parses steps, writes status and heartbeat, enforces the timeout, records the outcome, releases the lock. |
| `dashboard/components/pipeline_progress.py` | `render_pipeline_progress()`: auto-start hook, floating bar fragment, cache clear on completion, polling stopped when idle. |
| `tests/test_pipeline_runner/__init__.py` | Test package. |
| `tests/test_pipeline_runner/test_steps.py` | Step parsing tests. |
| `tests/test_pipeline_runner/test_phase1_store.py` | Phase 1 artifact tests. |
| `tests/test_pipeline_runner/test_run_state.py` | Lock, markers, day-state, liveness and view tests. |
| `tests/test_pipeline_runner/test_launcher.py` | Auto-decision, start and spawn-flag tests (fake spawn). |
| `tests/test_pipeline_runner/test_pipeline_job.py` | Wrapper tests with a fake child script. |
| `tests/test_pipeline_runner/test_pipeline_skip_phase1.py` | `cli/pipeline.py` `--skip-phase1` and artifact-write tests (all heavy calls monkeypatched). |

### Edited files

| File | Change |
|---|---|
| `pyproject.toml` | `"streamlit>=1.31.0"` becomes `"streamlit>=1.37.0"`. Nothing else. |
| `config/settings.py` | Add `auto_rank_enabled: bool = True` to `Settings`, with a "before" validator that turns blank into True (section 4.8). |
| `.env.example` | Add the `AUTO_RANK_ENABLED=true` block (section 4.8). |
| `cli/pipeline.py` | **The only edit to this file:** write the Phase 1 artifact after Phase 1; add `--skip-phase1` (section 4.5). |
| `dashboard/app.py` | Remove the inline pipeline runner and step tables. Buttons go through the launcher. Add a `KEY_SPEC` entry for `AUTO_RANK_ENABLED`. Add a last-run caption and a log expander. Add the bar call (section 7.6). |
| `dashboard/pages/6_stock_rankings.py` | Bar call. Cache-key param. "Re-run Ranking" starts a fast-rank run when no custom tickers are typed (section 7.7). |
| `dashboard/pages/11_sector_analysis.py` | Bar call. Cache-key param (section 7.4). |
| The other 19 pages (section 8) | Bar call only (2 lines). |

### Not touched

`cli/scheduler.py`, `src/ranking/ranker.py`, `src/alerts/triggers.py` (it has an unrelated
uncommitted change; leave it alone), API client modules in `docs/api_limits.md` (apart from the
non-API UI edits listed above), `dashboard/components/market_clock.py`, `launch_dashboard.bat`,
ARCHITECTURE.md and DOCUMENTATION.md (the main session handles those).

Note for the main session: `dashboard/app.py`, `pages/10_portfolio.py` and `pages/11_sector_analysis.py`
are listed as API client modules in `docs/api_limits.md`. This plan makes no change to their API calls,
so the coder agent does these edits, not api-integration.

---

## 4. Interfaces and signatures (must match exactly)

Conventions:
- Every timestamp stored on disk is a **UTC ISO-8601 string with offset**:
  `datetime.now(timezone.utc).isoformat(timespec="seconds")`, e.g. `"2026-10-11T15:02:03+00:00"`.
- Every "day" key is the **local** date `datetime.now().date().isoformat()`, e.g. `"2026-10-11"` (reason in 6.1).
- Files are UTF-8. JSON is written with `ensure_ascii=False` and always through the atomic writer.

Type aliases (in `run_state.py`, re-exported):

```python
RunMode    = Literal["phase1", "full"]
RunTrigger = Literal["auto", "auto_fallback", "button"]
RunState   = Literal["running", "done", "failed"]          # stored in the status file
ViewState  = Literal["idle", "running", "done", "failed"]  # derived for the UI
```

### 4.1 `src/pipeline_runner/steps.py`

```python
Step = tuple[list[str], str]   # (substring markers, user-facing label)

# Exact stdout strings shared with cli/pipeline.py. pipeline.py MUST import these constants
# rather than retyping them.
PHASE1_REUSE_MARKER    = "Reusing saved Phase 1 result"
PHASE1_FALLBACK_MARKER = "Saved Phase 1 result not usable"

PIPELINE_STEPS_PHASE1: list[Step]        # moved verbatim from app.py _PIPELINE_STEPS_PHASE1 (5 steps)
PIPELINE_STEPS_FULL: list[Step]          # moved verbatim from app.py _PIPELINE_STEPS_FULL (9 steps)
PIPELINE_STEPS_PHASE2_ONLY: list[Step]   # 8 steps:
#   ([PHASE1_REUSE_MARKER],              "Loading today's Phase 1 result..."),
#   (["Training ML models"],             "Phase 2: Training ML models on top candidates..."),
#   (["ML training done"],               "ML training done - re-ranking with ML signal..."),
#   (["Re-ranking top"],                 "Phase 2: Re-ranking all candidates with ML signal..."),
#   (["Re-rank done"],                   "Re-ranking done - collecting price targets..."),
#   (["Collecting price", "[Output]"],   "Collecting price targets..."),
#   (["Ranking cache saved"],            "Saving ranking cache..."),
#   (["AutoStockAnalyzer --"],           "Building results table..."),
# Entries 1..7 must have the same markers as PIPELINE_STEPS_FULL[2..8]. Labels may keep app.py's
# original text (the em dash in app.py labels may stay as-is; only the markers must match).

def steps_for_run(mode: RunMode, phase1_reused: bool = False) -> list[Step]: ...
    # phase1 -> PHASE1 (phase1_reused ignored); full & not reused -> FULL; full & reused -> PHASE2_ONLY.
    # Unknown mode -> ValueError.

def advance_step(line: str, steps: list[Step], current: int) -> int: ...
    # Same as app.py _advance_step: first i in range(current+1, len(steps)) where any marker of
    # steps[i] is a substring of line; else current. Never goes backwards. Case-sensitive.

def progress_fraction(step_index: int, step_total: int, state: RunState) -> float: ...
    # done -> 1.0; step_total <= 1 -> 0.0; else min(step_index/(step_total-1), 0.99); clamp to [0, 1].
```

Phase-1 and full markers stay byte-identical to today's app.py, including `"AutoStockAnalyzer --"`
(two ASCII hyphens, pipeline.py line 112) and `"[Output]"`.

### 4.2 `src/pipeline_runner/phase1_store.py`

```python
PHASE1_SCHEMA_VERSION = 1
PHASE1_FILENAME = "phase1_result.json"      # stored in <data_dir>/run_cache/

def phase1_result_path(data_dir: Path) -> Path: ...

def save_phase1_result(entries: Sequence[Any], settings: Any, universe_size: int,
                       now: datetime | None = None, day: str | None = None) -> Path: ...
    # entries: RankEntry-like objects (attributes .ticker, .rank, .composite_score), in the EXACT
    # list order produced by rank_tickers (including -inf failures at the end).
    # settings: needs .data_dir and .ticker_source.
    # Writes the JSON in section 5.5 atomically (tmp + os.replace, same helper semantics as
    # run_state.write_json_atomic; import it from run_state). Creates run_cache dir if needed.

@dataclass(frozen=True)
class Phase1Reuse:
    usable: bool
    reason: str                     # "ok" or a short human-readable reason (no newlines)
    pool_tickers: list[str] | None  # entries[:pool] tickers in saved order when usable and pool given
    created_at: str | None
    ranked_count: int

def check_phase1_reusable(settings: Any, pool: int | None = None,
                          today: str | None = None) -> Phase1Reuse: ...
    # settings needs .data_dir, .raw_daily_dir, .ticker_source.
    # Not usable (reasons, checked in this order):
    #   "missing"                         file absent
    #   "unreadable"                      bad JSON / not an object / wrong schema_version / bad types
    #   "not from today (<run_date>)"     run_date != today (local)
    #   "ticker source changed"           ticker_source != settings.ticker_source
    #   "empty"                           no entries
    #   "price data updated after Phase 1 ran"
    #                                     max st_mtime over raw_daily_dir/*.parquet > created_at
    # Usable: pool_tickers = [e["ticker"] for e in entries[:pool]] if pool is not None, else None.
    # pool/top differences are NOT a reason to refuse (see 6.8). pool > ranked_count simply
    # yields all entries, which is identical to phase1[:pool] in pipeline.py.
    # Never raises (any exception -> usable False, reason "unreadable").
```

### 4.3 `src/pipeline_runner/run_state.py`

Constants (module level; tests may patch them):

```python
STATUS_SCHEMA_VERSION = 1
DAY_STATE_SCHEMA_VERSION = 1
HEARTBEAT_INTERVAL_S  = 5      # wrapper writes heartbeat at least this often
PID_REUSE_GUARD_S     = 900    # SECONDARY guard only: pid looks alive but no heartbeat for 15 min
SPAWN_GRACE_S         = 60     # lock/run with no pid recorded yet is trusted this long
BREAKER_STALE_S       = 30     # lock-break mutex older than this is removed
DONE_DISMISS_S        = 6      # bar shows "done" this long after finished_at
FAILED_DISMISS_S      = 30     # bar shows "failed" this long after finished_at
MARKER_RETENTION_DAYS = 7
LOG_RETENTION_COUNT   = 14
```

Paths:

```python
@dataclass(frozen=True)
class RunPaths:
    data_dir: Path
    run_cache_dir: Path      # <data_dir>/run_cache
    lock_file: Path          # <run_cache_dir>/pipeline_run.lock
    breaker_file: Path       # <run_cache_dir>/pipeline_run.lock.break
    status_file: Path        # <run_cache_dir>/pipeline_status.json
    log_dir: Path            # <data_dir>/logs
    def done_marker(self, day: str) -> Path: ...   # <run_cache_dir>/.done_rank_<day>   (full success only)
    def day_state_file(self, day: str) -> Path: ...# <run_cache_dir>/auto_rank_<day>.json
    def log_file(self, run_id: str) -> Path: ...   # <log_dir>/pipeline_job_<run_id>.log

def get_run_paths(data_dir: Path | None = None) -> RunPaths: ...
    # None -> get_settings().data_dir. mkdir run_cache_dir and log_dir (parents, exist_ok).
    # Tests ALWAYS pass tmp_path.
```

Every function below takes `paths: RunPaths` first. Time functions accept `now: datetime | None`
(tz-aware UTC; default `datetime.now(timezone.utc)`). Day functions accept `day: str | None`
(default `local_today()`).

```python
def local_today() -> str: ...
def utc_now_iso() -> str: ...
def new_run_id(now: datetime | None = None) -> str: ...
    # f"{UTC %Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}" e.g. "20261011T150203Z-a1b2c3". Filename-safe.

# atomic JSON
def write_json_atomic(path: Path, data: dict) -> None: ...
    # tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"); write; os.replace.
    # On PermissionError (Windows reader holds the file) retry 5 times, 50 ms apart, then re-raise.
def read_json(path: Path) -> dict | None: ...   # None if missing/empty/invalid/not an object. Never raises.

# status
def read_status(paths: RunPaths) -> dict | None: ...   # validates schema_version==1 and state; else None
def write_status(paths: RunPaths, status: dict) -> None: ...

# lock
@dataclass
class LockInfo:
    run_id: str | None
    created_at: datetime     # from JSON; if missing/unparseable -> file mtime (UTC)
    run_date: str | None
    mode: str | None
    trigger: str | None

def read_lock(paths: RunPaths) -> LockInfo | None: ...  # None only if the file does not exist
def try_acquire_lock(paths: RunPaths, run_id: str, run_date: str, mode: RunMode,
                     trigger: RunTrigger, now: datetime | None = None) -> bool: ...
    # os.open(lock_file, O_CREAT|O_EXCL|O_WRONLY, 0o644); FileExistsError -> False.
    # Write lock JSON (5.2), flush, os.fsync, close -> True.
def release_lock(paths: RunPaths, run_id: str) -> bool: ...  # unlink only if lock.run_id == run_id
def lock_owner_alive(paths: RunPaths, lock: LockInfo, now: datetime | None = None) -> bool: ...  # 6.3
def is_run_active(paths: RunPaths, now: datetime | None = None) -> bool: ...  # read-only
def break_stale_lock(paths: RunPaths, now: datetime | None = None) -> bool: ...  # 6.4

def pid_alive(pid: int | None) -> bool: ...
    # None/<=0 -> False.
    # POSIX: os.kill(pid, 0): ProcessLookupError -> False; PermissionError -> True; ok -> True.
    # Windows: NEVER os.kill (it terminates the process). ctypes:
    #   OpenProcess(0x1000 PROCESS_QUERY_LIMITED_INFORMATION, False, pid); null -> False unless
    #   GetLastError()==5 (ACCESS_DENIED) -> True; GetExitCodeProcess -> alive iff 259 (STILL_ACTIVE);
    #   CloseHandle always.

# done marker (full success only)
def has_done_marker(paths: RunPaths, day: str | None = None) -> bool: ...
def write_done_markers(paths: RunPaths, days: Iterable[str], now: datetime | None = None) -> None: ...
    # one file per distinct day, content utc_now_iso()+"\n"; never raises (log OSError)

# per-day auto-state (replaces the earlier ".fail_rank_<day>" counter)
@dataclass(frozen=True)
class DayState:
    day: str
    full_failures: int = 0          # failed FULL runs today (any trigger, incl. stale-broken)
    fallback_started: bool = False  # the automatic Phase-1 fallback was claimed today
    phase1_success: bool = False    # any Phase-1 run (any trigger) succeeded today

def read_day_state(paths: RunPaths, day: str | None = None) -> DayState: ...
    # missing/invalid -> DayState(day) with defaults. Never raises.
def record_full_failure(paths: RunPaths, day: str) -> DayState: ...      # full_failures += 1
def record_fallback_started(paths: RunPaths, day: str) -> DayState: ...  # fallback_started = True
def record_phase1_success(paths: RunPaths, day: str) -> DayState: ...    # phase1_success = True
    # Each is read-modify-write via write_json_atomic. Callers always hold either the run lock or
    # the breaker mutex, so writes are serialized (see 6.7).

def cleanup_old_files(paths: RunPaths, today: str | None = None) -> None: ...
    # delete .done_rank_<d> and auto_rank_<d>.json older than MARKER_RETENTION_DAYS (skip unparseable
    # names); keep the newest LOG_RETENTION_COUNT pipeline_job_*.log by mtime. Never raises.

# view for UI
@dataclass(frozen=True)
class RunView:
    state: ViewState
    run_id: str | None
    mode: str | None
    trigger: str | None
    phase1_reused: bool
    label: str
    step_index: int
    step_total: int
    progress: float
    error: str | None
    started_at: str | None
    step_started_at: str | None
    finished_at: str | None

def get_run_view(paths: RunPaths, now: datetime | None = None) -> RunView: ...  # 7.2, read-only
def ranking_cache_version(data_dir: Path | None = None) -> float: ...
    # st_mtime of <data_dir>/ranking_cache.parquet or 0.0
def read_log_tail(paths: RunPaths, run_id: str, max_lines: int = 100) -> list[str]: ...
```

### 4.4 `src/pipeline_runner/launcher.py`

```python
DEFAULT_POOL: int = 100     # app.py number_input default
DEFAULT_TOP: int  = 50      # app.py number_input default

AutoAction = Literal["disabled", "already_done", "running", "no_data",
                     "start_full", "start_fallback", "buttons_only"]

def decide_auto_action(enabled: bool, has_marker: bool, lock_alive: bool,
                       day: DayState, has_data: bool) -> AutoAction: ...
    # PURE. Evaluate in this order:
    #   not enabled                                         -> "disabled"
    #   has_marker                                          -> "already_done"
    #   lock_alive                                          -> "running"
    #   not has_data                                        -> "no_data"
    #   day.full_failures == 0                              -> "start_full"
    #   not day.fallback_started and not day.phase1_success -> "start_fallback"
    #   else                                                -> "buttons_only"

StartReason = Literal["started", "already_done", "running", "no_data",
                      "spawn_error", "disabled", "buttons_only"]

@dataclass(frozen=True)
class StartResult:
    started: bool
    reason: StartReason
    run_id: str | None
    mode: RunMode | None
    phase1_reused: bool
    message: str          # short, user-facing

SpawnFn = Callable[[list[str], Path], int]   # (argv, cwd) -> pid

def has_scraped_data(raw_daily_dir: Path | None = None) -> bool: ...
    # dir exists and next(iter(dir.glob("*.parquet")), None) is not None

def build_wrapper_cmd(run_id: str, run_date: str, mode: RunMode, pool: int, top: int,
                      trigger: RunTrigger, reuse_phase1: bool,
                      root_dir: Path | None = None) -> list[str]: ...
    # [sys.executable, str(ROOT/"cli"/"pipeline_job.py"), "--run-id", run_id, "--run-date", run_date,
    #  "--mode", mode, "--pool", str(pool), "--top", str(top), "--trigger", trigger]
    #  + (["--reuse-phase1"] if reuse_phase1 else [])
    # ROOT = config.settings.ROOT_DIR when root_dir is None. All str. No shell.

def spawn_detached(argv: list[str], cwd: Path) -> int: ...    # section 9
def reap_children() -> None: ...   # poll() all Popen in module-level _SPAWNED; drop finished; never raises

def start_pipeline_run(mode: RunMode, pool: int = DEFAULT_POOL, top: int = DEFAULT_TOP,
                       trigger: RunTrigger = "button",
                       paths: RunPaths | None = None, settings: Any | None = None,
                       spawn: SpawnFn | None = None, now: datetime | None = None,
                       _expected_auto_action: AutoAction | None = None) -> StartResult: ...
    # Algorithm 6.5. Buttons call this directly; a button always ignores the marker and the day-state cap.

def maybe_start_daily_run(paths: RunPaths | None = None, settings: Any | None = None,
                          spawn: SpawnFn | None = None, now: datetime | None = None) -> StartResult: ...
    # Algorithm 6.6.
```

`settings` (when given) must provide `.data_dir`, `.raw_daily_dir`, `.ticker_source` and
`.auto_rank_enabled`. Tests pass a `types.SimpleNamespace`. When `None`, use `get_settings()`.
Validation (raise `ValueError` before any side effect): `pool` an int in 10..503, `top` in 5..200
(the app.py bounds), `mode` in RunMode, `trigger` in RunTrigger.

### 4.5 `cli/pipeline.py` changes (the only edit to this file)

Additive. Default behaviour without the new flag is unchanged, apart from also writing the artifact.

```python
def run_pipeline(pool: int = 150, top: int = 50, no_deep: bool = False,
                 phase1_only: bool = False, output: str | None = None,
                 skip_phase1: bool = False) -> None: ...
```

1. Imports: `from src.pipeline_runner.phase1_store import save_phase1_result, check_phase1_reusable`
   and `from src.pipeline_runner.steps import PHASE1_REUSE_MARKER, PHASE1_FALLBACK_MARKER`.
2. `if phase1_only and skip_phase1: raise ValueError("--skip-phase1 cannot be combined with --phase1-only")`.
3. If `skip_phase1`:
   `reuse = check_phase1_reusable(settings, pool=pool)`.
   - Usable: `pool_tickers = reuse.pool_tickers`, then
     `print(f"[Phase 1] {PHASE1_REUSE_MARKER} from {reuse.created_at} ({reuse.ranked_count} ranked) -- skipping Phase 1, pool: top {len(pool_tickers)} candidates")`,
     then go straight to Phase 2.
   - Not usable: `print(f"[Phase 1] {PHASE1_FALLBACK_MARKER}: {reuse.reason} -- running Phase 1")`,
     then run the normal Phase 1 block.
4. Normal Phase 1 block (any mode, including the fallback path), right after `phase1 = rank_tickers(...)`:
   ```
   try: save_phase1_result(phase1, settings, universe_size=len(available))
   except Exception as exc: print(f"[Phase 1] Warning: could not save Phase 1 result: {exc}")
   ```
   Saving must never fail the run. Write it **before** the `[Phase 1] Done` print.
5. Phase 2, price collection, cache save, printing and CSV are unchanged.
6. argparse: put `--phase1-only` and the new `--skip-phase1` (`action="store_true"`,
   help "Reuse today's saved Phase 1 result and run only Phase 2; falls back to a full run if
   it is missing or stale.") in a mutually exclusive group. Pass `skip_phase1=args.skip_phase1`.
7. Update the module docstring usage block with one line for `--skip-phase1`.
8. Neither printed line may contain any other step marker. Check that `"Fast-ranking"`,
   `"[Phase 1] Done"` and `"Training ML models"` do not appear in them.

### 4.6 `cli/pipeline_job.py`

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace: ...
    # required: --run-id, --run-date (YYYY-MM-DD), --mode {phase1,full}, --pool int, --top int,
    #           --trigger {auto,auto_fallback,button}
    # optional: --reuse-phase1 (store_true; only valid with --mode full, else exit 2)
    # test hooks: --data-dir PATH, --pipeline-script PATH, --max-runtime-s int, --heartbeat-s float

def pipeline_argv(mode: RunMode, pool: int, top: int, script: Path, reuse_phase1: bool) -> list[str]: ...
    # phase1: [sys.executable, "-u", str(script), "--pool", P, "--top", T, "--phase1-only"]
    # full:   [sys.executable, "-u", str(script), "--pool", P, "--top", T, "--no-deep"]
    #         + ["--skip-phase1"] if reuse_phase1

def run_job(args: argparse.Namespace) -> int: ...   # 6.9. 0 ok, 1 failed, 2 not owner/bad args, 3 superseded
def main() -> None: ...                              # sys.exit(run_job(parse_args()))
```

Max runtime defaults: phase1 1800 s, full 14400 s (4 h). This cap is enforced only by the
wrapper; readers never time out a run by age.
Priority: on POSIX the wrapper calls `os.nice(10)` once (try/except OSError), and the child
inherits it. On Windows the child gets `BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW`
(via `getattr(subprocess, ..., 0)`).
`sys.path.insert(0, ROOT)` at the top, like the other cli scripts.

### 4.7 `dashboard/components/pipeline_progress.py`

```python
POLL_INTERVAL = "2s"
def render_pipeline_progress() -> None: ...                 # only public function
def _maybe_clear_caches_for(run_id: str) -> None: ...       # once per process per run_id
@st.fragment(run_every=POLL_INTERVAL)
def _progress_fragment() -> None: ...
def _bar_html(view: RunView, now: datetime | None = None) -> str: ...   # pure; html.escape all dynamic text
def _fmt_elapsed(seconds: float) -> str: ...                # "45s", "7m 12s", "1h 03m"
```

Process-wide: `_CLEARED_RUN_IDS: set[str]` guarded by `_CLEAR_LOCK = threading.Lock()`.
Session keys (prefix `_plbar_`): `_plbar_seen_run_id`, `_plbar_rerun_done_for`, `_plbar_last_start_reason`.

### 4.8 Setting `AUTO_RANK_ENABLED`

`config/settings.py`, in `Settings`, in a new `# Dashboard automation` group:
```python
auto_rank_enabled: bool = True   # AUTO_RANK_ENABLED; False disables only the automatic daily run

@field_validator("auto_rank_enabled", mode="before")
@classmethod
def _coerce_auto_rank_enabled(cls, v):
    # blank/None -> True; bool passes through;
    # true/1/yes/on/t/y -> True; false/0/no/off/f/n -> False (case-insensitive, stripped);
    # anything else -> False + logging warning (message must NOT include the bad value).
    ...
```
The validator is required. app.py's settings form can save a blank value, and a blank bool would
otherwise make `Settings()` raise on every page. Behavior (as implemented):
- `None` or a blank/whitespace string -> `True`.
- A `bool` passes through unchanged.
- Strings `true/1/yes/on/t/y` -> `True`; `false/0/no/off/f/n` -> `False` (case-insensitive).
- ANY other value -> `False`, and a warning is logged with stdlib `logging`. The log message does not
  include the bad value.

`Settings()` therefore never raises on an invalid `AUTO_RANK_ENABLED`; an invalid value simply
disables the automatic run. `maybe_start_daily_run` does not need to catch anything for this setting.

`.env.example`, appended after the alerts block:
```
# Dashboard automation
# Run the full ranking pipeline automatically once per day, the first time the
# dashboard is opened. Set to false to disable the automatic run (buttons still work).
AUTO_RANK_ENABLED=true
```

`dashboard/app.py` `KEY_SPEC`, appended:
`("AUTO_RANK_ENABLED", "Auto-run daily ranking", False, "true or false. Runs the full pipeline automatically on the first dashboard open each day. Buttons still work.")`
It fits: KEY_SPEC entries are plain strings, it is not sensitive, and the form already writes
any key into `.env`. Because `get_settings()` re-reads `.env` on every call, the change takes effect
on the next page load.

### 4.9 `pyproject.toml`

Change the single dependency line `"streamlit>=1.31.0",` to `"streamlit>=1.37.0",`.
Reason: `st.fragment(run_every=...)` and `st.rerun(scope="app")` need 1.37 or later.

---

## 5. On-disk formats

Everything is under `settings.data_dir` (= `ROOT_DIR/"data"`). No absolute or private paths.

### 5.1 Status file `data/run_cache/pipeline_status.json`

| Key | Type | Notes |
|---|---|---|
| `schema_version` | int | `1` |
| `run_id` | str | |
| `state` | str | `running` \| `done` \| `failed` |
| `mode` | str | `phase1` \| `full` |
| `trigger` | str | `auto` \| `auto_fallback` \| `button` |
| `phase1_reused` | bool | full runs: true while running Phase 2 only; set to false if pipeline fell back. Always false for phase1 |
| `pool` | int | |
| `top` | int | |
| `pid` | int \| null | wrapper pid. null only in the first pre-spawn record; the launcher sets `Popen.pid` right after spawn; the wrapper rewrites it with `os.getpid()` |
| `child_pid` | int \| null | pid of `cli/pipeline.py`; informational |
| `step_index` | int | 0-based into `steps_for_run(mode, phase1_reused)` |
| `step_total` | int | length of that table (5, 9 or 8) |
| `label` | str | current step label; `"Ranking complete"` on done; a failure label on failed |
| `run_date` | str | local `YYYY-MM-DD` at claim time |
| `started_at` | str | UTC ISO |
| `step_started_at` | str \| null | UTC ISO when the current step began |
| `heartbeat_at` | str | UTC ISO; at least every HEARTBEAT_INTERVAL_S while running |
| `finished_at` | str \| null | UTC ISO at done or failed |
| `returncode` | int \| null | pipeline child exit code |
| `error` | str \| null | failed only: reason plus last 20 output lines, at most 1000 chars |
| `log_path` | str | relative to ROOT_DIR with forward slashes, e.g. `data/logs/pipeline_job_<run_id>.log` |

Writers: the launcher (pre-spawn record, then the pid fill, then a failure record on spawn error),
the wrapper, and `break_stale_lock`. Readers never write.

### 5.2 Lock file `data/run_cache/pipeline_run.lock`

`{"run_id", "created_at" (UTC ISO), "run_date", "mode", "trigger", "launcher_pid"}`. Created only by
`try_acquire_lock` (O_EXCL), never rewritten. Removed only by `release_lock` (owner) or
`break_stale_lock`.

### 5.3 Step markers

Unchanged from app.py for phase1 (5) and full (9). Phase-2-only (8) is defined in 4.1.

### 5.4 Per-day files

| File | Content | Writer |
|---|---|---|
| `data/run_cache/.done_rank_<day>` | `utc_now_iso()+"\n"` | wrapper, **full success only** |
| `data/run_cache/auto_rank_<day>.json` | `{"schema_version":1,"day":str,"full_failures":int,"fallback_started":bool,"phase1_success":bool,"updated_at":UTC ISO}` | wrapper (outcome), launcher (fallback claim / spawn error), `break_stale_lock` |
| `data/run_cache/pipeline_run.lock.break` | `str(os.getpid())` | `break_stale_lock` (milliseconds) |
| `data/logs/pipeline_job_<run_id>.log` | header plus the full pipeline stdout/stderr, utf-8 | wrapper |

These cannot collide with the scheduler's markers (`.done_<job>_<UTC date>` in `data/logs/`).

### 5.5 Phase 1 artifact `data/run_cache/phase1_result.json`

```json
{
  "schema_version": 1,
  "created_at": "2026-10-11T15:02:40+00:00",
  "run_date": "2026-10-11",
  "ticker_source": "sp500",
  "universe_size": 498,
  "include_indicators": true,
  "entries": [
    {"ticker": "NVDA", "rank": 1, "score": 0.4123},
    {"ticker": "XYZ",  "rank": 0, "score": null}
  ]
}
```
- `entries` is in the exact order of the `phase1` list from `rank_tickers`, including failed
  tickers (`composite_score == -inf`, `rank 0`), which are stored with `"score": null` because JSON
  has no -inf. This keeps `entries[:pool]` identical to `phase1[:pool]`.
- `score` is a float rounded to 6 decimals, or null. `rank` is an int.
- Written by `cli/pipeline.py` after every Phase 1 computation (phase1-only runs, full runs,
  and the fallback path of `--skip-phase1`). Read by `cli/pipeline.py --skip-phase1` and by the launcher.
- Single file; each Phase 1 run overwrites it. Not deleted by cleanup (it is one small file).

### 5.6 Ranking cache `data/ranking_cache.parquet`

Unchanged schema and writer (`save_ranking_cache`). A Phase-1 run overwrites it with Phase-1
(no-ML) rankings; a full or Phase-2-only run overwrites it with the Phase-2 pool. This is existing
behaviour (see 6.10).

---

## 6. Semantics

### 6.1 "Today" = LOCAL date

The guard means "the first dashboard open on the user's calendar day". The scheduler uses UTC
because its cron triggers are UTC. For a US user, UTC midnight is 4-8 pm local, which would cause
a surprise second auto-run in the evening. So day keys use `datetime.now().date()` on the
Streamlit machine. Status timestamps stay UTC for machine comparisons, and the UI shows local time.

### 6.2 Day attribution and midnight rollover

- `run_date` = local date when the lock is acquired (stored in the lock, the status and the wrapper args).
- A full success writes done markers for `{run_date, local_today() at finish}`. A full run that
  crosses midnight therefore counts for both days, and there is no second full auto-run straight
  after midnight.
- Full failure: `record_full_failure(run_date)`. Phase-1 success: `record_phase1_success(run_date)`
  and also `record_phase1_success(local_today())` if that is a different day.
  Phase-1 failure: no day-state change. (`fallback_started` was already set when the fallback was claimed.)
- The Phase 1 artifact's `run_date` is the local date when Phase 1 finished. A full run after
  midnight will not reuse a Phase 1 from the previous day.
- A tab left open over midnight auto-starts only at its next script run. There is no timer.

### 6.3 Liveness (`lock_owner_alive`): PID first

Given lock `L` and `S = read_status(paths)`:
1. If `S` exists and `S.run_id == L.run_id`:
   - `S.state` is done or failed: alive = `pid_alive(S.pid)` or `now - S.finished_at < SPAWN_GRACE_S`.
   - `S.state == "running"` and `S.pid` is set: alive = `pid_alive(S.pid)` and
     `now - S.heartbeat_at < PID_REUSE_GUARD_S`. A dead pid means stale immediately, at any age.
     The 15-min heartbeat rule only catches a reused pid or a frozen or suspended process. The wrapper's
     heartbeat thread runs on its own regardless of child output, so a healthy run that is silent
     during ML training is never caught by it.
   - `S.state == "running"` and `S.pid` is None: alive = `now - L.created_at < SPAWN_GRACE_S`.
2. Otherwise (no status for this run, or a corrupt lock with `run_id None`): alive = `now - L.created_at < SPAWN_GRACE_S`.

Callers in the launcher and the dashboard call `launcher.reap_children()` before any liveness
check. On POSIX an unreaped zombie answers `os.kill(pid, 0)` as alive.

### 6.4 `break_stale_lock` (race-safe)

1. Read the lock. If it is missing or `lock_owner_alive`, return False.
2. Breaker mutex: `os.open(breaker_file, O_CREAT|O_EXCL|O_WRONLY)`. On FileExistsError: if its
   mtime is older than BREAKER_STALE_S, unlink it (ignore FileNotFoundError) and retry once; otherwise return False.
3. With the mutex held, re-read the lock. If it is missing, has a different run_id, or is now alive,
   release the mutex and return False.
4. If `S.run_id == L.run_id` and `S.state == "running"`: write `S` as failed with `finished_at=now` and
   `error="Run stopped responding (process exited or was killed without reporting). See log."`.
   Then, if the mode is full (`L.mode` or `S.mode`), call `record_full_failure(L.run_date or local_today())`.
   A dead phase1 run changes no day state. If `S` was already terminal, write nothing.
5. Unlink the lock (ignore FileNotFoundError). Unlink the breaker in `finally`. Return True.

### 6.5 `start_pipeline_run(mode, pool, top, trigger, ...)`

```
validate (ValueError before side effects)
reap_children()
paths, settings = defaults if None
today = local_today()
lock = read_lock(paths)
if lock: if lock_owner_alive -> (False,"running",lock.run_id,"A ranking run is already in progress.")
         else break_stale_lock(paths)
if not has_scraped_data(settings.raw_daily_dir) -> (False,"no_data","No scraped price data found. Run the scraper first.")
reuse = mode == "full" and check_phase1_reusable(settings, pool=pool).usable
run_id = new_run_id(now)
if not try_acquire_lock(paths, run_id, today, mode, trigger) -> (False,"running", current lock run_id, ...)
# --- lock held from here; every exit path below must release it if no wrapper was spawned ---
if _expected_auto_action is not None:            # auto triggers only (see 6.6)
    recompute decide_auto_action(enabled, has_done_marker(today), lock_alive=False,
                                 read_day_state(today), has_data=True)
    if result != _expected_auto_action: release_lock; return (False, mapped reason)
       # mapping: already_done->already_done, buttons_only->buttons_only, disabled->disabled,
       #          start_full/start_fallback mismatch->"running" (another tab moved state on)
if trigger == "auto_fallback": record_fallback_started(paths, today)   # claimed even if spawn fails
try:
    write_status(pre-spawn: running, pid None, phase1_reused=reuse, step 0 of steps_for_run(mode, reuse),
                 label, started_at=step_started_at=heartbeat_at=now, run_date=today, log_path, ...)
    pid = (spawn or spawn_detached)(build_wrapper_cmd(..., reuse_phase1=reuse), ROOT_DIR)
    if (s := read_status(paths)) and s["run_id"] == run_id and s["pid"] is None:
        write_status({**s, "pid": pid})          # never overwrite a record the wrapper already wrote
except Exception as exc:
    write_status(failed, error=f"Could not start background run: {exc}"[:1000], finished_at=now)
    if mode == "full": record_full_failure(paths, today)
    release_lock(paths, run_id)
    -> (False,"spawn_error",run_id,...)
-> (True,"started",run_id,mode,reuse,message)
```
Buttons never consult the done marker or the day-state cap. They only refuse on a live lock or missing data.
Messages: started full -> "Full analysis started in the background." (if reuse: "Full analysis started
(reusing today's Phase 1, running ML only)."); started phase1 -> "Fast rank started in the background."

### 6.6 `maybe_start_daily_run()`

```
reap_children()
try: settings = settings or get_settings()
except Exception: log warning -> StartResult(False,"disabled",...)      # any settings load failure (an invalid AUTO_RANK_ENABLED no longer raises; see 4.8)
paths = paths or get_run_paths(settings.data_dir)
today = local_today()
lock = read_lock(paths)
lock_alive = bool(lock) and lock_owner_alive(paths, lock)
if lock and not lock_alive: break_stale_lock(paths); lock_alive = False
action = decide_auto_action(settings.auto_rank_enabled, has_done_marker(paths, today), lock_alive,
                            read_day_state(paths, today), has_scraped_data(settings.raw_daily_dir))
"start_full"     -> start_pipeline_run("full",   DEFAULT_POOL, DEFAULT_TOP, trigger="auto",
                                       _expected_auto_action="start_full", ...)
"start_fallback" -> start_pipeline_run("phase1", DEFAULT_POOL, DEFAULT_TOP, trigger="auto_fallback",
                                       _expected_auto_action="start_fallback", ...)
otherwise        -> StartResult(False, action mapped to StartReason, ...)
```
The first two checks (enabled, done marker) are a settings read and one stat. That is the common
case on every page load after the first run. The component wraps this call in `try/except Exception`
and logs through `src.utils.logging.setup_logger("dashboard.pipeline_progress")`, so it never raises into a page.

### 6.7 Daily auto state machine

States are derived from the files (done marker, day state, lock, status):

```
NOT_STARTED --(page load, enabled, data)--> AUTO_FULL_RUNNING          [trigger auto, mode full]
AUTO_FULL_RUNNING --success--> DONE_TODAY                              [.done_rank_<day>]
AUTO_FULL_RUNNING --failure / stale-broken--> FULL_FAILED              [full_failures=1]
FULL_FAILED --(next page load, no phase1_success today)--> FALLBACK_RUNNING
                                                    [trigger auto_fallback, mode phase1, fallback_started=true]
FULL_FAILED --(phase1_success already true today)--> BUTTONS_ONLY
FALLBACK_RUNNING --success--> BUTTONS_ONLY (rankings exist; phase1_success=true)
FALLBACK_RUNNING --failure--> BUTTONS_ONLY
Any state --button--> a button run (refused only if a run is live or there is no data)
A successful full button run in any state --> DONE_TODAY
Not enabled --> automation does nothing (DISABLED); buttons unaffected
```
- The full run is never auto-retried. The fallback runs at most once per day.
- Full failures from any trigger count: a failed full button run before the auto run also moves the day
  to FULL_FAILED. The bar for that run says "failed". The next page load starts the fallback only if no
  Phase-1 success exists today.
- All day-state writers hold the run lock (wrapper, launcher) or the breaker mutex over a dead lock
  (`break_stale_lock`), so read-modify-write cannot interleave.
- After an auto full failure, the open tab's fragment does `st.rerun(scope="app")` when the failed bar is
  dismissed (7.3). That full rerun calls `maybe_start_daily_run`, so the fallback starts about 30 s
  later without any user action. Any earlier interaction in any tab starts it sooner.

### 6.8 Phase 1 reuse rules

- Reuse happens for **every full run (auto or button)** whose launcher check finds a usable artifact.
  In practice that is a full button run after a fast rank (button, page 6, or auto fallback) or after
  the Phase 1 half of a failed full run, on the same day, with no newer price data.
- **pool/top mismatches do not refuse reuse.** The artifact stores the complete ordered Phase-1
  ranking, not a pool-sized slice. So `entries[:pool]` for any requested pool is exactly what Phase 1
  would have produced (same data, same day, same ticker source). `top` only slices the final output
  after Phase 2. The coordinator suggested refusing on a pool/top mismatch; that would only force a
  redundant Phase 1. This is an intentional deviation, flagged in section 13.
- Staleness is checked twice: in the launcher (picks the step table and the `--skip-phase1` flag) and
  again inside `cli/pipeline.py` (data could change between the two checks). If the pipeline-side check
  fails, it prints `PHASE1_FALLBACK_MARKER` and runs a normal Phase 1. The wrapper switches the bar to
  the 9-step table (6.9 step 8).
- Freshness only compares against `raw_daily_dir/*.parquet` mtimes. News, forecasts and options files
  updated later the same day are not detected. Phase 2 recomputes all signals for the pool anyway, so
  only pool membership could be slightly out of date. Accepted.

### 6.9 Wrapper `run_job`

1. `paths = get_run_paths(args.data_dir)`. If `read_lock(paths).run_id != args.run_id`, exit 2 and write nothing.
   `--reuse-phase1` with `--mode phase1` also exits 2.
2. `cleanup_old_files(paths)`. POSIX: `os.nice(10)` (try/except).
3. Open the log (append, utf-8) and write a header line (run_id, mode, pool, top, trigger, reuse, UTC start).
4. Write status running with `pid=os.getpid()`. Keep `started_at` from the pre-spawn record when the
   run_id matches. Set `phase1_reused=args.reuse_phase1` and the step table from `steps_for_run`.
5. Start the daemon heartbeat thread (interval `--heartbeat-s`, default HEARTBEAT_INTERVAL_S). All status
   writes go through one `threading.Lock`, via `_write_status_if_owner()`.
6. `_write_status_if_owner()`: if `read_lock(paths).run_id != args.run_id` (superseded, e.g. a lock broken
   during a long system sleep), set `superseded`, terminate the child (as in step 9), write nothing,
   touch no markers or lock, and exit 3.
7. Spawn the child with `pipeline_argv(...)`: `stdout=PIPE, stderr=STDOUT, stdin=DEVNULL, text=True,
   encoding="utf-8", errors="replace", bufsize=1, cwd=ROOT_DIR, env={**os.environ,
   "PYTHONIOENCODING":"utf-8", "PYTHONUTF8":"1", "PYTHONUNBUFFERED":"1"}`. On Windows add the
   creationflags from 4.6. No `shell=True`. Record `child_pid`.
   `-u` / PYTHONUNBUFFERED matter because pipeline.py uses plain `print()`, which block-buffers when piped.
   Without them the markers would only arrive at exit.
8. For each line: write it to the log, append it to `deque(maxlen=20)`, and:
   - if `phase1_reused` and `PHASE1_FALLBACK_MARKER in line`: switch the table to `PIPELINE_STEPS_FULL`,
     set `phase1_reused=False`, `step_index=0`, `step_total=9`, `step_started_at=now`, and write status now;
   - else `new = advance_step(line, steps, current)`. If it changed, update `step_index`, `label` and
     `step_started_at` and write status now.
   The heartbeat thread keeps writing `heartbeat_at` during silent stretches (ML training).
9. Timeout: the heartbeat thread checks elapsed time against `max_runtime_s`. When exceeded: `proc.terminate()`,
   wait 10 s, then `proc.kill()`, and set the error to `"Timed out after N s"`.
10. After `proc.wait()`, the **order is mandatory**: day state or marker first, status second, lock last.
    - returncode 0 and no timeout:
      - full: `write_done_markers({run_date, local_today()})`
      - phase1: `record_phase1_success(run_date)` (and for today if different)
      - then status done (`step_index = step_total-1`, label "Ranking complete", finished_at, returncode 0),
        then `release_lock`. Exit 0.
    - otherwise:
      - full: `record_full_failure(run_date)`; phase1: no day-state change
      - then status failed (error = reason + last 20 lines, at most 1000 chars), then `release_lock`. Exit 1.
11. Unexpected wrapper exception: best-effort failure path (`f"Wrapper error: {type(exc).__name__}: {exc}"`). Exit 1.
12. POSIX: a SIGTERM handler terminates the child and runs the failure path. Windows relies on PID-based staleness.

### 6.10 Fast rank after a full run (existing behaviour, kept)

A Phase-1 run (Fast Rank button, page 6 Re-run, or the fallback) overwrites `data/ranking_cache.parquet`
with no-ML rankings. That replaces any ML-based rankings from earlier the same day. This is what the
Fast Rank button already does today. The Fast Rank and page 6 Re-run buttons get a help text that says
so (7.6, 7.7). The done marker is not removed, so no new auto run is triggered.

### 6.11 Two tabs

O_EXCL guarantees one spawn. The re-validation under the lock (6.5) means a tab with an out-of-date
decision cannot start a second full run or a second fallback.

---

## 7. Dashboard behaviour

### 7.1 `render_pipeline_progress()`

```
try: res = maybe_start_daily_run(); st.session_state["_plbar_last_start_reason"] = res.reason
except Exception: log
paths = get_run_paths()
status = read_status(paths)
if status and status["state"] == "done" and status["run_id"] not in _CLEARED_RUN_IDS:
    _maybe_clear_caches_for(status["run_id"])     # covers runs that finished while no tab was open
view = get_run_view(paths)
if view.state == "idle": return                   # no fragment, so no polling
_progress_fragment()
```

### 7.2 `get_run_view`

- No or invalid status -> idle.
- running: a lock with the same run_id that is alive -> running; no lock, or a lock with another run_id -> idle;
  a lock with the same run_id that is dead -> failed ("Run stopped responding"; display only, cleanup happens in `break_stale_lock`).
- done: `now - finished_at < DONE_DISMISS_S` -> done, else idle.
- failed: `now - finished_at < FAILED_DISMISS_S` -> failed, else idle.
- `progress = progress_fraction(step_index, step_total, state)`.

### 7.3 `_progress_fragment()` (st.fragment, run_every="2s")

```
view = get_run_view(...)
if view.state != "idle": st.session_state["_plbar_seen_run_id"] = view.run_id
if view.state == "done": _maybe_clear_caches_for(view.run_id)
if view.state == "idle":
    seen = st.session_state.get("_plbar_seen_run_id")
    if seen and st.session_state.get("_plbar_rerun_done_for") != seen:
        st.session_state["_plbar_rerun_done_for"] = seen
        st.rerun(scope="app")   # page shows fresh data; render sees idle; polling stops;
                                # this full run also calls maybe_start_daily_run (fallback trigger)
    return
st.markdown(_bar_html(view), unsafe_allow_html=True)
```
There is at most one app rerun per run_id per session, so it cannot loop. Each poll costs one status
read, one lock read, and a `pid_alive` call. Accepted limitation: an idle tab does not see a run started
from another tab until its next script run (idle tabs do not poll, by requirement).

### 7.4 Cache clear and cache keys

`_maybe_clear_caches_for(run_id)`: with `_CLEAR_LOCK` held, if the run_id has not been seen, call
`st.cache_data.clear()` and add it to the set. This happens once per server process per successful run.
To avoid a stale entry being re-cached by a computation that was already in flight, pages 6 and 11 key
their cached functions on `ranking_cache_version()` (the parquet mtime). The pipeline writes the parquet
before it exits, and `done` is written after the child exits.
- Page 6: `_compute_rankings(top_n, include_ml, include_indicators, custom, ranking_cache_version: float)`, called with `ranking_cache_version()`; the body ignores the value.
- Page 11: `_load_top_picks(n: int = 30, ranking_cache_version: float = 0.0)`, called as `_load_top_picks(top_n, ranking_cache_version())`.

### 7.5 Bar appearance (`_bar_html`)

- One `<div class="asa-plbar">` with a `<style>` block: `position: fixed; top: 3.75rem; left: 50%;
  transform: translateX(-50%); width: min(560px, 92vw); z-index: 999991; pointer-events: none;
  border-radius: 6px; padding: 6px 12px; font-size: 0.85rem;` semi-opaque dark background, and a thin
  track whose fill width is `round(progress*100)%`. It sits below Streamlit's header, so the menu stays
  usable, and `pointer-events: none` means it never blocks clicks.
- Running text: `"{prefix}: {label} - step {i+1}/{n} - {pct}% - {elapsed} elapsed"`.
  - Prefix: trigger auto -> `"Daily full analysis"`; auto_fallback -> `"Fallback fast rank (full analysis failed)"`;
    button full -> `"Full analysis"`; button phase1 -> `"Fast rank"`.
  - Full with `phase1_reused` adds `" (ML only, reusing today's Phase 1)"` after the prefix.
    `n` is 8 for Phase-2-only and 9 for full. If the pipeline falls back, the next poll shows `n = 9`.
  - `elapsed` comes from `started_at` and is computed in the UI on each poll (no extra writes). It keeps
    the bar visibly moving during silent ML training. If the current step has run for 60 s or more,
    append `" (this step {step_elapsed})"` from `step_started_at`.
  - The fill has a pure-CSS `@keyframes` pulse or stripe while running (no JS).
  - For trigger auto only, add a muted second line: "You can keep using the dashboard."
- Done: `"Ranking updated - refreshing data"` (green accent). Failed: `"{prefix} failed: {first error line, 140 chars max}. See Settings page."` (red accent).
  For an auto full failure while `fallback_started` is false today, add " A fast rank will run instead."
- **Every dynamic string is passed through `html.escape()`.** No emojis.

### 7.6 `dashboard/app.py` changes

- Delete `_PIPELINE_STEPS_PHASE1`, `_PIPELINE_STEPS_FULL`, `_advance_step` and `_run_pipeline`.
- Imports: `start_pipeline_run`, `DEFAULT_POOL`, `DEFAULT_TOP` from the launcher; `get_run_paths`,
  `is_run_active`, `read_status`, `read_log_tail` from run_state; `render_pipeline_progress`.
- Add `render_pipeline_progress()` right after `render_market_clock()`.
- `KEY_SPEC`: append the `AUTO_RANK_ENABLED` entry (4.8).
- Pipeline section intro: add a caption saying "The full pipeline runs automatically in the background
  once per day on the first dashboard open (set AUTO_RANK_ENABLED=false above to turn this off). If it
  fails, a fast rank runs instead."
- Number inputs default to `DEFAULT_POOL` and `DEFAULT_TOP` (same bounds).
- `active = is_run_active(get_run_paths())`, called after `reap_children()`. Both buttons get `disabled=active`.
- Fast Rank button (manual only): `start_pipeline_run("phase1", int(pipe_pool), int(pipe_top), trigger="button")`.
  Help: "Quick no-ML ranking (~30s). Replaces the current rankings, including today's ML rankings if any."
- Full button: `start_pipeline_run("full", ...)`. Help: "Phase 1 + ML training. If a fast rank already ran
  today, only the ML part runs."
- On `res.started`: `st.success(res.message)` then `st.rerun()`. Otherwise `st.warning(res.message)`.
  Never block, and never call `st.cache_data.clear()` here.
- Below the buttons: a "Last run" caption from `read_status()` (state, mode, trigger, reuse, local-time
  start and finish, error if failed), plus `st.expander("Last run log")` showing `read_log_tail(...)` in `st.code`.
- `_run_scraper` and all other sections are unchanged.

### 7.7 Page 6 "Re-run Ranking"

```
active = is_run_active(get_run_paths())     # after reap_children()
if st.button("Re-run Ranking", type="primary", disabled=active and not custom_tickers.strip(),
             help="Starts a fast (no-ML) ranking in the background. Replaces the current rankings."):
    if custom_tickers.strip():
        st.cache_data.clear()                # existing behaviour for custom lists, unchanged
    else:
        res = start_pipeline_run("phase1", DEFAULT_POOL, DEFAULT_TOP, trigger="button")
        if res.started: st.rerun()
        else: st.warning(res.message)
```
Keep the existing label text and icon exactly (only the parameters and handler change). The rest of
the page (cached load from the ranking cache) is unchanged apart from 7.4.

---

## 8. Where the component call goes

Insert these 2 lines right after the existing `render_market_clock()` line (before any `st.stop()`):
```python
from dashboard.components.pipeline_progress import render_pipeline_progress
render_pipeline_progress()
```
22 files: `dashboard/app.py`; `dashboard/pages/1_data_overview.py`, `2_morning_report.py`,
`2a_stock_summary.py`, `3_long_term_forecast.py`, `4_short_term_signals.py`, `5_ml_predictions.py`,
`6_stock_rankings.py`, `7_news_sentiment.py`, `8_political_insider.py`, `9_options_flow.py`,
`10_portfolio.py`, `11_sector_analysis.py`, `12_market_regime.py`, `13_backtesting.py`, `14_calendar.py`,
`15_watchlist.py`, `16_peer_comparison.py`, `17_trading.py`, `18_trade_journal.py`, `19_alerts.py`,
`20_auto_trades.py`. Excluded: none. (`dashboard/pages/__init__.py` is not a page.)

---

## 9. Detached spawn (`spawn_detached`)

Common: `subprocess.Popen(argv, cwd=str(cwd), stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL,
close_fds=True, env={**os.environ, "PYTHONIOENCODING":"utf-8", "PYTHONUTF8":"1"})`. `argv[0]` is
`sys.executable`. No `shell=True`. The parent never holds pipes. Append the Popen to `_SPAWNED`.

POSIX: `start_new_session=True`. The new session survives Ctrl-C and a closed terminal. Reap through
`reap_children()` before liveness checks (6.3). If the Streamlit server exits, init reaps the child.

Windows: `creationflags = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB`.
If that raises `OSError` with `winerror == 5`, retry once without BREAKAWAY. **Never `DETACHED_PROCESS`**:
a console-less wrapper would make the `python.exe` grandchild open a visible console window.
`CREATE_NO_WINDOW` gives the wrapper a hidden console that the child inherits. Because the wrapper is
not attached to the .bat console, closing the cmd window does not kill it. Use `getattr(subprocess, NAME, 0)`
so the module still imports on POSIX.

---

## 10. Edge cases

| Case | Behaviour |
|---|---|
| App or browser closed mid-run | The detached wrapper finishes and records the outcome; the next page load shows the result and clears the cache once. |
| Streamlit server killed mid-run | Same; the run is not tied to the server. |
| Reboot / wrapper killed | `pid_alive` is False, so stale immediately. The next load breaks the lock (a full run counts as a failure), and the state machine continues (fallback if applicable). |
| Long silent ML step | The pid is alive and the heartbeat continues, so the run is not stale. The bar keeps moving through elapsed time, step time and the pulse. |
| Full run over 4 h | Killed by the wrapper; failed; full_failures +1, then the fallback. |
| Auto full fails | FULL_FAILED; a single Phase-1 fallback runs automatically (6.7), then buttons only. |
| Fallback fails | Buttons only today. The bar shows the failure; the Settings page shows the log. |
| Fast rank done, then full button | Reuses the artifact: Phase-2-only run, 8 steps, prefix notes "ML only". |
| Artifact from yesterday / prices rescraped since / ticker source changed / missing / corrupt | The launcher picks a normal 9-step full run. If the condition only appears between the launcher check and pipeline start, the pipeline prints the fallback marker, runs Phase 1, and the bar switches to 9 steps. |
| Full run fails in Phase 2 | The Phase 1 artifact from that run stays valid, so a later full button run reuses it. |
| Fast rank after a full ML run | Overwrites the ML rankings (existing behaviour, 6.10). The marker is kept, so no auto run follows. |
| AUTO_RANK_ENABLED=false | No automatic run of any kind. Buttons work. A blank value means true. An invalid value is treated as disabled, with a warning in the log. |
| Spawn fails | spawn_error; full -> full_failures +1; fallback -> already claimed, so buttons only. Lock released. |
| PID reused after a crash | Caught by the 15-min no-heartbeat guard. Until then the bar shows running. Accepted as rare. |
| Laptop sleep | The pid stays alive. If another tab broke the lock in the stale window and started a new run, the old wrapper sees it is superseded and exits 3 without writing. |
| Zombie wrapper (POSIX) | `reap_children()` runs before liveness checks. |
| Two tabs, first load | O_EXCL plus re-validation under the lock: exactly one run. |
| Two tabs breaking the same stale lock | Breaker mutex plus re-check: exactly one break and one failure count. |
| Button while active | Disabled, and refused server-side ("A ranking run is already in progress."). No queue. |
| No scraped data | Auto: `no_data`, nothing happens. Buttons: refused with a message. |
| Midnight rollover | Full success marks both days; the artifact's run_date is the Phase-1 finish day. |
| Status read during a write | `os.replace` is atomic; the Windows writer retries; a bad read shows idle for one poll. |
| Unicode output ("…") | The child uses UTF-8 env vars; the wrapper reads utf-8 with errors="replace"; the log is utf-8. |
| Missing run_cache/logs dirs | Created by `get_run_paths` / `save_phase1_result`. |
| Corrupt or empty lock | run_id None, aged by mtime, stale after SPAWN_GRACE_S. |

---

## 11. Test list (tester)

Rules: use `tmp_path` for every data dir, inject `now`/`today`, and never run the real pipeline, ML
or network, or touch the repo's `data/`. Pass settings as `types.SimpleNamespace(data_dir=..., raw_daily_dir=...,
ticker_source="sp500", auto_rank_enabled=True)`.

`test_steps.py`
1. `advance_step`: matches in order; never goes backwards; can skip ahead; a non-match returns current.
2. The real phase-1 lines reach index 4: `"[Phase 1] Fast-ranking 500 tickers (no ML) ..."`, `"[Phase 1] Done in 12s -- pool: ..."`, `"[Phase 1 Only] Collecting price targets ..."`, `"[Cache] Ranking cache saved -> x"`, `"  AutoStockAnalyzer -- Two-Phase Pipeline Results"`.
3. The full sequence, including the "…" line, reaches index 8.
4. The Phase-2-only sequence (reuse line, then "[Phase 2] Training ML models on 100 candidates ...", ...) reaches index 7; `PHASE2_ONLY[1:]` markers == `FULL[2:]` markers.
5. `steps_for_run` for all combinations; a bad mode raises ValueError.
6. `progress_fraction` edge cases.
7. The PHASE1/FULL tables equal the old app.py lists (hard-coded expected values).
8. Neither `PHASE1_REUSE_MARKER` nor `PHASE1_FALLBACK_MARKER` contains any marker from any table.

`test_phase1_store.py`
9. Save, then check: usable; `pool_tickers == first pool tickers in order`; a -inf entry is saved with score null and keeps its position.
10. Not usable for each reason: missing, unreadable JSON, wrong schema, run_date yesterday, ticker_source changed, empty entries, and a raw parquet mtime newer than created_at (set with `os.utime`).
11. pool > ranked_count returns all entries (usable). pool None gives pool_tickers None.
12. Atomic write: no `.tmp` files are left behind.

`test_run_state.py`
13. Lock acquire once; second acquire False; release with the wrong run_id is a no-op; the right one unlinks.
14. Lock race: 16 threads with a Barrier give exactly one True; 4 processes (multiprocessing spawn context) give exactly one True.
15. Liveness: no status, fresh lock -> alive; past SPAWN_GRACE_S -> dead. Running with `pid=os.getpid()`, started 3 h ago, heartbeat 10 min old -> **alive**. Heartbeat older than PID_REUSE_GUARD_S -> dead. Dead pid (monkeypatched) with a heartbeat 1 s old -> dead. pid None -> grace rule. Terminal status: alive while the pid is alive or within grace.
16. `break_stale_lock`: stale full run -> status failed and full_failures +1; stale phase1 run -> no day-state change; live lock -> False and nothing changes; fresh breaker -> False; old breaker -> proceeds.
17. 8 threads break the same stale full lock -> full_failures increases by exactly 1.
18. A corrupt or empty lock is aged by mtime.
19. `write_json_atomic`/`read_json` round-trip; invalid inputs give None; `read_status` rejects a bad schema or state.
20. Done markers for two days; `has_done_marker`.
21. Day state: defaults when missing; `record_full_failure` twice gives 2; `record_fallback_started`; `record_phase1_success`; a corrupt file reads as defaults.
22. `cleanup_old_files` (markers, day-state files, logs).
23. `get_run_view` for every branch in 7.2.
24. `new_run_id` format `^\d{8}T\d{6}Z-[0-9a-f]{6}$`.
25. `pid_alive`: self True; None/0 False; on POSIX a reaped child's pid is False.

`test_launcher.py` (fake spawn records argv and returns 4242)
26. `decide_auto_action`: a full truth table over enabled, marker, lock_alive, has_data, full_failures (0/1/2), fallback_started and phase1_success, matching 4.4.
27. `maybe_start_daily_run`, fresh day -> started, mode full, argv has `--mode full --pool 100 --top 50`, no `--reuse-phase1`; status pid == 4242.
28. With a done marker -> already_done, no spawn. Disabled -> disabled, no spawn. No data -> no_data.
29. full_failures 1, nothing else -> started with mode phase1, trigger auto_fallback; day state now `fallback_started` true. A second call (after releasing the lock) -> buttons_only.
30. full_failures 1 and phase1_success true -> buttons_only.
31. Live lock -> running. Stale full lock -> broken (full_failures +1) and then, since full_failures is now 1, the fallback starts.
32. Re-validation: monkeypatch `try_acquire_lock` so that just before it returns True the day state already has `fallback_started=True` -> the result is not started (buttons_only) and the lock is released.
33. Button `start_pipeline_run("full")` with a usable artifact -> `--reuse-phase1` in argv; status phase1_reused true, step_total 8. With a stale artifact -> no flag, step_total 9.
34. A button run ignores the done marker and full_failures 2 (starts), but refuses on a live lock and on no data.
35. Spawn raises OSError: full -> spawn_error, lock released, status failed, full_failures +1. Fallback -> fallback_started stays true, full_failures unchanged.
36. The pid fill does not overwrite a status that the fake spawn already wrote with pid 999.
37. ValueError on a bad pool/top/mode/trigger, with no files created.
38. `spawn_detached` flags (monkeypatched Popen): POSIX `start_new_session=True`; simulated Windows -> NEW_PROCESS_GROUP|NO_WINDOW|BREAKAWAY, never DETACHED_PROCESS, retries without BREAKAWAY on winerror 5; never `shell=True`; argv[0] is `sys.executable`.
39. The settings factory raises (any exception, e.g. a monkeypatched `get_settings` that raises) -> `maybe_start_daily_run` returns disabled and does not raise. (An invalid `AUTO_RANK_ENABLED` no longer raises; it is covered by item 61.)

`test_pipeline_job.py` (a fake script in tmp_path prints the chosen markers, including "…", and exits with a chosen code; use `--pipeline-script`, `--data-dir`, `--heartbeat-s 0.5`)
40. Full success: done, step 8/9, done markers, lock removed, log contains "…", child_pid set, `step_started_at` changes between steps.
41. Phase1 success: done; no done marker; day state phase1_success true.
42. Full with `--reuse-phase1`, the fake prints the reuse line and the Phase-2 markers -> 8-step table, done at index 7, `phase1_reused` true.
43. Full with `--reuse-phase1`, the fake prints the FALLBACK marker and then the full markers -> table switches to 9 steps, `phase1_reused` false, done at index 8.
44. Full failure: failed, error has the tail, full_failures 1, no marker. Phase1 failure: no day-state change.
45. Order check: monkeypatched `release_lock` asserts the marker or day state and the terminal status already exist.
46. Not owner -> exit 2, nothing written. `--reuse-phase1` with phase1 -> exit 2.
47. Superseded (lock replaced mid-run) -> exit 3, child terminated, nothing written.
48. Timeout (`--max-runtime-s 2`, sleeping fake) -> failed with a timeout error.
49. Silent child (3 s sleep) -> `heartbeat_at` advances.
50. Midnight: `local_today` monkeypatched to differ at finish -> two done markers.

`test_pipeline_skip_phase1.py` (import `cli.pipeline`; monkeypatch `get_tickers`, `rank_tickers`, `save_ranking_cache`, `_current_price`, `_forecast_target`, `_ml_price_target` on that module; inject a fake `src.ml.runner` module via `monkeypatch.setitem(sys.modules, "src.ml.runner", fake)` exposing `run_tickers` and `predict_latest`; settings via a monkeypatched `get_settings` returning a namespace on tmp_path)
51. A normal full run writes `phase1_result.json` whose entries match the fake phase1 order, including a -inf entry.
52. A phase1-only run also writes the artifact.
53. `skip_phase1=True` with a usable artifact: the fake `rank_tickers` is called only once, with `include_ml=True` and the artifact's first `pool` tickers; stdout contains `PHASE1_REUSE_MARKER`; no Phase-1 call.
54. `skip_phase1=True` with a stale or missing artifact: stdout contains `PHASE1_FALLBACK_MARKER` and then the normal Phase-1 markers; `rank_tickers` called with `include_ml=False` then `True`; the artifact is rewritten.
55. `phase1_only=True, skip_phase1=True` raises ValueError. argparse rejects `--phase1-only --skip-phase1` (SystemExit).
56. An artifact save failure (monkeypatch `save_phase1_result` to raise) does not fail the run, and a warning is printed.
57. Without the new flag, the printed step markers are unchanged (compare against the steps tables).

Dashboard (pure functions, always run):
58. `_bar_html` escapes `<script>` in label and error.
59. `_bar_html`, running full, started 7m12s ago -> contains "step 3/9" and "7m 12s elapsed"; reused -> "step 2/8" and "ML only"; auto_fallback prefix text.
60. `_maybe_clear_caches_for` clears exactly once per run_id (monkeypatch `st.cache_data.clear`).

Settings:
61. `Settings(auto_rank_enabled="")` gives True; `"false"` gives False; the default is True (use `monkeypatch.delenv` / `_env_file=None`). Invalid value (e.g. `"maybe"`) -> `auto_rank_enabled` False, warning logged (check with `caplog`; the message does not contain the bad value), no exception.

Manual checks to report: Windows .bat launch shows no console flash; the run survives closing the cmd
window; the bar's appearance on a few pages; the page 6 Re-run flow.

---

## 12. Coder task order (one focused change each; pass this plan path every time)

1. `pyproject.toml`: streamlit pin to `>=1.37.0` (4.9). Independent.
2. `config/settings.py` + `.env.example`: `AUTO_RANK_ENABLED` (4.8). Independent.
3. `src/pipeline_runner/__init__.py` + `steps.py` (4.1).
4. `src/pipeline_runner/run_state.py` (4.3). Depends on 3.
5. `src/pipeline_runner/phase1_store.py` (4.2). Depends on 4 (atomic writer).
6. **`cli/pipeline.py`**: artifact write and `--skip-phase1` (4.5). Depends on 3 and 5. The only change to this file in the plan.
   Its tests (`test_pipeline_skip_phase1.py`, items 51-57) should be written right after this task (tester, or coder if the main session prefers) so the change is covered before the wrapper depends on it.
7. `src/pipeline_runner/launcher.py` (4.4, 6.5-6.7). Depends on 2-5.
8. `cli/pipeline_job.py` (4.6, 6.9). Depends on 3-5.
9. `dashboard/components/pipeline_progress.py` (4.7, 7.1-7.5). Depends on 4 and 7.
10. `dashboard/app.py` (7.6). Depends on 2, 7, 9. The step tables are removed in this same task.
11. Pages 6 and 11 (7.4, 7.7, plus the bar call). Depends on 7 and 9.
12. The other 19 pages: the 2-line bar call (8).
13. Remaining tests (section 11).

Implementation notes (from coder reports on tasks 3 and 5): `steps.py` defines local `RunMode`/`RunState`
Literal aliases, and `run_state.py` must import them from `steps.py` (not redefine them).
`phase1_store.py` currently has a private `_write_json_atomic`; once `run_state.py` lands it should be
swapped for `run_state.write_json_atomic`. Reviewer to check this at review.

Pipeline: reviewer, then **data-flow-checker (required)**. It should check:
pipeline.py stdout markers -> wrapper -> status/lock/day-state/marker files -> run_state -> component/app.py;
the Phase 1 artifact pipeline.py -> phase1_store -> pipeline.py `--skip-phase1` and the launcher;
ranking parquet -> page 6/11 cache keys; and `.env` -> settings -> launcher.
Then tester, then security-auditor before push (`html.escape`, no `shell=True`, no private paths).

---

## 13. Resolved questions and remaining notes

Resolved by the user (2026-10-11):
- Q1 Page 6 Re-run: starts a manual Phase-1 run through the launcher when no custom tickers are typed (7.7).
- Q2 Button while active: refused, no queue.
- Q3 Failure accounting: only FULL failures count (any trigger, including stale). Superseded by the Q7 state machine (6.7).
- Q4 Streamlit pin: bumped to >=1.37.0 as task 1.
- Q5 Opt-out: `AUTO_RANK_ENABLED`, default true, disables only the automatic start (4.8).
- Q6 Docs: the main session handles ARCHITECTURE.md and DOCUMENTATION.md.
- Q7 Retry: no auto retry of the full run. A failed auto full run triggers one automatic Phase-1 fallback, then buttons only (6.7).
- Q8 Lower CPU priority: accepted.
- New requirement: a full run after a same-day fast rank runs only Phase 2 (4.2, 4.5, 6.8).

Notes for the main session (decisions the plan made, not open questions for the coder):
- **N1 (deviation from the coordinator's suggestion):** a pool/top mismatch does NOT refuse reuse of the
  saved Phase 1. The artifact holds the complete ordered ranking, so any pool slice is exact (6.8). If the
  user still wants a strict match, change `check_phase1_reusable` to also store and compare `pool`. That is a one-line spec change.
- **N2:** a fast rank (any trigger) overwrites same-day ML rankings in the ranking cache. This is existing
  behaviour, now stated in the button help text (6.10). If the user wants a fast rank never to overwrite a
  same-day full result, that is a new change to `cli/pipeline.py` and needs its own plan revision.
- **N3:** `cli/pipeline.py` is changed in exactly one task (task 6). Everything else in that file stays as it is.

Out of scope, reported only: `launch_dashboard.bat` line 3 hard-codes a private drive path in its `cd /d`
line, which matches an entry on the CLAUDE.md Private strings list. The security-auditor will flag it.

---

## Revisions

- 2026-10-11: initial plan (architect).
- 2026-10-11: user decision change. The daily auto-run became the FULL pipeline (`--no-deep`, full step markers, pool 100 / top 50); the Phase-1 button became manual only. Staleness became PID-liveness first, with a 15-min heartbeat guard. Added `child_pid` and `step_started_at`, and elapsed time plus a pulse on the bar. Lower CPU priority.
- 2026-10-11: user answered section 13. Changes:
  - Page 6 Re-run now starts a Phase-1 run.
  - Streamlit pin task added.
  - `AUTO_RANK_ENABLED` setting added (settings.py, .env.example, app.py KEY_SPEC).
  - The "one auto retry of the full run" rule was replaced by a single automatic Phase-1 fallback, then buttons only. The `.fail_rank_<day>` counter was replaced by `auto_rank_<day>.json` day state (full_failures, fallback_started, phase1_success). Added the pure `decide_auto_action`, re-validation under the lock, and trigger `auto_fallback`.
  - New requirement: Phase-1 reuse via the new `phase1_store.py`, the `data/run_cache/phase1_result.json` artifact, and `cli/pipeline.py --skip-phase1` (the only pipeline.py change), with an 8-step Phase-2-only table and an in-run switch to 9 steps on fallback.
  - Status field `phase1_reused` added. Tests renumbered (1-61). Coder task order updated (13 tasks).
- 2026-10-11: correction to match implemented `config/settings.py`. An invalid `AUTO_RANK_ENABLED` now gives False with a logged warning instead of raising (4.8, 6.6 comment, tests 39 and 61). Added implementation notes under section 12 (RunMode/RunState aliases live in steps.py; phase1_store `_write_json_atomic` to be swapped for `run_state.write_json_atomic`).
