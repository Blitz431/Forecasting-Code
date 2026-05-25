# Security Audit — AutoStockAnalyzer

**Date:** 2026-04-16
**Scope:** Full codebase — `src/`, `dashboard/`, `cli/`, `config/`, root.
**Deployment model reviewed:** local single-user Streamlit dashboard + CLI. No HTTP endpoints, no authentication, no multi-tenant surface.

---

## 1. Requested items

| # | Request | Status | Notes |
|---|---------|--------|-------|
| 1 | Rate limiting on login / all endpoints | **N/A** | No HTTP endpoints exist. Streamlit pages run in-process; the only network callers are outbound (Alpaca, FRED, yfinance). See §3. |
| 2 | Scan for hardcoded API keys / tokens / passwords | **PASS** | No hardcoded secrets found. All credentials load from `.env` via `config/settings.py` (`pydantic-settings`). New **Settings page** (`dashboard/pages/0_settings.py`) lets you enter keys and export a redacted `.docx` inventory. |
| 3 | Move all sensitive data to env vars, nothing in frontend or git | **PASS** | Already the pattern. `.env` is gitignored; `.env.example` (placeholders only) is tracked. `.gitignore` now also blocks `*.key`, `*.pem`, `secrets.*`, `credentials.*`. |
| 4 | Sanitize user inputs, reject oversize/malformed | **DONE** | New `src/utils/input_sanitize.py` with `clean_ticker`, `clean_ticker_list`, `clean_text`. Wired into pages 5 (stock rankings), 12 (calendar), 14 (watchlist), 16 (peer comparison). Ticker regex `^[A-Z][A-Z0-9.\-]{0,9}$`, list cap 50, text cap 500 chars, control chars stripped. |
| 5 | Full security audit | **THIS DOCUMENT** | See §3–§5 below. |

---

## 2. Bug fixes bundled in this pass

- **Stock Rankings chart** — `ranking_bar` duplicated the `yaxis` kwarg via `_LAYOUT_BASE`. Added `_merged_layout()` helper in `dashboard/components/charts.py` and applied to `ranking_bar`, `score_breakdown`, `feature_importance`, `earnings_timeline`.
- **Options Flow PCR log** — fixed malformed f-string (`{pcr:.3f if ... else 'N/A'}`) in `src/options/options_data.py`.
- **Backtesting `add_vline`** — plotly failed on string x-values when computing annotation position. Now passes epoch-ms int (`ts.value // 10**6`).
- **Morning Report PDF** — fpdf2 Helvetica is latin-1. Added `_pdf_safe()` helper in `src/reports/morning_report.py` that maps em/en-dashes, smart quotes, ellipsis, bullets, NBSPs to ASCII equivalents. Applied to every `cell`/`multi_cell` call. Title string `—` replaced with `-`.

---

## 3. Threat-model summary

Application is **single-user, local-only**. Attack surface is limited to:

1. **Local files** — `.env`, parquet caches, model pickles, SQLite-like stores. Protected by OS file permissions.
2. **Outbound HTTP** to Alpaca, FRED, yfinance, feedparser, Discord, Discord. No inbound.
3. **Third-party dependencies** — CVE exposure via `pip install` surface.

Threats that **do not apply**:
- Multi-user auth, session hijacking, CSRF, rate-limited login bruteforce (no login).
- XSS / SQLi (no user-submitted HTML, no SQL).
- Server-side request forgery (no server).
- Rate limiting on HTTP endpoints (no endpoints).

---

## 4. Remaining risks & recommendations

| Risk | Severity | Recommendation |
|------|----------|---------------|
| **Pickle/joblib model loading** (`src/ml/models/*.pkl`, `*.joblib`) | Medium | `pickle.load` is code-execution if a model file is swapped by an attacker. Mitigation: models are produced locally and the directory is gitignored — acceptable for single-user, but document. Consider moving to ONNX or checksum-verifying before load for shared deployments. |
| **Dependency CVEs** | Medium (time-dependent) | Run `pip-audit` quarterly. Notable dep surface: torch, transformers, alpaca-trade-api. |
| **Discord/Discord webhook URLs in `.env`** | Low | Treat as sensitive. Do not paste logs containing webhook responses publicly. |
| **Streamlit cache poisoning on shared machines** | Low | `@st.cache_data` stores outputs under `~/.streamlit`. If multiple OS users share the same account, anything computed here is visible. Not applicable for intended single-user use. |
| **FPDF input strings** | Low (now mitigated) | `_pdf_safe` scrubs non-latin-1 chars. If you ever need full Unicode, ship a DejaVuSans TTF and register with `pdf.add_font(..., uni=True)`. |
| **Ticker input** (now mitigated) | Low | Free-text ticker fields are regex-validated. CLI args (`cli/*.py`) still accept raw strings — add `clean_ticker` there if you start taking ticker args from untrusted sources. |
| **`.env` file lives at repo root** | Low | Consider moving to `%APPDATA%/AutoStockAnalyzer/.env` on Windows for defense-in-depth against accidental `git add`. Current gitignore rules prevent commits even if it stays. |

Confirmed **no** tracked secret files:

```
git ls-files | grep -Ei '\.(env|key|pem)$'    # → empty
```

---

## 5. Re-audit triggers

Re-run a full audit if **any** of the following become true:

- You add a FastAPI / Flask / Django layer, or any HTTP endpoint.
- You enable multi-user access (auth, sessions, cookies).
- You deploy the app to a shared host, container, or cloud.
- You add file-upload handlers accepting user-supplied files.
- You start accepting ticker / free-text inputs from untrusted CLI or API callers.
- You add a database with user-controlled query fragments.

Until any of those, the posture above is considered sufficient for local use.

---

## 6. Files touched in this security pass

- `dashboard/pages/0_settings.py` *(new)* — keys UI + Word export
- `src/utils/input_sanitize.py` *(new)* — validators
- `dashboard/pages/5_stock_rankings.py` — wired sanitizer
- `dashboard/pages/12_calendar.py` — wired sanitizer
- `dashboard/pages/14_watchlist.py` — wired sanitizer
- `dashboard/pages/16_peer_comparison.py` — wired sanitizer
- `dashboard/components/charts.py` — `_merged_layout` + fixes
- `src/options/options_data.py` — PCR log fix
- `dashboard/pages/11_backtesting.py` — `add_vline` fix
- `src/reports/morning_report.py` — PDF Unicode scrub
- `.gitignore` — broadened secret patterns
- `pyproject.toml` — added `python-docx`
