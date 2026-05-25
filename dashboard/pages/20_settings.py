"""Settings & Data Management page."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import get_settings

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
st.title("⚙️ Settings & Data Management")
st.divider()

settings = get_settings()
env_file = _ROOT / ".env"

# ---------------------------------------------------------------------------#
# Helper — read / write .env
# ---------------------------------------------------------------------------#

def _read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                values[k.strip()] = v.strip()
    return values


def _write_env(values: dict[str, str]) -> None:
    lines = [
        "# Alpaca API",
        f"ALPACA_API_KEY={values.get('ALPACA_API_KEY', '')}",
        f"ALPACA_SECRET_KEY={values.get('ALPACA_SECRET_KEY', '')}",
        f"ALPACA_BASE_URL={values.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')}",
        "",
        "# FRED API",
        f"FRED_API_KEY={values.get('FRED_API_KEY', '')}",
        "",
        "# Trading mode",
        f"TRADING_MODE={values.get('TRADING_MODE', 'paper')}",
        "",
        "# Alerts",
        f"DISCORD_WEBHOOK_URL={values.get('DISCORD_WEBHOOK_URL', '')}",
        f"TELEGRAM_BOT_TOKEN={values.get('TELEGRAM_BOT_TOKEN', '')}",
        f"TELEGRAM_CHAT_ID={values.get('TELEGRAM_CHAT_ID', '')}",
        f"SMTP_HOST={values.get('SMTP_HOST', '')}",
        f"SMTP_PORT={values.get('SMTP_PORT', '587')}",
        f"SMTP_USER={values.get('SMTP_USER', '')}",
        f"SMTP_PASSWORD={values.get('SMTP_PASSWORD', '')}",
        f"ALERT_EMAIL_TO={values.get('ALERT_EMAIL_TO', '')}",
    ]
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------#
# Tab layout
# ---------------------------------------------------------------------------#

tab_scraper, tab_api, tab_thresholds, tab_paths = st.tabs(
    ["🔄 Data Scraper", "🔑 API Keys", "⚙️ Thresholds & Rules", "📁 Data Paths"]
)

# ===========================================================================
# TAB 1 — Data Scraper
# ===========================================================================
with tab_scraper:
    st.subheader("Data Scraper")
    st.markdown(
        "Run the scraper to fetch stock prices, macro data, and dividends for all S&P 500 tickers."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Daily Update (Incremental)")
        st.caption(
            "Downloads only the latest missing prices. Run this every trading day. "
            "Typically takes 2–5 minutes."
        )
        prices_only_daily = st.checkbox("Prices only (skip macro & dividends)", key="daily_prices_only")
        run_daily = st.button("▶ Run Daily Update", type="primary", use_container_width=True)

    with col_b:
        st.markdown("#### Initial / Full Backfill")
        st.caption(
            "Downloads full history from 2015 for all ~500 tickers. "
            "Run once on first setup. Takes 15–30 minutes."
        )
        prices_only_backfill = st.checkbox("Prices only (skip macro & dividends)", key="backfill_prices_only")
        run_backfill = st.button("▶ Run Full Backfill (2015→today)", use_container_width=True)

    st.divider()

    # Module-level buffers so the background thread never touches st.session_state
    if "_scraper_log_buf" not in st.session_state:
        st.session_state["_scraper_log_buf"] = []
    if "_scraper_done" not in st.session_state:
        st.session_state["_scraper_done"] = True

    def _stream_scraper(log_buf: list, done_flag: list, cmd: list[str]) -> None:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(_ROOT),
            )
            for line in proc.stdout:
                log_buf.append(line.rstrip())
            proc.wait()
            log_buf.append(
                "\n✅ Done (exit code 0)" if proc.returncode == 0
                else f"\n❌ Exited with code {proc.returncode}"
            )
        except Exception as exc:
            log_buf.append(f"\n❌ Error: {exc}")
        finally:
            done_flag[0] = True

    if "scraper_running" not in st.session_state:
        st.session_state["scraper_running"] = False
    if "scraper_log" not in st.session_state:
        st.session_state["scraper_log"] = []

    def _launch(cmd: list[str]) -> None:
        buf: list[str] = []
        done_flag = [False]
        st.session_state["_scraper_log_buf"] = buf
        st.session_state["_scraper_done_flag"] = done_flag
        st.session_state["scraper_running"] = True
        st.session_state["scraper_log"] = []
        threading.Thread(target=_stream_scraper, args=(buf, done_flag, cmd), daemon=True).start()

    if run_daily and not st.session_state["scraper_running"]:
        cmd = [sys.executable, str(_ROOT / "cli" / "scrape.py")]
        if prices_only_daily:
            cmd.append("--prices-only")
        _launch(cmd)
        st.rerun()

    if run_backfill and not st.session_state["scraper_running"]:
        cmd = [sys.executable, str(_ROOT / "cli" / "scrape.py"), "--backfill"]
        if prices_only_backfill:
            cmd.append("--prices-only")
        _launch(cmd)
        st.rerun()

    # Drain the thread buffer into session_state on each rerun
    if st.session_state["scraper_running"]:
        buf = st.session_state.get("_scraper_log_buf", [])
        done_flag = st.session_state.get("_scraper_done_flag", [True])
        st.session_state["scraper_log"] = list(buf)
        if done_flag[0]:
            st.session_state["scraper_running"] = False
        else:
            st.info("Scraper is running… page auto-refreshes every 3 s.")

    log_area = st.empty()
    if st.session_state["scraper_log"]:
        log_text = "\n".join(st.session_state["scraper_log"])
        log_area.code(log_text, language="bash")
        if not st.session_state["scraper_running"]:
            if st.button("Clear Log"):
                st.session_state["scraper_log"] = []
                st.rerun()

    if st.session_state["scraper_running"]:
        time.sleep(3)
        st.rerun()

    # Data summary
    st.divider()
    st.subheader("Current Data Summary")
    c1, c2, c3, c4 = st.columns(4)

    def _count(p: Path) -> int:
        return len(list(p.glob("*.parquet"))) if p.exists() else 0

    c1.metric("Daily Price Files", _count(settings.raw_daily_dir))
    c2.metric("Quarterly Files", _count(settings.raw_quarterly_dir))
    c3.metric("Forecast Files", _count(settings.forecasts_dir))
    c4.metric("Macro Series", _count(settings.raw_macro_dir))


# ===========================================================================
# TAB 2 — API Keys
# ===========================================================================
with tab_api:
    st.subheader("API Keys & Credentials")
    st.caption(f"Stored in `{env_file}`")

    env = _read_env()

    with st.form("api_keys_form"):
        st.markdown("#### Alpaca (Trading)")
        alpaca_key = st.text_input("API Key", value=env.get("ALPACA_API_KEY", ""), type="password")
        alpaca_secret = st.text_input("Secret Key", value=env.get("ALPACA_SECRET_KEY", ""), type="password")
        alpaca_url = st.selectbox(
            "Base URL",
            ["https://paper-api.alpaca.markets", "https://api.alpaca.markets"],
            index=0 if env.get("ALPACA_BASE_URL", "").startswith("https://paper") else 1,
        )

        st.markdown("#### FRED (Macro Data)")
        fred_key = st.text_input("FRED API Key", value=env.get("FRED_API_KEY", ""), type="password")

        st.markdown("#### Trading Mode")
        trading_mode = st.selectbox(
            "Mode",
            ["paper", "live"],
            index=0 if env.get("TRADING_MODE", "paper") == "paper" else 1,
        )
        if trading_mode == "live":
            st.warning("Live trading mode — real money will be used.")

        st.markdown("#### Alerts (optional)")
        discord_url = st.text_input("Discord Webhook URL", value=env.get("DISCORD_WEBHOOK_URL", ""))
        tg_token = st.text_input("Telegram Bot Token", value=env.get("TELEGRAM_BOT_TOKEN", ""), type="password")
        tg_chat = st.text_input("Telegram Chat ID", value=env.get("TELEGRAM_CHAT_ID", ""))
        smtp_host = st.text_input("SMTP Host", value=env.get("SMTP_HOST", ""))
        smtp_port = st.text_input("SMTP Port", value=env.get("SMTP_PORT", "587"))
        smtp_user = st.text_input("SMTP User", value=env.get("SMTP_USER", ""))
        smtp_pass = st.text_input("SMTP Password", value=env.get("SMTP_PASSWORD", ""), type="password")
        alert_email = st.text_input("Alert Email To", value=env.get("ALERT_EMAIL_TO", ""))

        if st.form_submit_button("💾 Save API Keys", type="primary"):
            _write_env({
                "ALPACA_API_KEY": alpaca_key,
                "ALPACA_SECRET_KEY": alpaca_secret,
                "ALPACA_BASE_URL": alpaca_url,
                "FRED_API_KEY": fred_key,
                "TRADING_MODE": trading_mode,
                "DISCORD_WEBHOOK_URL": discord_url,
                "TELEGRAM_BOT_TOKEN": tg_token,
                "TELEGRAM_CHAT_ID": tg_chat,
                "SMTP_HOST": smtp_host,
                "SMTP_PORT": smtp_port,
                "SMTP_USER": smtp_user,
                "SMTP_PASSWORD": smtp_pass,
                "ALERT_EMAIL_TO": alert_email,
            })
            st.success("Saved to .env — restart the app for changes to take effect.")


# ===========================================================================
# TAB 3 — Thresholds & Rules
# ===========================================================================
with tab_thresholds:
    st.subheader("Indicator Thresholds & Trading Rules")
    st.caption("These values are read-only here — edit `config/settings.py` to change defaults.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**RSI**")
        st.write(f"Overbought: `{settings.rsi_overbought}`")
        st.write(f"Oversold: `{settings.rsi_oversold}`")

        st.markdown("**Stochastic**")
        st.write(f"Overbought: `{settings.stochastic_overbought}`")
        st.write(f"Oversold: `{settings.stochastic_oversold}`")

        st.markdown("**Short Interest**")
        st.write(f"High threshold (days to cover): `{settings.short_interest_high}`")

        st.markdown("**Forecasting**")
        st.write(f"Forecast horizons (quarters): `{settings.forecast_horizons}`")
        st.write(f"Holdout periods: `{settings.holdout_periods}`")

    with col2:
        st.markdown("**Position Sizing**")
        st.write(f"Max position size: `{settings.max_position_pct}%`")
        st.write(f"Kelly criterion: `{'enabled' if settings.kelly_criterion_enabled else 'disabled'}`")
        st.write(f"Kelly fraction: `{settings.kelly_fraction}`")

        st.markdown("**Stop Loss / Circuit Breaker**")
        st.write(f"Trailing stop: `{settings.trailing_stop_pct}%` (after `{settings.trailing_stop_days}` profitable days)")
        st.write(f"Portfolio circuit breaker: `{settings.circuit_breaker_daily_pct}%` daily drop")
        st.write(f"Single stock force-sell: `{settings.circuit_breaker_single_stock_pct}%` loss")

        st.markdown("**ML Train/Test Split**")
        st.write(f"Train: `{settings.ml_train_start}` → `{settings.ml_train_end}`")
        st.write(f"Test: `{settings.ml_test_start}` → `{settings.ml_test_end}`")

        st.markdown("**Ranking**")
        st.write(f"Top N picks: `{settings.top_n_picks}`")


# ===========================================================================
# TAB 4 — Data Paths
# ===========================================================================
with tab_paths:
    st.subheader("Data Directory Paths")

    paths = {
        "Project Root": _ROOT,
        "Data Root": settings.data_dir,
        "Daily Prices": settings.raw_daily_dir,
        "Quarterly Prices": settings.raw_quarterly_dir,
        "Macro (FRED)": settings.raw_macro_dir,
        "Dividends": settings.raw_dividends_dir,
        "Processed": settings.processed_dir,
        "Forecasts": settings.forecasts_dir,
        "News Articles": settings.news_articles_dir,
        "Short Interest": settings.news_short_interest_dir,
        "Options": settings.options_dir,
        "Calendar": settings.calendar_dir,
        "Political / Congress": settings.political_congress_dir,
        "Political / Insider": settings.political_insider_dir,
        "Portfolio Snapshots": settings.portfolio_snapshots_dir,
        "Trade Journal": settings.trade_journal_dir,
        "Tax Lots": settings.tax_lots_dir,
    }

    for name, path in paths.items():
        exists = Path(path).exists()
        status = "✅" if exists else "❌ missing"
        st.write(f"**{name}** — `{path}` {status}")
