"""Page 20 — Auto Trades.

Tab 1: Automatic purchase log — every BUY the system placed, with strategy, score, size.
Tab 2: Exit calendar — holding horizon per position, exact exit date, days remaining,
       and trailing-stop status (5% wide → 1% tight after horizon expires).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.tables import style_generic

st.set_page_config(page_title="Auto Trades", page_icon="🤖", layout="wide")
st.title("🤖 Auto Trades")
st.caption(
    "Run the automated trading loop, track every automatic purchase the system made, "
    "and see when each position's holding horizon expires together with its trailing-stop status."
)
st.divider()

# ---------------------------------------------------------------------------#
# Run AutoTrader
# ---------------------------------------------------------------------------#

ROOT_DIR = Path(__file__).parent.parent.parent

with st.container(border=True):
    st.subheader("Run AutoTrader")

    _at_col1, _at_col2, _at_col3 = st.columns(3)
    with _at_col1:
        _mode = st.selectbox(
            "Trading mode",
            ["paper", "live"],
            index=0,
            help="Paper uses your Alpaca paper account. Live requires ALPACA_LIVE_TRADING=true in .env.",
        )
    with _at_col2:
        _dry_run = st.checkbox(
            "Dry run (simulate only — no orders placed)",
            value=True,
            help="Logs what would be bought/sold without submitting any real orders.",
        )
    with _at_col3:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        _run_at = st.button(
            "▶ Run AutoTrader",
            type="primary",
            use_container_width=True,
            help="Runs the full trading loop: regime check → ranker → signals → risk sizing → orders.",
        )

    if _run_at:
        if _mode == "live" and not _dry_run:
            import os as _os
            if _os.getenv("ALPACA_LIVE_TRADING", "").strip().lower() != "true":
                st.error(
                    "Live mode requires `ALPACA_LIVE_TRADING=true` in your `.env` file. "
                    "Set it on the Settings page, then reload before running."
                )
            else:
                _confirmed = st.session_state.get("at_live_confirmed", False)
                if not _confirmed:
                    st.warning("You are about to place REAL orders. Click Run AutoTrader again to confirm.")
                    st.session_state["at_live_confirmed"] = True
                    st.stop()

        st.session_state.pop("at_live_confirmed", None)
        _cmd = [sys.executable, str(ROOT_DIR / "cli" / "trade.py"), "--mode", _mode]
        if _dry_run:
            _cmd.append("--dry-run")

        import os as _os2
        _env = {**_os2.environ, "PYTHONIOENCODING": "utf-8"}

        _AT_STEPS = [
            (["Paper trading mode", "LIVE TRADING MODE"],  "Connecting to Alpaca..."),
            (["Portfolio:"],                               "Connected — checking account..."),
            (["Circuit breaker armed"],                    "Circuit breaker armed..."),
            (["Force-sell", "[CIRCUIT BREAKER]", "[REGIME]"],  "Checking positions + market regime..."),
            (["[RANKING]"],                                "Loading stock rankings..."),
            (["[HIGH-CONVICTION]", "generate_all_signals", "[OPTIONS] Metrics"], "Generating trading signals..."),
            (["[OPTIONS]"],                                "Running options analysis..."),
            (["[DRY-RUN]", "[BUY]", "[SELL]", "[OPT-"],   "Processing orders..."),
            (["[DONE]"],                                   "Saving state + EOD snapshot..."),
        ]
        _at_total = len(_AT_STEPS)
        _at_current = 0

        _at_prog  = st.progress(0, text=_AT_STEPS[0][1])
        _at_label = st.empty()
        _at_label.caption(f"Step 1 / {_at_total} — {_AT_STEPS[0][1]}")
        _log_box  = st.empty()
        _lines: list[str] = []

        def _at_advance(line: str, current: int) -> int:
            for i in range(current + 1, _at_total):
                if any(m in line for m in _AT_STEPS[i][0]):
                    return i
            return current

        try:
            _proc = subprocess.Popen(
                _cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_env,
                cwd=str(ROOT_DIR),
            )
            for _line in _proc.stdout:
                _lines.append(_line.rstrip())
                _new = _at_advance(_line, _at_current)
                if _new != _at_current:
                    _at_current = _new
                    _frac = _at_current / (_at_total - 1)
                    _lbl = _AT_STEPS[_at_current][1]
                    _at_prog.progress(_frac, text=_lbl)
                    _at_label.caption(f"Step {_at_current + 1} / {_at_total} — {_lbl}")
                _log_box.text_area("Output", "\n".join(_lines[-120:]), height=360)
            _proc.wait()
            if _proc.returncode == 0:
                _at_prog.progress(1.0, text="Done!")
                _at_label.caption(f"Step {_at_total} / {_at_total} — Complete")
                st.success("AutoTrader run complete.")
                st.cache_data.clear()
            else:
                st.error(f"AutoTrader exited with code {_proc.returncode}.")
        except Exception as _exc:
            st.error(f"Failed to start AutoTrader: {_exc}")

st.divider()

settings = get_settings()
_STATES_FILE = Path(__file__).parent.parent.parent / "data" / "strategy_states.json"


# ---------------------------------------------------------------------------#
# Data loaders
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=60, show_spinner="Loading buy log …")
def _load_buys() -> pd.DataFrame:
    try:
        from src.trading.trade_journal import TradeJournal
        df = TradeJournal(settings).load()
        if df.empty or "side" not in df.columns:
            return pd.DataFrame()
        return df[df["side"] == "BUY"].copy()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner="Loading position states …")
def _load_states() -> dict:
    if _STATES_FILE.exists():
        try:
            return json.loads(_STATES_FILE.read_text())
        except Exception:
            pass
    return {}


def _load_live_positions() -> dict[str, object]:
    """Fetch current Alpaca positions — intentionally not cached for live accuracy."""
    try:
        from src.trading.alpaca_client import AlpacaClient
        return {p.ticker: p for p in AlpacaClient(settings).list_positions()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------#
# Tabs
# ---------------------------------------------------------------------------#

tab_buys, tab_exit = st.tabs(["📋 Auto-Buy Log", "📅 Exit Calendar"])


# ===========================================================================
# TAB 1 — Auto-Buy Log
# ===========================================================================

with tab_buys:
    buys = _load_buys()

    if buys.empty:
        st.info(
            "No automatic buys recorded yet.\n\n"
            "Run `python cli/trade.py --mode paper` to start the trading loop."
        )
    else:
        # ── Summary metrics ────────────────────────────────────────────────
        total_capital = float(buys["dollar_value"].sum()) if "dollar_value" in buys.columns else 0.0
        unique_tickers = int(buys["ticker"].nunique()) if "ticker" in buys.columns else 0
        hc_count = int((buys["strategy"] == "high_conviction").sum()) if "strategy" in buys.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Auto-Buys", len(buys))
        c2.metric("Capital Deployed", f"${total_capital:,.0f}")
        c3.metric("Unique Tickers", unique_tickers)
        c4.metric("High-Conviction Buys", hc_count)
        st.divider()

        # ── Filters ────────────────────────────────────────────────────────
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            strategies = sorted(buys["strategy"].dropna().unique().tolist()) if "strategy" in buys.columns else []
            strat_filter = st.multiselect("Strategy", strategies, default=strategies)
        with fc2:
            all_tickers = sorted(buys["ticker"].dropna().unique().tolist()) if "ticker" in buys.columns else []
            ticker_filter = st.multiselect("Ticker", all_tickers, default=[])
        with fc3:
            try:
                earliest = min(pd.to_datetime(buys["timestamp"]).min().date(), date.today())
            except Exception:
                earliest = date.today() - timedelta(days=30)
            default_start = max(earliest, date.today() - timedelta(days=30))
            date_range = st.date_input(
                "Date range",
                value=(default_start, date.today()),
            )

        # ── Apply filters ──────────────────────────────────────────────────
        filtered = buys.copy()
        if strat_filter and "strategy" in filtered.columns:
            filtered = filtered[filtered["strategy"].isin(strat_filter)]
        if ticker_filter and "ticker" in filtered.columns:
            filtered = filtered[filtered["ticker"].isin(ticker_filter)]
        if "timestamp" in filtered.columns and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            ts = pd.to_datetime(filtered["timestamp"], utc=True).dt.tz_localize(None)
            filtered = filtered[
                (ts >= pd.Timestamp(date_range[0])) &
                (ts < pd.Timestamp(date_range[1]) + pd.Timedelta(days=1))
            ]

        # ── Score extraction (signals dict stores {ticker: composite_score}) ─
        def _composite_score(row) -> str:
            try:
                raw = row.get("signals", "{}")
                d = json.loads(raw) if isinstance(raw, str) else raw
                ticker = row.get("ticker", "")
                if isinstance(d, dict):
                    if ticker in d:
                        return f"{float(d[ticker]):.3f}"
                    if d:
                        return f"{float(next(iter(d.values()))):.3f}"
            except Exception:
                pass
            return "—"

        # ── Build display frame ────────────────────────────────────────────
        col_map = {
            "timestamp":   "Date",
            "ticker":      "Ticker",
            "strategy":    "Strategy",
            "shares":      "Shares",
            "price":       "Price",
            "dollar_value": "Value ($)",
        }
        avail = {k: v for k, v in col_map.items() if k in filtered.columns}
        disp = filtered[list(avail.keys())].rename(columns=avail).reset_index(drop=True).copy()

        # Insert score column after Strategy
        scores = filtered.apply(_composite_score, axis=1).values
        insert_pos = list(disp.columns).index("Strategy") + 1 if "Strategy" in disp.columns else len(disp.columns)
        disp.insert(insert_pos, "Score", scores)

        if "Date" in disp.columns:
            disp["Date"] = pd.to_datetime(disp["Date"]).dt.strftime("%Y-%m-%d %H:%M")
        if "Price" in disp.columns:
            disp["Price"] = disp["Price"].map(lambda x: f"${float(x):.2f}" if pd.notna(x) else "—")
        if "Value ($)" in disp.columns:
            disp["Value ($)"] = disp["Value ($)"].map(lambda x: f"${float(x):,.0f}" if pd.notna(x) else "—")

        st.metric("Trades shown", len(disp))
        st.dataframe(style_generic(disp), use_container_width=True, height=440, hide_index=True)


# ===========================================================================
# TAB 2 — Exit Calendar
# ===========================================================================

with tab_exit:
    states = _load_states()
    live   = _load_live_positions()
    today  = date.today()

    if not states:
        st.info(
            "No position states found in `data/strategy_states.json`.\n\n"
            "States are written after the first trading cycle."
        )
    else:
        rows = []
        for ticker, sd in states.items():
            entry_str = sd.get("entry_date", "")
            horizon   = int(sd.get("holding_horizon_days", 30))
            trail_pct = float(sd.get("broker_trail_pct", 5.0))
            entry_px  = float(sd.get("entry_price", 0.0))

            try:
                entry_date = date.fromisoformat(entry_str)
            except (ValueError, TypeError):
                continue

            exit_date  = entry_date + timedelta(days=horizon)
            days_held  = (today - entry_date).days
            days_left  = max(0, (exit_date - today).days)

            if days_left == 0:
                status = "🔴 Overdue"
            elif days_left <= 3:
                status = "🟡 Due soon"
            else:
                status = "🟢 Active"

            horizon_label = {14: "Short (14d)", 30: "Medium (30d)", 60: "Long (60d)"}.get(
                horizon, f"Custom ({horizon}d)"
            )
            trail_label = "Tight (1%)" if trail_pct <= 1.0 else "Wide (5%)"

            pos = live.get(ticker)
            cur_price = float(pos.current_price)   if pos else None
            unr_plpc  = float(pos.unrealized_plpc) if pos else None

            rows.append({
                "Status":         status,
                "Ticker":         ticker,
                "Entry Date":     entry_date.isoformat(),
                "Exit Date":      exit_date.isoformat(),
                "Days Held":      days_held,
                "Days Left":      days_left,
                "Horizon":        horizon_label,
                "Trail":          trail_label,
                "Entry $":        f"${entry_px:.2f}",
                "Current $":      f"${cur_price:.2f}"   if cur_price is not None else "—",
                "Unrealized %":   f"{unr_plpc:+.2f}%"  if unr_plpc is not None else "—",
                # internal only — used for progress bar
                "_horizon_days":  horizon,
            })

        if not rows:
            st.info("No valid position states with entry dates found.")
        else:
            df_exit = pd.DataFrame(rows).sort_values("Days Left").reset_index(drop=True)

            # ── Summary metrics ────────────────────────────────────────────
            overdue   = int((df_exit["Days Left"] == 0).sum())
            due_soon  = int(((df_exit["Days Left"] > 0) & (df_exit["Days Left"] <= 3)).sum())
            tightened = int((df_exit["Trail"] == "Tight (1%)").sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Positions Tracked", len(df_exit))
            c2.metric("🔴 Overdue", overdue)
            c3.metric("🟡 Due within 3 days", due_soon)
            c4.metric("✅ Trail Tightened", tightened)
            st.divider()

            # ── Horizon progress bars + per-position controls ──────────────
            st.subheader("Horizon Progress")

            def _save_state_change(ticker: str, patch: dict) -> None:
                """Merge *patch* into the saved state for *ticker* and persist."""
                raw = json.loads(_STATES_FILE.read_text()) if _STATES_FILE.exists() else {}
                sd  = raw.get(ticker, {})
                sd.update(patch)
                raw[ticker] = sd
                _STATES_FILE.parent.mkdir(parents=True, exist_ok=True)
                _STATES_FILE.write_text(json.dumps(raw, indent=2))
                st.cache_data.clear()

            for _, row in df_exit.iterrows():
                ticker_r      = row["Ticker"]
                days_held_val = int(row["Days Held"])
                days_left_val = int(row["Days Left"])
                total         = days_held_val + days_left_val
                progress      = min(1.0, days_held_val / max(1, total))
                raw_sd        = states.get(ticker_r, {})
                cur_mode      = raw_sd.get("tighten_mode", "auto")
                cur_td        = raw_sd.get("tighten_on_date", "")
                cur_horizon   = int(raw_sd.get("holding_horizon_days", 30))

                days_word = "day" if days_left_val == 1 else "days"
                label = (
                    f"{row['Status']}  &nbsp; **{ticker_r}**"
                    f"  ·  exit **{row['Exit Date']}**"
                    f"  ·  **{days_left_val} {days_word} left**"
                    f"  ·  trail: *{row['Trail']}*"
                    f"  ·  unrealized: {row['Unrealized %']}"
                )
                st.markdown(label, unsafe_allow_html=True)
                st.progress(progress)

                # Per-position tighten controls
                with st.expander(f"⚙️ Tighten settings for {ticker_r}", expanded=False):
                    ctl1, ctl2, ctl3 = st.columns([2, 2, 1])

                    with ctl1:
                        new_mode = st.selectbox(
                            "When to tighten (5% → 1%)",
                            options=["auto", "on_date", "manual"],
                            index=["auto", "on_date", "manual"].index(cur_mode),
                            format_func=lambda x: {
                                "auto":    "Auto — after horizon expires",
                                "on_date": "On a specific date",
                                "manual":  "Manual — I'll do it myself",
                            }[x],
                            key=f"mode_{ticker_r}",
                        )

                    with ctl2:
                        if new_mode == "auto":
                            new_horizon = st.number_input(
                                "Horizon days",
                                min_value=1, max_value=365,
                                value=cur_horizon,
                                key=f"horizon_{ticker_r}",
                            )
                            new_td = ""
                        elif new_mode == "on_date":
                            try:
                                default_td = date.fromisoformat(cur_td) if cur_td else today + timedelta(days=30)
                            except ValueError:
                                default_td = today + timedelta(days=30)
                            new_td_val = st.date_input(
                                "Exact tighten date",
                                value=default_td,
                                min_value=today,
                                key=f"td_{ticker_r}",
                            )
                            new_td = new_td_val.isoformat()
                            new_horizon = cur_horizon
                        else:
                            st.markdown("*Tightening is manual — use the button →*")
                            new_td = ""
                            new_horizon = cur_horizon

                    with ctl3:
                        st.markdown("&nbsp;", unsafe_allow_html=True)  # vertical spacer
                        if st.button("💾 Save", key=f"save_{ticker_r}"):
                            _save_state_change(ticker_r, {
                                "tighten_mode":       new_mode,
                                "tighten_on_date":    new_td,
                                "holding_horizon_days": int(new_horizon),
                            })
                            st.success(f"Saved for {ticker_r}.")
                            st.rerun()

                        if row["Trail"] == "Wide (5%)":
                            if st.button(f"🔧 Tighten to 1% now", key=f"tighten_{ticker_r}"):
                                try:
                                    from src.trading.alpaca_client import AlpacaClient
                                    _client = AlpacaClient(settings)
                                    # Cancel existing trailing stops for this ticker
                                    for _o in _client.list_open_orders(ticker_r):
                                        _otype = _o.order_type.value if hasattr(_o.order_type, "value") else str(_o.order_type)
                                        _oside = _o.side.value if hasattr(_o.side, "value") else str(_o.side)
                                        if _otype == "trailing_stop" and _oside == "sell":
                                            _client.cancel_order(str(_o.id))
                                    pos_r = live.get(ticker_r)
                                    qty_r = pos_r.qty if pos_r else 0.0
                                    if qty_r > 0:
                                        ts_r = _client.place_trailing_stop(
                                            ticker_r, qty_r,
                                            trail_percent=settings.broker_trailing_stop_tight_pct,
                                        )
                                        if ts_r:
                                            _save_state_change(ticker_r, {
                                                "broker_trail_pct": settings.broker_trailing_stop_tight_pct,
                                                "tighten_mode":     "manual",
                                            })
                                            st.success(f"✅ {ticker_r} tightened to 1% trail.")
                                            st.rerun()
                                        else:
                                            st.error("Trailing stop placement failed.")
                                    else:
                                        st.warning("No live position found — cannot determine qty.")
                                except Exception as _ex:
                                    st.error(f"Error: {_ex}")
                        else:
                            st.markdown("*Already tight (1%)*")

                st.markdown("---")

            st.divider()

            # ── Full table ─────────────────────────────────────────────────
            st.subheader("Full Table")
            table_df = df_exit.drop(columns=["_horizon_days"])
            st.dataframe(style_generic(table_df), use_container_width=True, hide_index=True)
