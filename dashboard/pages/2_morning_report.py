"""Page 19 — Morning Report.

Top section : Dropdown to select any saved report date from data/reports/.
              "Generate Today's Report" button calls MorningReport.generate() and
              shows the saved file paths on success.

Download    : PDF and Excel download buttons appear for the selected report.

Six live-data tabs (data loaded fresh from the same functions used by
morning_report.py — not read from the saved files):
    1. Overview      — Market regime, VIX, 10Y-2Y yield spread, regime score
    2. Top Picks     — Top-N ranked picks table
    3. Portfolio     — Open positions with P&L
    4. Trade Journal — Last 5 trades + win rate + total P&L
    5. Earnings      — Upcoming earnings in next 7 days for held tickers
    6. Alerts        — Active alert events colour-coded by level
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings

st.set_page_config(page_title="Morning Report", page_icon="📋", layout="wide")

from dashboard.components.market_clock import render_market_clock
render_market_clock()

st.title("📋 Morning Report")
st.caption(
    "Auto-generated daily report with top picks, portfolio status, "
    "key signals, and alerts."
)
st.divider()

settings = get_settings()

reports_dir: Path = settings.data_dir / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------#
# Report picker + generate button
# ---------------------------------------------------------------------------#

pdf_files = sorted(reports_dir.glob("morning_*.pdf"), reverse=True)
available_dates = [f.stem.replace("morning_", "") for f in pdf_files]

col_pick, col_btn = st.columns([3, 1])

with col_pick:
    selected_date = st.selectbox(
        "Saved report date",
        options=available_dates if available_dates else ["(no reports yet)"],
        index=0,
    )

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    generate = st.button("Generate Today's Report", type="primary")

if generate:
    with st.spinner("Generating PDF + Excel report …"):
        try:
            from src.reports.morning_report import MorningReport

            report = MorningReport(settings)
            paths = report.generate()
            st.success(
                f"Report saved!  \n"
                f"**PDF:** `{paths['pdf']}`  \n"
                f"**Excel:** `{paths['xlsx']}`"
            )
            # Refresh the date list so the new report is selectable
            pdf_files = sorted(reports_dir.glob("morning_*.pdf"), reverse=True)
            available_dates = [f.stem.replace("morning_", "") for f in pdf_files]
            if available_dates:
                selected_date = available_dates[0]
        except Exception as exc:
            st.error(f"Generation failed: {exc}")

# ---------------------------------------------------------------------------#
# Download buttons
# ---------------------------------------------------------------------------#

if selected_date and selected_date != "(no reports yet)":
    pdf_path = reports_dir / f"morning_{selected_date}.pdf"
    xlsx_path = reports_dir / f"morning_{selected_date}.xlsx"

    dl1, dl2, _ = st.columns([1, 1, 4])
    if pdf_path.exists():
        with dl1:
            st.download_button(
                label="⬇ Download PDF",
                data=pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
            )
    if xlsx_path.exists():
        with dl2:
            st.download_button(
                label="⬇ Download Excel",
                data=xlsx_path.read_bytes(),
                file_name=xlsx_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

st.divider()

# ---------------------------------------------------------------------------#
# Live-data tabs
# ---------------------------------------------------------------------------#

(
    tab_overview,
    tab_picks,
    tab_portfolio,
    tab_journal,
    tab_earnings,
    tab_alerts,
) = st.tabs(["Overview", "Top Picks", "Portfolio", "Trade Journal", "Earnings", "Alerts"])


# ---- Cached data loaders ---- #

@st.cache_data(ttl=300, show_spinner=False)
def _regime() -> dict:
    try:
        from src.reports.morning_report import _get_regime_data
        return _get_regime_data(settings)
    except Exception as exc:
        return {"_error": str(exc)}


@st.cache_data(ttl=300, show_spinner=False)
def _top_picks() -> pd.DataFrame:
    try:
        from src.reports.morning_report import _get_top_picks
        return _get_top_picks(settings)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def _portfolio() -> pd.DataFrame:
    try:
        from src.reports.morning_report import _get_portfolio_positions
        return _get_portfolio_positions(settings)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def _journal() -> tuple[pd.DataFrame, dict]:
    try:
        from src.reports.morning_report import _get_trade_journal_summary
        return _get_trade_journal_summary(settings)
    except Exception:
        return pd.DataFrame(), {}


@st.cache_data(ttl=300, show_spinner=False)
def _held_tickers() -> tuple[str, ...]:
    try:
        from src.reports.morning_report import MorningReport
        return tuple(MorningReport(settings)._load_held_tickers())
    except Exception:
        return ()


@st.cache_data(ttl=300, show_spinner=False)
def _earnings(held: tuple[str, ...]) -> pd.DataFrame:
    try:
        from src.reports.morning_report import _get_upcoming_earnings
        return _get_upcoming_earnings(list(held), settings)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def _alerts(held: tuple[str, ...]) -> list[dict]:
    try:
        from src.reports.morning_report import _get_active_alerts
        evts = _get_active_alerts(list(held), settings)
        return [
            {
                "level":   e.level,
                "trigger": e.trigger_type,
                "ticker":  e.ticker or "—",
                "title":   e.title,
                "body":    e.body,
            }
            for e in evts
        ]
    except Exception as exc:
        return [
            {
                "level":   "critical",
                "trigger": "error",
                "ticker":  "—",
                "title":   "Alert check failed",
                "body":    str(exc),
            }
        ]


# ---- Tab 1: Overview ---- #

with tab_overview:
    with st.spinner("Loading market regime …"):
        regime = _regime()

    if "_error" in regime:
        st.warning(f"Could not load regime data: {regime['_error']}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Regime", regime.get("regime", "Unknown"))
        c2.metric(
            "VIX",
            f"{regime['vix']:.1f}" if regime.get("vix") is not None else "N/A",
        )
        c3.metric(
            "10Y-2Y Spread",
            f"{regime['yield_spread']:.2f}%"
            if regime.get("yield_spread") is not None
            else "N/A",
        )
        c4.metric(
            "Regime Score",
            f"{regime['score']:.2f}" if regime.get("score") is not None else "N/A",
        )

# ---- Tab 2: Top Picks ---- #

with tab_picks:
    with st.spinner("Loading top picks …"):
        picks_df = _top_picks()

    if picks_df.empty:
        st.info("Ranking data not available. Run `cli/rank.py` or `cli/scheduler.py --run-now`.")
    else:
        st.metric("Picks loaded", len(picks_df))
        st.dataframe(picks_df, use_container_width=True, hide_index=True)

# ---- Tab 3: Portfolio ---- #

with tab_portfolio:
    with st.spinner("Loading positions …"):
        pos_df = _portfolio()

    if pos_df.empty:
        st.info("No open positions.")
    else:
        st.metric("Open positions", len(pos_df))
        st.dataframe(pos_df, use_container_width=True, hide_index=True)

# ---- Tab 4: Trade Journal ---- #

with tab_journal:
    with st.spinner("Loading trade journal …"):
        j_df, j_stats = _journal()

    if j_stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Trades", j_stats.get("total_trades", 0))
        c2.metric("Win Rate", f"{j_stats.get('win_rate', 0):.1%}")
        c3.metric("Total P&L", f"${j_stats.get('total_pnl', 0):+,.2f}")

    if j_df.empty:
        st.info("No trade history yet.")
    else:
        st.dataframe(j_df, use_container_width=True, hide_index=True)

# ---- Tab 5: Earnings ---- #

with tab_earnings:
    held = _held_tickers()
    with st.spinner("Loading earnings calendar …"):
        earn_df = _earnings(held)

    if earn_df.empty:
        st.info("No earnings in the next 7 days for currently held tickers.")
    else:
        st.metric("Upcoming earnings", len(earn_df))
        st.dataframe(earn_df, use_container_width=True, hide_index=True)

# ---- Tab 6: Alerts ---- #

with tab_alerts:
    held = _held_tickers()
    with st.spinner("Checking alerts …"):
        alert_rows = _alerts(held)

    _icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
    _color = {"info": "blue", "warning": "orange", "critical": "red"}

    if not alert_rows:
        st.success("No active alerts.")
    else:
        st.metric("Active alerts", len(alert_rows))
        for row in alert_rows:
            icon = _icon.get(row["level"], "")
            color = _color.get(row["level"], "gray")
            ticker_tag = f"  Ticker: `{row['ticker']}`" if row["ticker"] != "—" else ""
            st.markdown(
                f":{color}[{icon} **{row['title']}**  `{row['trigger']}`{ticker_tag}]"
            )
            with st.expander("Details"):
                st.text(row["body"])
