"""Page 12 — Calendar.

Upcoming earnings dates + economic events (FOMC, CPI, jobs, GDP).
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import earnings_timeline
from dashboard.components.tables import style_earnings_table, style_generic

st.set_page_config(page_title="Calendar", page_icon="📅", layout="wide")
st.title("📅 Earnings & Economic Calendar")
st.caption("Upcoming earnings dates, beat/miss history, and market-moving economic events.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Earnings calendar
# ---------------------------------------------------------------------------#

st.subheader("📆 Upcoming Earnings")

@st.cache_data(ttl=600, show_spinner="Loading earnings calendar …")
def _load_all_earnings(lookahead: int) -> pd.DataFrame:
    """Aggregate earnings data for all tickers with stored data."""
    rows = []
    cal_dir = settings.calendar_dir / "earnings"
    if not cal_dir.exists():
        return pd.DataFrame()

    for fp in cal_dir.glob("*.parquet"):
        ticker = fp.stem
        try:
            df = pd.read_parquet(fp)
            if df.empty:
                continue
            latest = df.sort_index().iloc[-1]
            days = latest.get("days_to_earnings")
            if days is not None and 0 <= days <= lookahead:
                rows.append({
                    "Ticker":        ticker,
                    "Next Earnings": latest.get("next_earnings_date"),
                    "Days Away":     int(days),
                    "Beat Rate":     latest.get("beat_rate"),
                    "Avg EPS Surp.": latest.get("avg_eps_surprise_pct"),
                    "Signal":        round(float(latest.get("earnings_signal", 0.0)), 3),
                })
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Days Away")


with st.sidebar:
    lookahead_days = st.slider("Lookahead (days)", 7, 90, value=30)
    fetch_ticker = st.text_input("Fetch earnings for ticker",
                                  placeholder="AAPL")

# Fetch button
if fetch_ticker.strip():
    t = fetch_ticker.strip().upper()
    if st.sidebar.button(f"Fetch {t} Earnings"):
        with st.spinner(f"Fetching earnings data for {t} …"):
            try:
                from src.calendar.earnings import compute_earnings_metrics, save_earnings_metrics
                data = compute_earnings_metrics(t)
                save_earnings_metrics(data, settings.calendar_dir)
                st.success(f"Saved earnings data for {t}")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

earnings_df = _load_all_earnings(lookahead_days)

if earnings_df.empty:
    st.info(
        f"No earnings data found for the next {lookahead_days} days.\n\n"
        "Run `python cli/scrape.py` to populate the calendar, "
        "or use the sidebar to fetch for a specific ticker."
    )
else:
    # Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Upcoming Earnings",    len(earnings_df))
    bullish = (earnings_df["Signal"] > 0.1).sum()
    col2.metric("Bullish Setups (signal > 0.1)", bullish)
    col3.metric("This week (≤ 7 days)", (earnings_df["Days Away"] <= 7).sum())

    st.plotly_chart(earnings_timeline(earnings_df), use_container_width=True)
    st.dataframe(style_earnings_table(earnings_df), use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------#
# Economic events
# ---------------------------------------------------------------------------#

st.subheader("🏦 Economic Events")

@st.cache_data(ttl=3600, show_spinner="Loading economic calendar …")
def _load_economic_events(lookahead: int) -> list[dict]:
    try:
        from src.calendar.economic import get_upcoming_events
        return get_upcoming_events(settings.calendar_dir, lookahead_days=lookahead)
    except Exception:
        return []

@st.cache_data(ttl=3600)
def _economic_summary(events) -> dict:
    try:
        from src.calendar.economic import get_economic_summary
        return get_economic_summary(list(events), lookahead_days=lookahead_days)
    except Exception:
        return {}

econ_events = _load_economic_events(lookahead_days)
econ_summary = _economic_summary(tuple(str(e) for e in econ_events))

# Refresh economic calendar
if st.button("🔄 Refresh Economic Calendar"):
    with st.spinner("Fetching economic events …"):
        try:
            from src.calendar.economic import fetch_economic_events, save_economic_calendar
            events = fetch_economic_events(lookahead_days=90, lookback_days=30)
            save_economic_calendar(events, settings.calendar_dir)
            st.success(f"Fetched {len(events)} economic events")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Failed: {e}")

if not econ_events:
    st.info(
        "No economic events cached. Click **Refresh Economic Calendar** to fetch from FRED."
    )
else:
    # Key event metrics
    col1, col2, col3, col4 = st.columns(4)
    fomc_days = econ_summary.get("fomc_days_away")
    cpi_days  = econ_summary.get("cpi_days_away")
    jobs_days = econ_summary.get("jobs_days_away")
    risk      = econ_summary.get("event_risk", 0.0)
    col1.metric("FOMC Days Away",    f"{fomc_days}d" if fomc_days is not None else "—")
    col2.metric("CPI Days Away",     f"{cpi_days}d"  if cpi_days  is not None else "—")
    col3.metric("Jobs Days Away",    f"{jobs_days}d" if jobs_days is not None else "—")
    col4.metric("Event Risk Score",  f"{risk:.2f}", help="0 = calm, 1 = heavy event week")

    # Events table
    events_display = pd.DataFrame(econ_events)
    if not events_display.empty:
        display_cols = [c for c in ["event_type", "event_name", "event_date",
                                     "days_away", "importance"] if c in events_display.columns]
        st.dataframe(style_generic(events_display[display_cols] if display_cols else events_display),
                     use_container_width=True)

    # High-risk warning
    if risk > 0.5:
        st.warning(
            f"⚠️ High event-risk week ahead (score {risk:.2f}). "
            "Multiple market-moving events may cause elevated volatility."
        )
