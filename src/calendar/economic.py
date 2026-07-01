from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.scraper.storage import upsert_dataframe, load_dataframe
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
STATUS: Work in progress

Purpose: Economic event calendar — tracks upcoming FOMC meetings, CPI releases, and jobs reports.

Connections:
  - src/scraper/storage.py: upsert_dataframe(), load_dataframe() for parquet persistence
  - config/settings.py: FRED API key, calendar_dir
  - src/ranking/ranker.py: planned to read event_risk as a global market-wide adjustment
  - src/utils/logging.py: logger

In:  FRED API (release calendars for CPIAUCSL, UNRATE, GDP) + static FOMC date list
Out: data/calendar/economic.parquet (event_type, event_name, days_away — market-wide, not per-ticker)
"""

# ---------------------------------------------------------------------------#
# Static FOMC calendar (updated annually)
# The Fed publishes these dates in advance at federalreserve.gov/monetarypolicy
# ---------------------------------------------------------------------------#

_FOMC_DATES_2025: list[str] = [
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-10",
]

_FOMC_DATES_2026: list[str] = [
    "2026-01-28",
    "2026-03-18",
    "2026-04-29",
    "2026-06-17",
    "2026-07-29",
    "2026-09-16",
    "2026-10-28",
    "2026-12-09",
]

_FOMC_DATES: list[date] = [
    date.fromisoformat(d)
    for d in _FOMC_DATES_2025 + _FOMC_DATES_2026
]


# ---------------------------------------------------------------------------#
# FRED release calendar fetch
# ---------------------------------------------------------------------------#

def _fetch_fred_release_dates(
    series_id: str,
    fred_api_key: str,
    lookback_days: int = 30,
    lookahead_days: int = 90,
) -> list[date]:
    """Fetch release dates for a FRED series from the FRED release calendar API.

    Calls:
      https://api.stlouisfed.org/fred/series/release?series_id={id}&api_key={key}
    to get the release ID, then:
      https://api.stlouisfed.org/fred/release/dates?release_id={id}&...
    to get actual release dates.

    Args:
        series_id: FRED series (e.g. "CPIAUCSL").
        fred_api_key: FRED API key from settings.
        lookback_days: Include dates this many days in the past.
        lookahead_days: Include dates this many days in the future.

    Returns:
        Sorted list of release dates within the window.
    """
    try:
        import requests  # already installed (Phase 5)
    except ImportError:
        logger.warning("requests not installed — cannot fetch FRED release dates")
        return []

    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    end = (date.today() + timedelta(days=lookahead_days)).isoformat()

    # Step 1: get the release ID for this series
    try:
        import requests as _requests
        series_url = (
            f"https://api.stlouisfed.org/fred/series/release"
            f"?series_id={series_id}&api_key={fred_api_key}&file_type=json"
        )
        resp = _requests.get(series_url, timeout=10)
        resp.raise_for_status()
        release_id = resp.json()["releases"][0]["id"]
    except Exception as exc:
        logger.debug(f"Could not get FRED release ID for {series_id}: {exc}")
        return []

    # Step 2: get release dates for that release
    try:
        dates_url = (
            f"https://api.stlouisfed.org/fred/release/dates"
            f"?release_id={release_id}&realtime_start={start}&realtime_end={end}"
            f"&api_key={fred_api_key}&file_type=json&include_release_dates_with_no_data=false"
        )
        resp = _requests.get(dates_url, timeout=10)
        resp.raise_for_status()
        raw_dates = [d["date"] for d in resp.json().get("release_dates", [])]
        return sorted([date.fromisoformat(d) for d in raw_dates])
    except Exception as exc:
        logger.debug(f"Could not get FRED release dates for {series_id} (id={release_id}): {exc}")
        return []


# ---------------------------------------------------------------------------#
# Event assembly
# ---------------------------------------------------------------------------#

_FRED_SERIES_LABELS: dict[str, str] = {
    "CPIAUCSL": "CPI Release",
    "UNRATE": "Jobs Report",
    "GDP": "GDP Release",
}


def fetch_economic_events(
    lookahead_days: int = 90,
    lookback_days: int = 30,
) -> list[dict]:
    """Build a combined economic event calendar.

    Merges FOMC dates (static) with FRED-sourced release dates for CPI,
    jobs, and GDP.  If the FRED API key is missing, only FOMC dates are
    included (they are static and require no API call).

    Args:
        lookahead_days: How many days ahead to include.
        lookback_days: How many days in the past to include (for context).

    Returns:
        List of event dicts sorted by date, each with keys:
          - event_date (date)
          - event_type (str): "FOMC", "CPI", "Jobs", "GDP"
          - event_name (str): human-readable label
          - days_away (int): negative = past, 0 = today, positive = future
    """
    settings = get_settings()
    today = date.today()
    window_start = today - timedelta(days=lookback_days)
    window_end = today + timedelta(days=lookahead_days)

    events: list[dict] = []

    # FOMC (static — no API needed)
    for fomc_date in _FOMC_DATES:
        if window_start <= fomc_date <= window_end:
            events.append({
                "event_date": fomc_date,
                "event_type": "FOMC",
                "event_name": "FOMC Meeting",
                "days_away": (fomc_date - today).days,
            })

    # FRED-sourced releases (CPI, Jobs, GDP)
    fred_key = settings.fred.api_key
    if fred_key:
        for series_id, label in _FRED_SERIES_LABELS.items():
            event_type = label.split()[0]  # "CPI", "Jobs", "GDP"
            fred_dates = _fetch_fred_release_dates(
                series_id, fred_key,
                lookback_days=lookback_days,
                lookahead_days=lookahead_days,
            )
            for d in fred_dates:
                if window_start <= d <= window_end:
                    events.append({
                        "event_date": d,
                        "event_type": event_type,
                        "event_name": label,
                        "days_away": (d - today).days,
                    })
    else:
        logger.info("FRED API key not set — economic calendar limited to FOMC dates only")

    events.sort(key=lambda e: e["event_date"])
    logger.info(f"Economic calendar: {len(events)} events in [{lookback_days}d ago, +{lookahead_days}d]")
    return events


# ---------------------------------------------------------------------------#
# Summary helpers
# ---------------------------------------------------------------------------#

def _days_to_next(events: list[dict], event_type: str) -> int | None:
    """Return days_away for the next upcoming event of a given type."""
    future = [e for e in events if e["event_type"] == event_type and e["days_away"] >= 0]
    if not future:
        return None
    return min(e["days_away"] for e in future)


def get_economic_summary(events: list[dict], lookahead_days: int = 30) -> dict:
    """Summarise the economic calendar into ranker-ready fields.

    Args:
        events: Output of :func:`fetch_economic_events`.
        lookahead_days: Window for computing event_risk.

    Returns:
        Dict with keys:
          - upcoming_events: list of event dicts with days_away >= 0
          - fomc_days_away (int | None)
          - cpi_days_away (int | None)
          - jobs_days_away (int | None)
          - gdp_days_away (int | None)
          - event_risk (float): fraction of next 30 days with a macro event,
                                used as a global risk-off signal [0, 1]
    """
    upcoming = [e for e in events if 0 <= e["days_away"] <= lookahead_days]

    event_dates: set[date] = {e["event_date"] for e in upcoming}
    event_risk = len(event_dates) / max(lookahead_days, 1)

    return {
        "upcoming_events": upcoming,
        "fomc_days_away": _days_to_next(events, "FOMC"),
        "cpi_days_away": _days_to_next(events, "CPI"),
        "jobs_days_away": _days_to_next(events, "Jobs"),
        "gdp_days_away": _days_to_next(events, "GDP"),
        "event_risk": min(float(event_risk), 1.0),
    }


# ---------------------------------------------------------------------------#
# Persistence
# ---------------------------------------------------------------------------#

def save_economic_calendar(events: list[dict], calendar_dir: Path) -> None:
    """Persist the economic event list to data/calendar/economic.parquet.

    Args:
        events: Output of :func:`fetch_economic_events`.
        calendar_dir: Root calendar storage directory.
    """
    if not events:
        return

    calendar_dir.mkdir(parents=True, exist_ok=True)
    filepath = calendar_dir / "economic.parquet"

    rows = []
    for e in events:
        rows.append({
            "event_type": e["event_type"],
            "event_name": e["event_name"],
            "days_away": e["days_away"],
        })

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(
        [pd.Timestamp(e["event_date"], tz="UTC") for e in events],
        name="date",
    ))
    upsert_dataframe(df, filepath)
    logger.debug(f"Economic calendar saved: {len(events)} events")


def load_economic_calendar(calendar_dir: Path) -> pd.DataFrame:
    """Load the stored economic event calendar.

    Args:
        calendar_dir: Root calendar storage directory.

    Returns:
        DataFrame with DatetimeIndex (event dates) and columns:
        event_type, event_name, days_away.
        Empty DataFrame if not yet populated.
    """
    filepath = calendar_dir / "economic.parquet"
    return load_dataframe(filepath)


def get_upcoming_events(
    calendar_dir: Path,
    lookahead_days: int = 30,
) -> list[dict]:
    """Load stored calendar and return events in the upcoming window.

    Falls back to fetching live if the stored calendar is empty.

    Args:
        calendar_dir: Root calendar storage directory.
        lookahead_days: Days ahead to include.

    Returns:
        List of upcoming event dicts sorted by days_away.
    """
    df = load_economic_calendar(calendar_dir)
    today = date.today()

    if not df.empty:
        rows = []
        for idx, row in df.iterrows():
            event_date = idx.date()
            days_away = (event_date - today).days
            if 0 <= days_away <= lookahead_days:
                rows.append({
                    "event_date": event_date,
                    "event_type": row["event_type"],
                    "event_name": row["event_name"],
                    "days_away": days_away,
                })
        return sorted(rows, key=lambda e: e["days_away"])

    # No stored data — fetch live
    logger.info("No stored economic calendar — fetching live")
    events = fetch_economic_events(lookahead_days=lookahead_days)
    save_economic_calendar(events, calendar_dir)
    return [e for e in events if 0 <= e["days_away"] <= lookahead_days]
