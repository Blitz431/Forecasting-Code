from datetime import date, datetime, timedelta

import pandas as pd

"""
Purpose: Market calendar helpers — trading day navigation, quarter date ranges, and market-hours check.

Connections:
  - src/scraper/price_scraper.py: uses get_last_trading_day() for incremental update logic
  - src/trading/backtester.py: uses date_range_trading_days() to iterate the test window
  - standalone — no project-internal imports

In:  date objects or ISO strings
Out: adjusted date objects and pd.DatetimeIndex ranges (weekdays only)
"""


# US market holidays (major ones — extend as needed)
US_MARKET_HOLIDAYS_NAMES = {
    "New Year's Day",
    "Martin Luther King Jr. Day",
    "Presidents' Day",
    "Good Friday",
    "Memorial Day",
    "Juneteenth",
    "Independence Day",
    "Labor Day",
    "Thanksgiving",
    "Christmas",
}


def is_weekday(d: date) -> bool:
    """Check if a date is a weekday (Mon-Fri)."""
    return d.weekday() < 5


def get_last_trading_day(d: date | None = None) -> date:
    """Get the most recent trading day (skips weekends).

    Does not account for holidays — use as a best-effort estimate.
    """
    if d is None:
        d = date.today()

    # If it's a weekend, go back to Friday
    while not is_weekday(d):
        d -= timedelta(days=1)

    return d


def get_previous_trading_day(d: date | None = None) -> date:
    """Get the trading day before the given date."""
    if d is None:
        d = date.today()

    d -= timedelta(days=1)
    return get_last_trading_day(d)


def get_quarter(d: date) -> str:
    """Get calendar quarter string (e.g., '2024Q1')."""
    quarter = (d.month - 1) // 3 + 1
    return f"{d.year}Q{quarter}"


def get_quarter_dates(year: int, quarter: int) -> tuple[date, date]:
    """Get start and end dates for a calendar quarter.

    Args:
        year: Calendar year.
        quarter: Quarter number (1-4).

    Returns:
        Tuple of (start_date, end_date).
    """
    start_month = (quarter - 1) * 3 + 1
    start = date(year, start_month, 1)

    end_month = start_month + 2
    if end_month == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, end_month + 1, 1) - timedelta(days=1)

    return start, end


def date_range_trading_days(start: str | date, end: str | date) -> pd.DatetimeIndex:
    """Generate a range of trading days (weekdays only).

    Args:
        start: Start date.
        end: End date.

    Returns:
        DatetimeIndex of weekday dates.
    """
    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d").date()
    if isinstance(end, str):
        end = datetime.strptime(end, "%Y-%m-%d").date()

    return pd.bdate_range(start=start, end=end)


def is_market_open_now() -> bool:
    """Check if US stock market is currently open (rough estimate).

    Uses EST timezone, market hours 9:30 AM - 4:00 PM.
    """
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("US/Eastern"))
    if not is_weekday(now.date()):
        return False

    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now <= market_close
