"""Shared NY market-hours clock — rendered at the top of every dashboard page.

Usage in any page (right after st.set_page_config):
    from dashboard.components.market_clock import render_market_clock
    render_market_clock()
"""

from __future__ import annotations

import streamlit as st

"""
Purpose: One-line banner showing the current time in New York plus whether the market
         is open/closed and a countdown to the next open/close. Helps users outside
         Eastern time (e.g. SF/PT) avoid mistiming trading-sensitive actions.

Connections:
  - Uses Alpaca's TradingClient.get_clock() (accounts for market holidays automatically)
  - config/settings.py: AlpacaSettings (api_key, secret_key)
  - Called at the top of every page in dashboard/app.py and dashboard/pages/*.py

In:  ALPACA_API_KEY / ALPACA_SECRET_KEY
Out: a Streamlit caption line (no return value)
"""


@st.cache_data(ttl=30, show_spinner=False)
def _get_clock() -> dict | None:
    """Fetch Alpaca's market clock. Cached 30s and shared across all pages/sessions."""
    try:
        from config.settings import get_settings
        settings = get_settings()
        if not settings.alpaca.api_key or not settings.alpaca.secret_key:
            return None
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.secret_key,
            paper=True,  # market clock/calendar is identical for paper & live accounts
        )
        clock = client.get_clock()
        return {
            "timestamp": clock.timestamp,
            "is_open": clock.is_open,
            "next_open": clock.next_open,
            "next_close": clock.next_close,
        }
    except Exception:
        return None


def _format_countdown(delta) -> str:
    total_min = max(int(delta.total_seconds() // 60), 0)
    days, rem = divmod(total_min, 24 * 60)
    hrs, mins = divmod(rem, 60)
    if days:
        return f"{days}d {hrs}h"
    if hrs:
        return f"{hrs}h {mins}m"
    return f"{mins}m"


@st.fragment(run_every="30s")
def render_market_clock() -> None:
    """Render the NY market-hours banner. Call once near the top of any page."""
    clock = _get_clock()

    if clock is None:
        st.caption(
            "🕐 Market clock unavailable — set ALPACA_API_KEY/ALPACA_SECRET_KEY in Settings."
        )
        return

    now_ny = clock["timestamp"]
    ny_time_str = now_ny.strftime("%a %b %d, %I:%M:%S %p ET")

    if clock["is_open"]:
        countdown = _format_countdown(clock["next_close"] - now_ny)
        st.caption(f"🟢 Market **OPEN** — {ny_time_str}  ·  closes in {countdown}")
    else:
        next_open_str = clock["next_open"].strftime("%a %I:%M %p ET")
        countdown = _format_countdown(clock["next_open"] - now_ny)
        st.caption(
            f"🔴 Market **CLOSED** — {ny_time_str}  ·  opens {next_open_str} (in {countdown})"
        )
