"""Page 18 — Alerts.

Left panel  : Active alerts table from TriggerChecker.check_all().
              Rows are colour-coded by level (info=blue, warning=orange, critical=red).
              "Run Trigger Check" button refreshes the table live.

Right panel : Scrollable log viewer showing the last 100 lines of
              data/alerts/alerts.log.
              "Send Test Notification" button calls AlertNotifier.send_test().
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings

st.set_page_config(page_title="Alerts", page_icon="🔔", layout="wide")

from dashboard.components.market_clock import render_market_clock
render_market_clock()

st.title("🔔 Alerts")
st.caption("Active trigger alerts and notification log.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Level → CSS colour
# ---------------------------------------------------------------------------#

_LEVEL_STYLE: dict[str, str] = {
    "info":     "background-color:#1a3a5c; color:#aed6f1",
    "warning":  "background-color:#5c3a00; color:#fad7a0",
    "critical": "background-color:#5c0000; color:#f1948a",
}

_LEVEL_ICON: dict[str, str] = {
    "info":     "ℹ️",
    "warning":  "⚠️",
    "critical": "🚨",
}


def _style_level(val: str) -> str:
    return _LEVEL_STYLE.get(str(val).lower(), "")


# ---------------------------------------------------------------------------#
# Two-column layout
# ---------------------------------------------------------------------------#

col_left, col_right = st.columns([3, 2], gap="large")

# ---------------------------------------------------------------------------#
# LEFT — active alerts
# ---------------------------------------------------------------------------#

with col_left:
    st.subheader("Active Alerts")

    if st.button("Run Trigger Check", type="primary"):
        st.cache_data.clear()

    @st.cache_data(ttl=60, show_spinner="Checking triggers …")
    def _load_alerts() -> list[dict]:
        try:
            from src.alerts.triggers import TriggerChecker
            from src.trading.portfolio import PortfolioTracker

            tracker = PortfolioTracker(settings)
            snap = tracker.get_snapshot()
            held = [p.ticker for p in snap.positions] if snap and snap.positions else []

            checker = TriggerChecker(settings)
            events = checker.check_all(held)
            return [
                {
                    "level":   evt.level,
                    "trigger": evt.trigger_type,
                    "ticker":  evt.ticker or "—",
                    "title":   evt.title,
                    "body":    evt.body,
                }
                for evt in events
            ]
        except Exception as exc:
            return [
                {
                    "level":   "critical",
                    "trigger": "error",
                    "ticker":  "—",
                    "title":   "Trigger check failed",
                    "body":    str(exc),
                }
            ]

    alerts = _load_alerts()

    if not alerts:
        st.success("No active alerts.")
    else:
        st.metric("Active alerts", len(alerts))

        df = pd.DataFrame(alerts)
        styled = (
            df[["level", "trigger", "ticker", "title"]]
            .style
            .map(_style_level, subset=["level"])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("**Details**")
        for row in alerts:
            icon = _LEVEL_ICON.get(row["level"], "")
            with st.expander(f"{icon} {row['title']}"):
                st.text(row["body"])

# ---------------------------------------------------------------------------#
# RIGHT — alerts log + test notification
# ---------------------------------------------------------------------------#

with col_right:
    st.subheader("Alerts Log")

    log_path: Path = settings.data_dir / "alerts" / "alerts.log"

    if st.button("Send Test Notification"):
        with st.spinner("Sending …"):
            try:
                from src.alerts.notifier import AlertNotifier

                notifier = AlertNotifier(settings)
                results = notifier.send_test()
                for channel, ok in results.items():
                    if ok:
                        st.success(f"{channel}: sent")
                    else:
                        st.warning(f"{channel}: skipped / failed (no credentials?)")
            except Exception as exc:
                st.error(f"send_test() raised: {exc}")

    st.caption(f"`{log_path}`")

    if log_path.exists():
        raw = log_path.read_text(encoding="utf-8", errors="replace")
        lines = raw.splitlines()
        last_100 = "\n".join(lines[-100:]) if lines else "(empty)"
        st.text_area("Last 100 lines", value=last_100, height=480, disabled=True, key="alerts_last_100_lines")
    else:
        st.info(
            "No alerts log yet.  "
            "Critical alerts are written here automatically when triggered.  "
            "Configure Discord or email in `.env` to enable notifications."
        )
