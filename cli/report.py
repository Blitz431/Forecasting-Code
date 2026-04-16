"""CLI entry point for morning reports and alert dispatch.

Usage
-----
    python cli/report.py --morning
        Generate today's PDF + Excel report, then send a Discord/Telegram summary
        (title + top 3 picks + regime).

    python cli/report.py --alerts
        Run trigger checks, print all active alerts to stdout, and dispatch
        them via notifier.py.

    python cli/report.py --test-notify
        Send a test message to all configured channels to verify credentials.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make project root importable when running as a script
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------#
# Sub-commands
# ---------------------------------------------------------------------------#

def cmd_morning(settings) -> int:
    """Generate PDF + Excel report and send a Discord/Telegram summary."""
    from src.reports.morning_report import MorningReport
    from src.alerts.notifier import AlertNotifier

    print("[report] Generating morning report...")
    report = MorningReport(settings)

    try:
        paths = report.generate()
        print(f"  PDF  saved: {paths['pdf']}")
        print(f"  Excel saved: {paths['xlsx']}")
    except Exception as exc:
        print(f"[report] ERROR generating report: {exc}", file=sys.stderr)
        logger.error(f"Morning report generation failed: {exc}")
        return 1

    # Send summary to Discord / Telegram
    print("[report] Sending Discord/Telegram summary...")
    title, body = report.discord_summary()

    notifier = AlertNotifier(settings)
    results = notifier.send_alert(title, body, level="info")

    for channel, ok in results.items():
        status = "sent" if ok else "skipped/failed"
        print(f"  {channel}: {status}")

    return 0


def cmd_alerts(settings) -> int:
    """Run trigger checks, print to stdout, and dispatch via notifier."""
    from src.alerts.triggers import TriggerChecker
    from src.alerts.notifier import AlertNotifier

    print("[report] Checking alert triggers...")
    checker = TriggerChecker(settings)
    events = checker.check_all()

    if not events:
        print("  No active alerts.")
        return 0

    print(f"  {len(events)} alert(s) found:\n")
    for i, evt in enumerate(events, 1):
        ticker_tag = f" [{evt.ticker}]" if evt.ticker else ""
        print(f"  {i}. [{evt.level.upper()}] {evt.title}{ticker_tag}")
        for line in evt.body.splitlines():
            print(f"     {line}")
        print()

    # Dispatch each alert
    print("[report] Dispatching alerts...")
    notifier = AlertNotifier(settings)
    for evt in events:
        results = notifier.send_alert(evt.title, evt.body, level=evt.level)
        channels_sent = [k for k, v in results.items() if v]
        print(f"  '{evt.title}' dispatched via: {channels_sent or 'none'}")

    return 0


def cmd_test_notify(settings) -> int:
    """Send a test message to all configured channels."""
    from src.alerts.notifier import AlertNotifier

    print("[report] Sending test notifications...")
    notifier = AlertNotifier(settings)
    results = notifier.send_test()

    any_sent = False
    for channel, ok in results.items():
        status = "OK" if ok else "skipped (no credentials) / FAILED"
        print(f"  {channel}: {status}")
        if ok:
            any_sent = True

    if not any_sent:
        print("\n  No channels sent. Check your .env credentials:")
        print("    DISCORD_WEBHOOK_URL, TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID,")
        print("    SMTP_HOST + SMTP_USER + SMTP_PASSWORD + ALERT_EMAIL_TO")

    return 0 if any_sent else 1


# ---------------------------------------------------------------------------#
# Main
# ---------------------------------------------------------------------------#

def main() -> int:
    parser = argparse.ArgumentParser(
        description="AutoStockAnalyzer — report and alert CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--morning",
        action="store_true",
        help="Generate PDF + Excel morning report and send summary to Discord/Telegram",
    )
    group.add_argument(
        "--alerts",
        action="store_true",
        help="Run trigger checks, print active alerts, and dispatch via notifier",
    )
    group.add_argument(
        "--test-notify",
        action="store_true",
        dest="test_notify",
        help="Send a test message to all configured channels",
    )

    args = parser.parse_args()
    settings = get_settings()

    if args.morning:
        return cmd_morning(settings)
    elif args.alerts:
        return cmd_alerts(settings)
    elif args.test_notify:
        return cmd_test_notify(settings)

    return 0


if __name__ == "__main__":
    sys.exit(main())
