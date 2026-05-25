"""Multi-channel alert dispatcher.

Supported channels
------------------
- Discord  : discord-webhook  (DISCORD_WEBHOOK_URL in .env)
- Email    : smtplib  (SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD / ALERT_EMAIL_TO)

All credentials are loaded from ``config/settings.py`` → ``AlertSettings``,
which reads them from the ``.env`` file.  Any channel whose credentials are
missing is silently skipped.

Public API
----------
    notifier = AlertNotifier(settings)
    notifier.send_alert("Trade Signal", "AAPL BUY triggered", level="info")

``level`` is one of:
  "info"     — Discord embed: blue  (0x3498DB)
  "warning"  — Discord embed: yellow (0xF1C40F)
  "critical" — Discord embed: red  (0xE74C3C) + also appended to data/alerts/alerts.log

Critical alerts are logged to ``data/alerts/alerts.log`` regardless of whether
any channel is configured.
"""

from __future__ import annotations

import logging
import smtplib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

from config.settings import get_settings
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------#
# Colour map for Discord embeds
# ---------------------------------------------------------------------------#

_DISCORD_COLORS: dict[str, int] = {
    "info":     0x3498DB,   # blue
    "warning":  0xF1C40F,   # yellow
    "critical": 0xE74C3C,   # red
}

_LEVEL_EMOJI: dict[str, str] = {
    "info":     "ℹ️",
    "warning":  "⚠️",
    "critical": "🚨",
}


# ---------------------------------------------------------------------------#
# AlertNotifier
# ---------------------------------------------------------------------------#

class AlertNotifier:
    """Dispatch alerts to Discord and/or email in parallel."""

    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        self._s = settings
        self._a = settings.alerts
        self._log_dir: Path = settings.data_dir / "alerts"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file: Path = self._log_dir / "alerts.log"

    # ---------------------------------------------------------------------- #
    # Public interface
    # ---------------------------------------------------------------------- #

    def send_alert(self, title: str, body: str, level: str = "info") -> dict[str, bool]:
        """Send an alert to all configured channels in parallel.

        Parameters
        ----------
        title:  Short headline (shown in Discord embed title, email subject).
        body:   Full alert message.
        level:  "info" | "warning" | "critical"

        Returns
        -------
        Dict mapping channel name -> True (sent) / False (failed/skipped).
        """
        level = level.lower()
        if level not in _DISCORD_COLORS:
            level = "info"

        if level == "critical":
            self._log_to_file(title, body)

        senders = {
            "discord": self._send_discord,
            "email":   self._send_email,
        }

        results: dict[str, bool] = {}

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(fn, title, body, level): name
                for name, fn in senders.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    logger.error(f"[{name}] Unexpected error: {exc}")
                    results[name] = False

        sent = [k for k, v in results.items() if v]
        skipped = [k for k, v in results.items() if not v]
        logger.info(
            f"Alert '{title}' [{level}] — sent via {sent or 'none'}"
            + (f" | skipped: {skipped}" if skipped else "")
        )
        return results

    def send_test(self) -> dict[str, bool]:
        """Send a test message to all configured channels."""
        return self.send_alert(
            title="Test Alert — AutoStockAnalyzer",
            body="This is a test notification. All channels are working correctly.",
            level="info",
        )

    # ---------------------------------------------------------------------- #
    # Discord
    # ---------------------------------------------------------------------- #

    def _send_discord(self, title: str, body: str, level: str) -> bool:
        """Send a rich embed to a Discord webhook. Returns True on success."""
        url = self._a.discord_webhook_url
        if not url:
            return False

        try:
            from discord_webhook import DiscordWebhook, DiscordEmbed

            webhook = DiscordWebhook(url=url)
            embed = DiscordEmbed(
                title=f"{_LEVEL_EMOJI.get(level, '')} {title}",
                description=body[:4096],   # Discord embed description limit
                color=_DISCORD_COLORS.get(level, 0x3498DB),
            )
            embed.set_footer(text=f"AutoStockAnalyzer • {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
            embed.set_timestamp()
            webhook.add_embed(embed)

            response = webhook.execute()
            # discord_webhook raises on HTTP errors; if we get here it succeeded
            return True
        except Exception as exc:
            logger.error(f"[Discord] send failed: {exc}")
            return False

    # ---------------------------------------------------------------------- #
    # Email
    # ---------------------------------------------------------------------- #

    def _send_email(self, title: str, body: str, level: str) -> bool:
        """Send an email via SMTP. Returns True on success."""
        host = self._a.smtp_host
        user = self._a.smtp_user
        password = self._a.smtp_password
        to_addr = self._a.alert_email_to

        if not all([host, user, password, to_addr]):
            return False

        try:
            subject = f"[AutoStockAnalyzer] [{level.upper()}] {title}"
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = user
            msg["To"] = to_addr

            with smtplib.SMTP(host, self._a.smtp_port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.login(user, password)
                server.sendmail(user, [to_addr], msg.as_string())

            return True
        except Exception as exc:
            logger.error(f"[Email] send failed: {exc}")
            return False

    # ---------------------------------------------------------------------- #
    # File log (critical alerts only)
    # ---------------------------------------------------------------------- #

    def _log_to_file(self, title: str, body: str) -> None:
        """Append a critical alert to data/alerts/alerts.log."""
        try:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            entry = (
                f"[{timestamp}] CRITICAL | {title}\n"
                f"{body}\n"
                f"{'-' * 80}\n"
            )
            with self._log_file.open("a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as exc:
            logger.error(f"[alerts.log] write failed: {exc}")


# ---------------------------------------------------------------------------#
# Helpers
# ---------------------------------------------------------------------------#

