"""Daily morning report generator.

Produces two files saved to ``data/reports/``:
  - ``morning_YYYY-MM-DD.pdf``   (fpdf2)
  - ``morning_YYYY-MM-DD.xlsx``  (openpyxl, one sheet per section)

Report sections (in order)
--------------------------
1. Header — date, market regime (Bull/Bear/Sideways), VIX, yield spread
2. Top 20 ranked picks — Ticker, Score, Rank, key signals
3. Open portfolio positions — ticker, qty, entry, current, unrealized P&L
4. Trade journal summary — last 5 trades + win rate + total P&L
5. Upcoming earnings in next 7 days for held tickers
6. Active alerts from triggers.py

Public API
----------
    report = MorningReport(settings)
    paths  = report.generate(held_tickers=["AAPL", "MSFT"])
    # returns {"pdf": Path, "xlsx": Path}
"""

from __future__ import annotations

import io
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# fpdf2's default Helvetica font uses latin-1 encoding and cannot render
# em/en-dashes, smart quotes, or other common Unicode punctuation. Map them
# to ASCII equivalents before passing any string to pdf.cell/multi_cell.
_PDF_REPLACEMENTS = {
    "\u2014": "-",    # em-dash
    "\u2013": "-",    # en-dash
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote / apostrophe
    "\u201c": '"',    # left double quote
    "\u201d": '"',    # right double quote
    "\u2026": "...",  # ellipsis
    "\u00a0": " ",    # non-breaking space
    "\u2022": "*",    # bullet
}


def _pdf_safe(text) -> str:
    """Return a latin-1-safe version of ``text`` for fpdf2 Helvetica."""
    if text is None:
        return ""
    s = str(text)
    for src, dst in _PDF_REPLACEMENTS.items():
        if src in s:
            s = s.replace(src, dst)
    # Final fallback: drop any remaining non-latin-1 characters.
    return s.encode("latin-1", errors="replace").decode("latin-1")

# ---------------------------------------------------------------------------#
# Section data builders (each returns a DataFrame or dict)
# ---------------------------------------------------------------------------#

def _get_regime_data(settings) -> dict:
    """Fetch current market regime snapshot."""
    try:
        from src.analytics.market_regime import MarketRegimeAnalyzer
        analyzer = MarketRegimeAnalyzer(settings)
        snap = analyzer.current_snapshot()
        return {
            "regime":       snap.regime.value if snap.regime else "Unknown",
            "vix":          snap.vix,
            "yield_spread": snap.yield_spread,
            "score":        snap.score,
        }
    except Exception as exc:
        logger.warning(f"[morning_report] regime: {exc}")
        return {"regime": "Unknown", "vix": None, "yield_spread": None, "score": None}


def _get_top_picks(settings) -> pd.DataFrame:
    """Return top-N ranked picks as a DataFrame."""
    try:
        from src.ranking.ranker import top_picks, to_dataframe
        picks = top_picks(settings.top_n_picks, settings)
        df = to_dataframe(picks)
        return df
    except Exception as exc:
        logger.warning(f"[morning_report] top_picks: {exc}")
        return pd.DataFrame()


def _get_portfolio_positions(settings) -> pd.DataFrame:
    """Return current open positions with P&L."""
    try:
        from src.trading.portfolio import PortfolioTracker
        tracker = PortfolioTracker(settings)
        snap = tracker.get_snapshot()
        if snap is None or not snap.positions:
            return pd.DataFrame()
        rows = []
        for pos in snap.positions:
            rows.append({
                "Ticker":          pos.ticker,
                "Qty":             pos.qty,
                "Avg Entry":       pos.avg_entry_price,
                "Current Price":   pos.current_price,
                "Market Value":    pos.market_value,
                "Unrealized P&L":  pos.unrealized_pl,
                "Unrealized P&L%": pos.unrealized_plpc,
            })
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning(f"[morning_report] portfolio: {exc}")
        return pd.DataFrame()


def _get_trade_journal_summary(settings) -> tuple[pd.DataFrame, dict]:
    """Return last 5 trades + summary stats."""
    try:
        from src.trading.trade_journal import TradeJournal
        journal = TradeJournal(settings)
        df = journal.load()
        stats = journal.summary_stats()
        last5 = df.head(5)[["timestamp", "ticker", "side", "price", "shares",
                              "realized_pnl", "exit_reason"]].copy()
        return last5, stats
    except Exception as exc:
        logger.warning(f"[morning_report] journal: {exc}")
        return pd.DataFrame(), {}


def _get_upcoming_earnings(held_tickers: list[str], settings) -> pd.DataFrame:
    """Return earnings within 7 days for held tickers."""
    try:
        from src.calendar.earnings import get_earnings_signal
        rows = []
        for ticker in held_tickers:
            sig = get_earnings_signal(ticker, settings.calendar_dir)
            days = sig.get("days_to_earnings")
            if days is not None and 0 <= days <= 7:
                rows.append({
                    "Ticker":           ticker,
                    "Days Away":        days,
                    "Earnings Date":    sig.get("next_earnings_date", "N/A"),
                    "Beat Rate":        sig.get("beat_rate"),
                    "Avg EPS Surprise": sig.get("avg_eps_surprise_pct"),
                })
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning(f"[morning_report] earnings: {exc}")
        return pd.DataFrame()


def _get_active_alerts(held_tickers: list[str], settings) -> list:
    """Run trigger checks and return list of AlertEvent objects."""
    try:
        from src.alerts.triggers import TriggerChecker
        checker = TriggerChecker(settings)
        return checker.check_all(held_tickers)
    except Exception as exc:
        logger.warning(f"[morning_report] alerts: {exc}")
        return []


# ---------------------------------------------------------------------------#
# PDF builder
# ---------------------------------------------------------------------------#

def _build_pdf(
    report_date: str,
    regime: dict,
    top_picks: pd.DataFrame,
    positions: pd.DataFrame,
    last5_trades: pd.DataFrame,
    journal_stats: dict,
    earnings: pd.DataFrame,
    alerts: list,
) -> bytes:
    """Build the PDF report and return raw bytes."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---- helper callables ----

    def section_title(text: str):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(40, 40, 80)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 8, _pdf_safe(text), new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    def body_text(text: str, bold: bool = False):
        pdf.set_font("Helvetica", "B" if bold else "", 10)
        pdf.multi_cell(0, 6, _pdf_safe(text))
        pdf.ln(1)

    def dataframe_table(df: pd.DataFrame, col_widths: list[float] | None = None):
        if df.empty:
            body_text("No data available.")
            return

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(200, 200, 230)
        cols = list(df.columns)
        n = len(cols)
        w = col_widths if col_widths else [pdf.epw / n] * n

        for i, col in enumerate(cols):
            pdf.cell(w[i], 6, _pdf_safe(str(col)[:20]), border=1, fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for _, row in df.iterrows():
            for i, col in enumerate(cols):
                val = row[col]
                if isinstance(val, float):
                    cell_str = f"{val:.2f}" if abs(val) < 1_000_000 else f"{val:,.0f}"
                else:
                    cell_str = str(val)[:22] if val is not None else ""
                pdf.cell(w[i], 5, _pdf_safe(cell_str), border=1)
            pdf.ln()
        pdf.ln(3)

    # ---- Section 1: Header ----
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, _pdf_safe("AutoStockAnalyzer - Morning Report"), new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, _pdf_safe(f"Generated: {report_date}"), new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    section_title("1. Market Overview")
    vix_str = f"{regime['vix']:.1f}" if regime.get("vix") is not None else "N/A"
    spread_str = f"{regime['yield_spread']:.2f}%" if regime.get("yield_spread") is not None else "N/A"
    body_text(f"Regime: {regime.get('regime', 'Unknown')}    VIX: {vix_str}    10Y-2Y Spread: {spread_str}")

    # ---- Section 2: Top 20 picks ----
    section_title("2. Top 20 Ranked Picks")
    if not top_picks.empty:
        display_cols = ["ticker", "rank", "composite_score"]
        extra_signal_cols = ["indicator_score", "forecast_signal", "ml_signal",
                              "news_sentiment", "congress_signal"]
        for c in extra_signal_cols:
            if c in top_picks.columns:
                display_cols.append(c)
        show_df = top_picks[[c for c in display_cols if c in top_picks.columns]].head(20)
        show_df.columns = [c.replace("_", " ").title() for c in show_df.columns]
        dataframe_table(show_df)
    else:
        body_text("Ranking data not available.")

    # ---- Section 3: Portfolio positions ----
    pdf.add_page()
    section_title("3. Open Portfolio Positions")
    dataframe_table(positions)

    # ---- Section 4: Trade journal ----
    section_title("4. Trade Journal Summary")
    if journal_stats:
        body_text(
            f"Total Trades: {journal_stats.get('total_trades', 0)}   "
            f"Win Rate: {journal_stats.get('win_rate', 0):.1%}   "
            f"Total P&L: ${journal_stats.get('total_pnl', 0):+,.2f}"
        )
    body_text("Last 5 Trades:", bold=True)
    dataframe_table(last5_trades)

    # ---- Section 5: Upcoming earnings ----
    section_title("5. Upcoming Earnings (next 7 days)")
    dataframe_table(earnings)

    # ---- Section 6: Active alerts ----
    section_title("6. Active Alerts")
    if not alerts:
        body_text("No active alerts.")
    else:
        for evt in alerts:
            level_tag = f"[{evt.level.upper()}]"
            ticker_tag = f" ({evt.ticker})" if evt.ticker else ""
            body_text(f"{level_tag} {evt.title}{ticker_tag}", bold=True)
            body_text(evt.body)
            pdf.ln(1)

    return bytes(pdf.output())


# ---------------------------------------------------------------------------#
# Excel builder
# ---------------------------------------------------------------------------#

def _build_xlsx(
    report_date: str,
    regime: dict,
    top_picks: pd.DataFrame,
    positions: pd.DataFrame,
    last5_trades: pd.DataFrame,
    journal_stats: dict,
    earnings: pd.DataFrame,
    alerts: list,
) -> bytes:
    """Build the Excel workbook and return raw bytes."""
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows

    wb = Workbook()

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="28285050")

    def _write_df(ws, df: pd.DataFrame, start_row: int = 1):
        if df.empty:
            ws.cell(row=start_row, column=1, value="No data available.")
            return
        for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True)):
            for c_idx, val in enumerate(row, 1):
                cell = ws.cell(row=start_row + r_idx - 1, column=c_idx, value=val)
                if r_idx == 1:
                    cell.font = header_font
                    cell.fill = PatternFill("solid", fgColor="28284F")
                    cell.alignment = Alignment(horizontal="center")

    # Sheet 1: Overview
    ws1 = wb.active
    ws1.title = "1_Overview"
    ws1["A1"] = "AutoStockAnalyzer Morning Report"
    ws1["A1"].font = Font(bold=True, size=14)
    ws1["A2"] = f"Date: {report_date}"
    ws1["A4"] = "Market Regime"
    ws1["B4"] = regime.get("regime", "Unknown")
    ws1["A5"] = "VIX"
    ws1["B5"] = regime.get("vix")
    ws1["A6"] = "10Y-2Y Yield Spread"
    ws1["B6"] = regime.get("yield_spread")

    # Sheet 2: Top Picks
    ws2 = wb.create_sheet("2_Top_Picks")
    _write_df(ws2, top_picks.head(20) if not top_picks.empty else top_picks)

    # Sheet 3: Portfolio
    ws3 = wb.create_sheet("3_Portfolio")
    _write_df(ws3, positions)

    # Sheet 4: Trade Journal
    ws4 = wb.create_sheet("4_Trade_Journal")
    if journal_stats:
        ws4["A1"] = f"Total Trades: {journal_stats.get('total_trades', 0)}"
        ws4["A2"] = f"Win Rate: {journal_stats.get('win_rate', 0):.1%}"
        ws4["A3"] = f"Total P&L: ${journal_stats.get('total_pnl', 0):+,.2f}"
    _write_df(ws4, last5_trades, start_row=5)

    # Sheet 5: Earnings
    ws5 = wb.create_sheet("5_Earnings")
    _write_df(ws5, earnings)

    # Sheet 6: Alerts
    ws6 = wb.create_sheet("6_Alerts")
    if not alerts:
        ws6["A1"] = "No active alerts."
    else:
        alert_rows = []
        for evt in alerts:
            alert_rows.append({
                "Level":        evt.level.upper(),
                "Trigger":      evt.trigger_type,
                "Ticker":       evt.ticker or "",
                "Title":        evt.title,
                "Body":         evt.body,
            })
        _write_df(ws6, pd.DataFrame(alert_rows))

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------#
# MorningReport
# ---------------------------------------------------------------------------#

class MorningReport:
    """Generate the daily morning PDF + Excel report."""

    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        self._s = settings
        self._reports_dir: Path = settings.data_dir / "reports"
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, held_tickers: list[str] | None = None) -> dict[str, Path]:
        """Build and save both files. Returns {"pdf": Path, "xlsx": Path}.

        Parameters
        ----------
        held_tickers:
            Tickers currently held in portfolio.  If None, loads from latest
            portfolio snapshot automatically.
        """
        report_date = date.today().isoformat()
        logger.info(f"[morning_report] Generating report for {report_date}")

        # Load held tickers if not provided
        if held_tickers is None:
            held_tickers = self._load_held_tickers()

        # Gather all section data
        regime = _get_regime_data(self._s)
        top_picks_df = _get_top_picks(self._s)
        positions_df = _get_portfolio_positions(self._s)
        last5_df, journal_stats = _get_trade_journal_summary(self._s)
        earnings_df = _get_upcoming_earnings(held_tickers, self._s)
        alerts = _get_active_alerts(held_tickers, self._s)

        # Build PDF
        pdf_path = self._reports_dir / f"morning_{report_date}.pdf"
        try:
            pdf_bytes = _build_pdf(
                report_date, regime, top_picks_df, positions_df,
                last5_df, journal_stats, earnings_df, alerts,
            )
            pdf_path.write_bytes(pdf_bytes)
            logger.info(f"[morning_report] PDF saved: {pdf_path}")
        except Exception as exc:
            logger.error(f"[morning_report] PDF generation failed: {exc}")
            raise

        # Build Excel
        xlsx_path = self._reports_dir / f"morning_{report_date}.xlsx"
        try:
            xlsx_bytes = _build_xlsx(
                report_date, regime, top_picks_df, positions_df,
                last5_df, journal_stats, earnings_df, alerts,
            )
            xlsx_path.write_bytes(xlsx_bytes)
            logger.info(f"[morning_report] Excel saved: {xlsx_path}")
        except Exception as exc:
            logger.error(f"[morning_report] Excel generation failed: {exc}")
            raise

        return {"pdf": pdf_path, "xlsx": xlsx_path}

    def discord_summary(
        self,
        held_tickers: list[str] | None = None,
    ) -> tuple[str, str]:
        """Build a short Discord-ready summary (title + body with top 3 picks + regime).

        Returns (title, body) strings suitable for passing to AlertNotifier.send_alert().
        """
        report_date = date.today().isoformat()
        regime = _get_regime_data(self._s)
        top_picks_df = _get_top_picks(self._s)

        regime_str = regime.get("regime", "Unknown")
        vix_str = f"{regime['vix']:.1f}" if regime.get("vix") is not None else "N/A"

        lines = [
            f"Morning Report — {report_date}",
            f"Regime: {regime_str}  |  VIX: {vix_str}",
            "",
            "Top 3 Picks:",
        ]

        if not top_picks_df.empty:
            top3 = top_picks_df.head(3)
            for _, row in top3.iterrows():
                ticker = row.get("ticker", row.get("Ticker", "?"))
                score = row.get("composite_score", row.get("Composite Score", 0.0))
                rank = row.get("rank", row.get("Rank", "?"))
                lines.append(f"  #{rank}  {ticker}  score={score:.3f}")
        else:
            lines.append("  (ranking data not available)")

        return "AutoStockAnalyzer Morning Report", "\n".join(lines)

    # ---------------------------------------------------------------------- #
    # Helper
    # ---------------------------------------------------------------------- #

    def _load_held_tickers(self) -> list[str]:
        snapshots_dir: Path = self._s.portfolio_snapshots_dir
        if not snapshots_dir.exists():
            return []
        files = sorted(snapshots_dir.glob("*.parquet"))
        if not files:
            return []
        try:
            df = pd.read_parquet(files[-1])
            if "ticker" in df.columns:
                return df["ticker"].dropna().unique().tolist()
        except Exception as exc:
            logger.warning(f"[morning_report] load_held_tickers: {exc}")
        return []
