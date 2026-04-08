"""Phase 6: Insider trading tracker via SEC EDGAR Form 4 filings.

Parses Form 4 filings (Statement of Changes in Beneficial Ownership) from
the SEC EDGAR public API to extract buy/sell transactions by corporate
insiders (CEOs, CFOs, directors, 10%+ shareholders).

Pipeline per ticker:
  1. Resolve ticker -> CIK via EDGAR company tickers JSON (cached in-memory).
  2. Fetch the company's recent filing list from
     https://data.sec.gov/submissions/CIK{padded_cik}.json
  3. Filter for form type "4" within the lookback window.
  4. For each filing, download and parse the Form 4 XML document.
  5. Extract nonDerivativeTransaction records (open-market stock trades).

Storage: data/political/insider/{ticker}.parquet
Columns: owner_name, owner_role, transaction_type (Buy/Sell),
         shares, price_per_share, value, is_buy, accession
         all indexed by transaction date (DatetimeIndex UTC).

Signal output (used by ranker.py and ml/feature_engineer.py):
  insider_net_buys     — number of buy transactions in window
  insider_net_sells    — number of sell transactions in window
  insider_buy_value    — total dollar value of buys
  insider_sell_value   — total dollar value of sells
  insider_signal       — float in [-1, 1]: positive = net bullish activity

SEC EDGAR rate limit: max 10 requests/second. We default to ~6-7/sec
(0.15 s delay between requests) as set by settings.edgar_rate_limit_delay.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

from config.settings import get_settings
from src.scraper.storage import upsert_dataframe, load_dataframe
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------#
# EDGAR URL constants
# ---------------------------------------------------------------------------#

_TICKERS_JSON_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_FILING_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"

# All EDGAR requests must include a contact User-Agent (SEC policy)
_HEADERS = {
    "User-Agent": "AutoStockAnalyzer research@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
_EDGAR_ARCHIVE_HEADERS = {
    "User-Agent": "AutoStockAnalyzer research@example.com",
    "Accept-Encoding": "gzip, deflate",
}


# ---------------------------------------------------------------------------#
# CIK resolution
# ---------------------------------------------------------------------------#

@lru_cache(maxsize=1)
def _load_ticker_cik_map() -> dict[str, str]:
    """Download and cache the SEC's full ticker->CIK mapping.

    Returns:
        Dict mapping uppercase ticker symbol -> zero-padded 10-digit CIK string.
        E.g. {"AAPL": "0000320193", "MSFT": "0000789019"}
    """
    try:
        resp = requests.get(
            _TICKERS_JSON_URL,
            headers={"User-Agent": "AutoStockAnalyzer research@example.com"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        # Format: {str(idx): {"cik_str": "...", "ticker": "...", "title": "..."}, ...}
        return {
            v["ticker"].upper(): str(v["cik_str"]).zfill(10)
            for v in data.values()
            if "ticker" in v and "cik_str" in v
        }
    except Exception as exc:
        logger.error(f"Failed to load SEC ticker->CIK map: {exc}")
        return {}


def _resolve_cik(ticker: str) -> str | None:
    """Return the zero-padded 10-digit CIK for a ticker, or None.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        10-digit CIK string or None if not found.
    """
    mapping = _load_ticker_cik_map()
    cik = mapping.get(ticker.upper())
    if not cik:
        logger.warning(f"[{ticker}] CIK not found in EDGAR ticker map")
    return cik


# ---------------------------------------------------------------------------#
# Submission fetching
# ---------------------------------------------------------------------------#

def _get_recent_form4_filings(
    cik: str,
    lookback_days: int,
    max_filings: int,
    delay: float,
) -> list[dict]:
    """Fetch recent Form 4 filings from the EDGAR submissions endpoint.

    Args:
        cik: Zero-padded 10-digit CIK.
        lookback_days: Only return filings within this many days.
        max_filings: Cap on the number of filings to process.
        delay: Seconds to sleep after the HTTP request.

    Returns:
        List of dicts with keys: accession_clean (no dashes), filing_date,
        primary_document (filename), period_of_report.
    """
    url = _SUBMISSIONS_URL.format(cik=cik)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning(f"[CIK {cik}] Failed to fetch submissions: {exc}")
        return []
    finally:
        time.sleep(delay)

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    periods = recent.get("reportDate", [])

    cutoff = date.today() - timedelta(days=lookback_days)
    filings: list[dict] = []

    for i, form in enumerate(forms):
        if form != "4":
            continue
        try:
            filing_date = date.fromisoformat(dates[i])
        except (IndexError, ValueError):
            continue

        if filing_date < cutoff:
            continue

        accession_raw = accessions[i] if i < len(accessions) else ""
        accession_clean = accession_raw.replace("-", "")
        doc = docs[i] if i < len(docs) else ""
        period = periods[i] if i < len(periods) else ""

        if not accession_clean or not doc:
            continue

        filings.append({
            "accession_clean": accession_clean,
            "filing_date": filing_date.isoformat(),
            "primary_document": doc,
            "period_of_report": period,
        })

        if len(filings) >= max_filings:
            break

    return filings


# ---------------------------------------------------------------------------#
# Form 4 XML parsing
# ---------------------------------------------------------------------------#

def _safe_text(element: ET.Element | None, path: str, default: str = "") -> str:
    """Extract text from an XML sub-element, returning default if missing."""
    if element is None:
        return default
    node = element.find(path)
    if node is None or node.text is None:
        return default
    return node.text.strip()


def _parse_form4_xml(xml_text: str, accession: str) -> list[dict]:
    """Parse a Form 4 XML document into a list of transaction records.

    Only processes nonDerivativeTransaction elements (open-market common
    stock trades). Derivative transactions (options, warrants) are skipped.

    Args:
        xml_text: Raw XML string of the Form 4 filing.
        accession: Filing accession number (for provenance tracking).

    Returns:
        List of transaction dicts with keys:
          - transaction_date (datetime, UTC-aware)
          - owner_name (str)
          - owner_role (str): "CEO", "CFO", "Director", "10%+ Owner", etc.
          - transaction_type (str): "Buy" or "Sell"
          - shares (float | None)
          - price_per_share (float | None)
          - value (float | None): shares * price
          - is_buy (bool)
          - accession (str)
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.debug(f"Form 4 XML parse error ({accession}): {exc}")
        return []

    # Owner info
    owner_el = root.find("./reportingOwner")
    owner_name = _safe_text(owner_el, "reportingOwnerId/rptOwnerName", "Unknown")

    rel_el = owner_el.find("reportingOwnerRelationship") if owner_el is not None else None
    if rel_el is not None:
        if _safe_text(rel_el, "isOfficer") == "1":
            title = _safe_text(rel_el, "officerTitle")
            owner_role = title if title else "Officer"
        elif _safe_text(rel_el, "isDirector") == "1":
            owner_role = "Director"
        elif _safe_text(rel_el, "isTenPercentOwner") == "1":
            owner_role = "10%+ Owner"
        else:
            owner_role = "Other"
    else:
        owner_role = "Unknown"

    records: list[dict] = []

    for txn_el in root.findall("./nonDerivativeTable/nonDerivativeTransaction"):
        try:
            date_str = _safe_text(txn_el, "transactionDate/value")
            if not date_str:
                continue

            txn_date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)

            shares_str = _safe_text(txn_el, "transactionAmounts/transactionShares/value")
            price_str = _safe_text(txn_el, "transactionAmounts/transactionPricePerShare/value")
            code = _safe_text(txn_el, "transactionAmounts/transactionAcquiredDisposedCode/value")

            shares = float(shares_str) if shares_str else None
            price = float(price_str) if price_str else None
            value = (shares * price) if (shares is not None and price is not None) else None

            # A = Acquired (Buy), D = Disposed (Sell)
            is_buy = code.upper() == "A"
            txn_type = "Buy" if is_buy else "Sell"

            records.append({
                "transaction_date": txn_date,
                "owner_name": owner_name,
                "owner_role": owner_role,
                "transaction_type": txn_type,
                "shares": shares,
                "price_per_share": price,
                "value": value,
                "is_buy": is_buy,
                "accession": accession,
            })
        except Exception as exc:
            logger.debug(f"Skipping Form 4 transaction record: {exc}")
            continue

    return records


# ---------------------------------------------------------------------------#
# Main fetch function
# ---------------------------------------------------------------------------#

def fetch_insider_trades(
    ticker: str,
    lookback_days: int | None = None,
    max_filings: int | None = None,
) -> list[dict]:
    """Fetch insider transactions for a single ticker from SEC EDGAR.

    Resolves the ticker to a CIK, pulls the company's recent Form 4 filings,
    downloads each XML, and extracts nonDerivative buy/sell transactions.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL").
        lookback_days: Only return filings filed within this many days.
                       Defaults to settings.insider_lookback_days.
        max_filings: Cap on Form 4 filings to fetch and parse.
                     Defaults to settings.insider_max_filings.

    Returns:
        List of transaction dicts sorted oldest-first.  Each dict has keys:
          transaction_date, owner_name, owner_role, transaction_type,
          shares, price_per_share, value, is_buy, accession.
        Returns an empty list if the ticker is not found or EDGAR is unreachable.
    """
    settings = get_settings()
    days = lookback_days if lookback_days is not None else settings.insider_lookback_days
    max_f = max_filings if max_filings is not None else settings.insider_max_filings
    delay = settings.edgar_rate_limit_delay

    cik = _resolve_cik(ticker)
    if not cik:
        return []

    filings = _get_recent_form4_filings(cik, days, max_f, delay)
    if not filings:
        logger.info(f"[{ticker}] No Form 4 filings found in last {days} days")
        return []

    all_records: list[dict] = []

    for filing in filings:
        accession = filing["accession_clean"]
        doc = filing["primary_document"]

        # Strip any xsl prefix that EDGAR sometimes prepends to the primary doc name
        if "/" in doc:
            doc = doc.split("/")[-1]

        url = _FILING_DOC_URL.format(cik=cik.lstrip("0"), accession=accession, doc=doc)

        try:
            resp = requests.get(url, headers=_EDGAR_ARCHIVE_HEADERS, timeout=20)
            resp.raise_for_status()
            records = _parse_form4_xml(resp.text, accession)
            all_records.extend(records)
            logger.debug(
                f"[{ticker}] Parsed filing {accession}: {len(records)} transactions"
            )
        except Exception as exc:
            logger.debug(f"[{ticker}] Failed to fetch/parse filing {accession}: {exc}")
        finally:
            time.sleep(delay)

    all_records.sort(key=lambda r: r["transaction_date"])
    logger.info(f"[{ticker}] Fetched {len(all_records)} insider transactions from {len(filings)} filings")
    return all_records


def fetch_batch_insider_trades(
    tickers: list[str],
    lookback_days: int | None = None,
    max_filings: int | None = None,
) -> dict[str, list[dict]]:
    """Fetch insider trades for multiple tickers.

    Args:
        tickers: List of ticker symbols.
        lookback_days: Days lookback; defaults to settings value.
        max_filings: Max Form 4 filings per ticker; defaults to settings value.

    Returns:
        Dict mapping ticker -> list of transaction dicts.
    """
    results: dict[str, list[dict]] = {}
    for ticker in tickers:
        results[ticker] = fetch_insider_trades(
            ticker,
            lookback_days=lookback_days,
            max_filings=max_filings,
        )
    return results


# ---------------------------------------------------------------------------#
# Persistence
# ---------------------------------------------------------------------------#

def save_insider_trades(records: list[dict], ticker: str, insider_dir: Path) -> None:
    """Persist insider transaction records to data/political/insider/{ticker}.parquet.

    Args:
        records: Output of :func:`fetch_insider_trades`.
        ticker: Ticker symbol (used for filename).
        insider_dir: Directory for insider Parquet files.
    """
    if not records:
        return

    insider_dir.mkdir(parents=True, exist_ok=True)
    filepath = insider_dir / f"{ticker}.parquet"

    df = pd.DataFrame(records).set_index("transaction_date")
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "date"

    upsert_dataframe(df, filepath)
    logger.debug(f"[{ticker}] Saved {len(records)} insider transactions to {filepath.name}")


def save_batch_insider_trades(
    batch: dict[str, list[dict]],
    insider_dir: Path,
) -> None:
    """Save insider trades for all tickers in a batch result.

    Args:
        batch: Output of :func:`fetch_batch_insider_trades`.
        insider_dir: Storage directory.
    """
    for ticker, records in batch.items():
        if records:
            save_insider_trades(records, ticker, insider_dir)


def load_insider_trades(ticker: str, insider_dir: Path) -> pd.DataFrame:
    """Load all stored insider trades for a ticker.

    Args:
        ticker: Stock ticker symbol.
        insider_dir: Directory containing insider Parquet files.

    Returns:
        DataFrame indexed by transaction date (UTC DatetimeIndex) with columns:
        owner_name, owner_role, transaction_type, shares, price_per_share,
        value, is_buy, accession.
        Empty DataFrame if no data is stored.
    """
    filepath = insider_dir / f"{ticker}.parquet"
    return load_dataframe(filepath)


# ---------------------------------------------------------------------------#
# Signal
# ---------------------------------------------------------------------------#

def get_insider_signal(
    ticker: str,
    insider_dir: Path,
    window_days: int | None = None,
) -> dict:
    """Compute an insider trading signal for the ranker.

    Aggregates buy/sell transactions over the most recent ``window_days``
    and computes a normalised signal in [-1, 1].

    Formula:
      insider_signal = (buy_value - sell_value) / (buy_value + sell_value + 1)

    Positive signal means insiders have been net buyers — historically a
    bullish indicator, especially for CEO/CFO purchases.

    Args:
        ticker: Stock ticker symbol.
        insider_dir: Storage directory.
        window_days: Rolling window. Defaults to settings.insider_lookback_days.

    Returns:
        Dict with keys: ticker, insider_net_buys, insider_net_sells,
        insider_buy_value, insider_sell_value, insider_signal.
    """
    settings = get_settings()
    days = window_days if window_days is not None else settings.insider_lookback_days

    base: dict = {
        "ticker": ticker,
        "insider_net_buys": 0,
        "insider_net_sells": 0,
        "insider_buy_value": 0.0,
        "insider_sell_value": 0.0,
        "insider_signal": 0.0,
    }

    df = load_insider_trades(ticker, insider_dir)
    if df.empty:
        return base

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    recent = df[df.index >= cutoff]
    if recent.empty:
        return base

    buy_rows = recent[recent["is_buy"] == True]   # noqa: E712
    sell_rows = recent[recent["is_buy"] == False]  # noqa: E712

    buy_value = buy_rows["value"].fillna(0).sum()
    sell_value = sell_rows["value"].fillna(0).sum()

    total = buy_value + sell_value
    signal = (buy_value - sell_value) / (total + 1.0)

    return {
        "ticker": ticker,
        "insider_net_buys": int(len(buy_rows)),
        "insider_net_sells": int(len(sell_rows)),
        "insider_buy_value": float(buy_value),
        "insider_sell_value": float(sell_value),
        "insider_signal": float(signal),
    }
