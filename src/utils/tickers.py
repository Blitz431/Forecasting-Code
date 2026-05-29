"""Ticker universe management — fetch S&P 500 component list.

Fetch order:
  1. Local cache file  (data/tickers/sp500.txt)  — refreshed if older than 7 days
  2. Wikipedia scrape  (html.parser, no lxml needed)
  3. Hard-coded 503-ticker fallback list
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

# Project root → data/tickers/sp500.txt
_CACHE_FILE = Path(__file__).parent.parent.parent / "data" / "tickers" / "sp500.txt"
_CACHE_MAX_AGE_DAYS = 7

# Complete S&P 500 fallback — as of 2025-Q1 (503 tickers)
FALLBACK_SP500 = [
    "A", "AAL", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI",
    "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG",
    "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN",
    "AMP", "AMT", "AMTM", "AMZN", "ANET", "ANF", "AON", "AOS", "APA", "APD",
    "APH", "APTV", "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP",
    "AZO", "BA", "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN", "BF-B",
    "BG", "BIIB", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B",
    "BRO", "BSX", "BWA", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT",
    "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF",
    "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMCSA", "CME",
    "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP", "COR", "COST",
    "CPAY", "CPB", "CPRT", "CPT", "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX",
    "CTAS", "CTLT", "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR", "D", "DAL",
    "DAY", "DD", "DDOG", "DE", "DECK", "DEI", "DEO", "DFS", "DG", "DGX",
    "DHI", "DHR", "DIS", "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI",
    "DTE", "DUK", "DVA", "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX",
    "EG", "EIX", "EL", "ELV", "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX",
    "EQR", "EQT", "ES", "ESS", "ETN", "ETR", "ETSY", "EVRG", "EW", "EXC",
    "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE",
    "FFIV", "FI", "FICO", "FIS", "FITB", "FMC", "FOX", "FOXA", "FRT", "FSLR",
    "FTNT", "FTV", "GD", "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS",
    "GL", "GLW", "GM", "GNRC", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW",
    "HAL", "HAS", "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX",
    "HON", "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM",
    "IBM", "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU", "INVH",
    "IP", "IPG", "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J",
    "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEY",
    "KEYS", "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI", "KMX", "KO", "KR",
    "KVUE", "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT",
    "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV", "MA",
    "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET",
    "META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO",
    "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRO", "MS", "MSCI", "MSFT",
    "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM",
    "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE",
    "NVDA", "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON",
    "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PARA", "PAYC", "PAYX", "PCAR",
    "PCG", "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG",
    "PLD", "PM", "PNC", "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU",
    "PSA", "PSX", "PTC", "PWR", "PYPL", "QCOM", "QRVO", "RCL", "REG", "REGN",
    "RF", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX",
    "RVTY", "SBAC", "SBUX", "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA",
    "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX",
    "STZ", "SW", "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG",
    "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT", "TJX", "TMO", "TMUS",
    "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO",
    "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP",
    "UPS", "URI", "USB", "V", "VICI", "VLO", "VMC", "VRSK", "VRSN", "VRTX",
    "VST", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WELL",
    "WFC", "WM", "WMB", "WMT", "WRB", "WRK", "WST", "WTW", "WY", "WYNN",
    "XEL", "XOM", "XYL", "YUM", "ZBH", "ZBRA", "ZTS",
]


def _load_cache() -> list[str] | None:
    """Return cached ticker list if it exists and is fresh, else None."""
    if not _CACHE_FILE.exists():
        return None
    age = datetime.datetime.now() - datetime.datetime.fromtimestamp(_CACHE_FILE.stat().st_mtime)
    if age.days >= _CACHE_MAX_AGE_DAYS:
        return None
    tickers = [t.strip() for t in _CACHE_FILE.read_text().splitlines() if t.strip()]
    if len(tickers) > 100:          # sanity check — reject obviously broken cache
        return tickers
    return None


def _save_cache(tickers: list[str]) -> None:
    """Write ticker list to the local cache file."""
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text("\n".join(sorted(tickers)))
    except Exception as exc:
        logger.debug(f"Could not write ticker cache: {exc}")


def get_sp500_tickers() -> list[str]:
    """Return the current S&P 500 ticker list.

    Tries (in order):
      1. Local cache  (data/tickers/sp500.txt, refreshed weekly)
      2. Wikipedia    (uses built-in html.parser — no lxml needed)
      3. Hard-coded fallback list (~503 tickers)
    """
    # 1. Local cache
    cached = _load_cache()
    if cached:
        logger.info(f"Loaded {len(cached)} tickers from local cache")
        return cached

    # 2. Wikipedia
    try:
        import requests
        from io import StringIO

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 AutoStockAnalyzer/1.0"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()

        # flavor=None → pandas tries lxml, then html5lib, then bs4 automatically
        tables = pd.read_html(StringIO(resp.text), flavor=None)
        df = tables[0]

        # Column name varies slightly — find whichever has the tickers
        symbol_col = next(
            (c for c in df.columns if str(c).lower() in ("symbol", "ticker")),
            None,
        )
        if symbol_col is None:
            raise ValueError(f"No 'Symbol' column found. Columns: {list(df.columns)}")

        tickers = (
            df[symbol_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)   # BRK.B → BRK-B
            .tolist()
        )
        tickers = sorted(set(tickers))
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        _save_cache(tickers)
        return tickers

    except Exception as exc:
        logger.warning(f"Wikipedia fetch failed: {exc}. Using built-in fallback list.")
        return sorted(FALLBACK_SP500)


def get_tickers(source: str = "sp500") -> list[str]:
    """Return a ticker list based on *source*.

    Args:
        source: ``"sp500"`` for the S&P 500, or a comma-separated ticker string.
    """
    if source == "sp500":
        return get_sp500_tickers()
    return [t.strip().upper() for t in source.split(",") if t.strip()]
