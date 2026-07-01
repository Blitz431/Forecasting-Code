from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import candlestick, price_line, _empty_fig
from dashboard.components.tables import style_generic

st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")

from dashboard.components.market_clock import render_market_clock
render_market_clock()

st.title("📊 Data Overview")
st.caption("Browse available OHLCV price data, date ranges, and data quality statistics.")
st.divider()

settings = get_settings()

"""
Purpose: Streamlit page — browse all scraped tickers (date ranges, row counts, quality) and drill into OHLCV candlestick charts.

Connections:
  - dashboard/components/charts.py: candlestick() and price_line() for drill-down view
  - dashboard/components/tables.py: style_generic() for the inventory table
  - config/settings.py: raw_daily_dir, raw_quarterly_dir

In:  data/raw/daily/*.parquet and data/raw/quarterly/*.parquet
Out: interactive Streamlit page (no files written)
"""


# ---------------------------------------------------------------------------#
# Data inventory table
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300, show_spinner="Scanning data directory …")
def _build_inventory() -> pd.DataFrame:
    rows = []
    daily_dir = settings.raw_daily_dir
    if not daily_dir.exists():
        return pd.DataFrame()

    for fp in sorted(daily_dir.glob("*.parquet")):
        ticker = fp.stem
        try:
            df = pd.read_parquet(fp)
            rows.append({
                "Ticker":     ticker,
                "Rows":       len(df),
                "Start":      str(df.index.min().date()) if not df.empty else "—",
                "End":        str(df.index.max().date()) if not df.empty else "—",
                "Columns":    ", ".join(df.columns.tolist()),
                "Has NaNs":   int(df.isnull().sum().sum()),
                "File (KB)":  round(fp.stat().st_size / 1024, 1),
            })
        except Exception as e:
            rows.append({"Ticker": ticker, "Rows": 0, "Start": "ERR", "End": str(e)[:40]})

    return pd.DataFrame(rows)


inventory = _build_inventory()

if inventory.empty:
    st.warning(
        "No price data found in `data/raw/daily/`.\n\n"
        "Run `python cli/scrape.py --backfill 2015` to populate data."
    )
    st.stop()

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tickers Available", len(inventory))
col2.metric("Total Rows", f"{inventory['Rows'].sum():,}")
earliest = inventory["Start"].replace("—", pd.NA).dropna().min()
latest   = inventory["End"].replace("—", pd.NA).dropna().max()
col3.metric("Earliest Date", earliest if earliest else "—")
col4.metric("Latest Date",   latest   if latest   else "—")

st.divider()

# Filter
search = st.text_input("Filter by ticker", placeholder="e.g. AAPL")
disp = inventory[inventory["Ticker"].str.contains(search.upper(), na=False)] if search else inventory

st.dataframe(style_generic(disp), use_container_width=True, height=400)
st.caption(f"Showing {len(disp)} of {len(inventory)} tickers")

st.divider()

# ---------------------------------------------------------------------------#
# Ticker drill-down
# ---------------------------------------------------------------------------#

st.subheader("Ticker Drill-Down")

all_tickers = inventory["Ticker"].tolist()
selected = st.selectbox("Select ticker", all_tickers, index=0)

@st.cache_data(ttl=300)
def _load_daily(ticker: str) -> pd.DataFrame:
    fp = settings.raw_daily_dir / f"{ticker}.parquet"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_parquet(fp)

df_daily = _load_daily(selected)

if df_daily.empty:
    st.warning(f"No data for {selected}")
else:
    # Date range slicer
    min_date = df_daily.index.min().date()
    max_date = df_daily.index.max().date()
    date_range = st.slider(
        "Date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )
    mask = (df_daily.index.date >= date_range[0]) & (df_daily.index.date <= date_range[1])
    df_slice = df_daily.loc[mask]

    tab1, tab2 = st.tabs(["Candlestick", "Raw Data"])
    with tab1:
        st.plotly_chart(candlestick(df_slice, selected), use_container_width=True)
    with tab2:
        st.dataframe(style_generic(df_slice.reset_index().tail(200)), use_container_width=True)

    # Quarterly data check
    q_path = settings.raw_quarterly_dir / f"{selected}.parquet"
    if q_path.exists():
        df_q = pd.read_parquet(q_path)
        with st.expander("Quarterly aggregated data"):
            st.dataframe(style_generic(df_q.reset_index().tail(50)), use_container_width=True)
