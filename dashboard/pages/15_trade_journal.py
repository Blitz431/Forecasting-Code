"""Page 15 — Trade Journal.

Audit log of every trade with entry/exit reasons, signals, P&L.
Tax lot tracking. Full implementation in Phase 10.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.tables import style_generic

st.set_page_config(page_title="Trade Journal", page_icon="📓", layout="wide")
st.title("📓 Trade Journal")
st.caption("Audit log of every trade with entry/exit reasons, signal triggers, P&L, and tax lots.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Phase note
# ---------------------------------------------------------------------------#

st.info(
    "**Phase 10** will implement:\n\n"
    "- `src/trading/trade_journal.py` — every trade logged with entry/exit reasons, "
    "which signals triggered, P&L, holding period\n"
    "- `src/trading/tax_lots.py` — FIFO/LIFO/specific cost basis, "
    "long-term vs short-term gains, tax-loss harvesting suggestions\n\n"
    "This page will auto-populate from Alpaca order history and simulated backtest trades."
)

st.divider()

# ---------------------------------------------------------------------------#
# Show Alpaca order history (if API key available)
# ---------------------------------------------------------------------------#

st.subheader("Order History (Alpaca)")

@st.cache_data(ttl=120, show_spinner="Fetching Alpaca orders …")
def _fetch_orders(limit: int = 100):
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(
            settings.alpaca.api_key,
            settings.alpaca.secret_key,
            settings.alpaca.base_url,
        )
        orders = api.list_orders(status="all", limit=limit)
        rows = []
        for o in orders:
            rows.append({
                "Submitted":   str(o.submitted_at)[:19] if o.submitted_at else "—",
                "Symbol":      o.symbol,
                "Side":        o.side.upper(),
                "Type":        o.order_type,
                "Qty":         o.qty,
                "Filled Qty":  o.filled_qty,
                "Avg Fill $":  f"${float(o.filled_avg_price):.2f}" if o.filled_avg_price else "—",
                "Status":      o.status,
            })
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame(), str(e)

if settings.alpaca.api_key:
    limit = st.number_input("Orders to fetch", min_value=10, max_value=500, value=100)
    result = _fetch_orders(int(limit))

    if isinstance(result, tuple):
        order_df, err = result
        if order_df.empty:
            st.error(f"Could not fetch orders: {err}")
    else:
        order_df = result

    if not order_df.empty:
        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            side_filter = st.multiselect("Side", ["BUY", "SELL"], default=["BUY", "SELL"])
        with col2:
            status_filter = st.multiselect("Status", order_df["Status"].unique().tolist(),
                                            default=order_df["Status"].unique().tolist())

        filtered = order_df[
            order_df["Side"].isin(side_filter) &
            order_df["Status"].isin(status_filter)
        ]

        st.metric("Orders shown", len(filtered))
        st.dataframe(style_generic(filtered), use_container_width=True, height=400)
    else:
        st.info("No orders returned.")
else:
    st.warning(
        "Alpaca API keys not configured.\n\n"
        "Add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` to your `.env` file."
    )

st.divider()

# ---------------------------------------------------------------------------#
# Local trade log (CSV/Parquet if exists)
# ---------------------------------------------------------------------------#

st.subheader("Local Trade Log")

trade_log_path = settings.data_dir / "trading" / "trade_journal.parquet"

if trade_log_path.exists():
    trade_log = pd.read_parquet(trade_log_path)
    st.dataframe(style_generic(trade_log), use_container_width=True)
else:
    st.info(
        f"No local trade journal found at `{trade_log_path}`.\n\n"
        "It will be populated automatically when Phase 10 trading is activated."
    )

st.divider()

# ---------------------------------------------------------------------------#
# Tax lots preview
# ---------------------------------------------------------------------------#

st.subheader("Tax Lots (Phase 10)")

tax_lot_path = settings.data_dir / "trading" / "tax_lots.parquet"

if tax_lot_path.exists():
    tax_df = pd.read_parquet(tax_lot_path)
    st.dataframe(style_generic(tax_df), use_container_width=True)
else:
    st.markdown("""
    Phase 10 will track:
    - **Cost basis** per lot (FIFO / LIFO / specific ID)
    - **Long-term vs short-term** capital gains classification
    - **Tax-loss harvesting** suggestions
    - **Wash-sale** rule detection
    """)
