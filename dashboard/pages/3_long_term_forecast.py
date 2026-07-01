"""Page 2 — Long-Term Forecast.

Run all 12 forecast methods for any ticker and compare their accuracy.
Shows a forecast overlay chart, method comparison table, and AutoBest selection.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import forecast_overlay, price_line
from dashboard.components.tables import style_forecast_table, style_generic
from dashboard.components.ticker_selector import render_ticker_sidebar

st.set_page_config(page_title="Long-Term Forecast", page_icon="🔭", layout="wide")

from dashboard.components.market_clock import render_market_clock
render_market_clock()

st.title("🔭 Long-Term Forecast")
st.caption("All 12 forecast methods per ticker — seasonal decomposition, smoothing, OLS, and AutoBest.")
st.divider()

settings = get_settings()


# ---------------------------------------------------------------------------#
# Ticker selector (global — persists across pages)
# ---------------------------------------------------------------------------#

selected = render_ticker_sidebar()
if not selected:
    st.warning("No price data found. Run `python cli/scrape.py --backfill 2015` first.")
    st.stop()

from dashboard.components.session_cache import format_freshness
st.caption(f"📅 Forecasts last saved: {format_freshness(settings.forecasts_dir / f'{selected}_forecasts.parquet')}")

col_left, col_right = st.columns([2, 3])
with col_left:
    horizons = st.slider("Forecast quarters", 1, 8, value=settings.forecast_horizons)
    holdout  = st.slider("Holdout periods (evaluation)", 4, 16, value=settings.holdout_periods)

# ---------------------------------------------------------------------------#
# Load cached history
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300)
def _load_price(ticker: str) -> pd.DataFrame:
    fp = settings.raw_daily_dir / f"{ticker}.parquet"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_parquet(fp)

@st.cache_data(ttl=300)
def _load_quarterly(ticker: str) -> pd.Series | None:
    fp = settings.raw_quarterly_dir / f"{ticker}.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    if df.empty or "Close" not in df.columns:
        return None
    s = df["Close"].dropna()
    s.name = ticker
    return s

df_price   = _load_price(selected)
series_q   = _load_quarterly(selected)

# ---------------------------------------------------------------------------#
# Check for saved forecasts (fast path — no re-run needed)
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300)
def _load_saved_forecasts(ticker: str) -> pd.DataFrame:
    fp = settings.forecasts_dir / f"{ticker}_forecasts.parquet"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_parquet(fp)

saved_forecasts = _load_saved_forecasts(selected)

# ---------------------------------------------------------------------------#
# Run / show controls
# ---------------------------------------------------------------------------#

st.divider()

run_col, info_col = st.columns([1, 3])
with run_col:
    run_now = st.button("▶  Run All 12 Methods", type="primary")
with info_col:
    if not saved_forecasts.empty:
        st.success(f"Saved forecasts found ({len(saved_forecasts)} rows). "
                   "Showing stored results — click **Run** to refresh.")

# ---------------------------------------------------------------------------#
# Run forecasting pipeline
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=600, show_spinner="Running 12 forecast methods …")
def _run_forecasts(ticker: str, _horizons: int, _holdout: int):
    from src.forecasting.runner import run_all_methods, comparison_table
    from src.scraper.storage import get_ticker_filepath, load_dataframe

    fp = get_ticker_filepath(ticker, settings.raw_quarterly_dir)
    df = load_dataframe(fp)
    if df.empty or "Close" not in df.columns:
        return [], pd.DataFrame()

    series = df["Close"].dropna()
    series.name = ticker
    results = run_all_methods(series, horizons=_horizons, holdout=_holdout)
    for r in results:
        r.ticker = ticker
    table = comparison_table(results)
    return results, table

if run_now:
    if series_q is None:
        st.error(f"No quarterly data for {selected}. Run `cli/scrape.py` first.")
    else:
        results, table = _run_forecasts(selected, horizons, holdout)

        if not results:
            st.error("Forecast runner returned no results.")
        else:
            # Summary banner
            from src.forecasting.runner import best_result
            best = best_result(results)
            if best:
                st.success(f"✅ Best method: **{best.method_name}** — RMSE {best.rmse:.4f}")

            # Overlay chart
            st.plotly_chart(
                forecast_overlay(df_price, results, selected),
                use_container_width=True,
            )

            # Comparison table
            st.subheader("Method Comparison")
            st.dataframe(style_forecast_table(table), use_container_width=True)

            # Per-method forecasts
            with st.expander("Detailed forecast values"):
                for r in results:
                    if r.error is None and not r.forecasts.empty:
                        st.markdown(f"**#{r.method_number} {r.method_name}** (RMSE={r.rmse:.4f})")
                        st.dataframe(r.forecasts.rename("Forecast Price").to_frame(), use_container_width=True)

elif not saved_forecasts.empty:
    # Display stored results without re-running
    st.subheader("Stored Forecast Values")

    # Best method from saved file
    if "RMSE" in saved_forecasts.columns and "Method_Name" in saved_forecasts.columns:
        best_row = saved_forecasts.loc[saved_forecasts["RMSE"].idxmin()]
        st.info(f"Best stored method: **{best_row['Method_Name']}** — RMSE {best_row['RMSE']:.4f}")

    # Group by method and show the comparison
    if "Method_Name" in saved_forecasts.columns and "RMSE" in saved_forecasts.columns:
        summary = (saved_forecasts
                   .groupby("Method_Name")
                   .agg(RMSE=("RMSE", "first"), MAE=("MAE", "first"),
                        MAPE=("MAPE", "first"), Points=("Forecast_Price", "count"))
                   .reset_index()
                   .sort_values("RMSE"))
        st.dataframe(style_forecast_table(summary), use_container_width=True)

    st.plotly_chart(price_line(df_price, selected), use_container_width=True)

    with st.expander("Raw forecast table"):
        st.dataframe(style_generic(saved_forecasts), use_container_width=True)

else:
    st.info(
        f"No forecasts found for **{selected}**.\n\n"
        "Click **Run All 12 Methods** above, or run `python cli/forecast.py --ticker {selected}` first."
    )
    if not df_price.empty:
        st.plotly_chart(price_line(df_price, selected), use_container_width=True)

st.divider()
with st.expander("📖 How the long-term forecasting works", expanded=False):
    st.markdown("""
### Long-Term Forecasting Engine

This page forecasts a stock's **price quarters into the future** using 12 different statistical
and machine-learning methods. All methods are evaluated using a holdout period (the most recent
quarters are withheld during training so accuracy can be measured on unseen data).

The **AutoBest** model is whichever method achieves the lowest RMSE on the holdout period — it's
selected automatically and used as the primary forecast.

| Method | How it works |
|---|---|
| **ARIMA** | Models price as a function of its own past values and past forecast errors. Good for stationary series. |
| **SARIMA** | Extends ARIMA with seasonal terms — captures quarterly/annual cycles in earnings or revenue. |
| **ETS (Exponential Smoothing)** | Weights recent observations more heavily than older ones. Simple but often accurate. |
| **Holt-Winters** | ETS with explicit trend and seasonal components. Works well when seasonality is strong. |
| **Prophet** | Facebook's forecasting library. Decomposes price into trend + seasonality + holidays. Robust to missing data. |
| **Linear Regression** | Fits a straight trend line to historical prices and projects it forward. Best as a baseline. |
| **Polynomial Regression** | Fits a curve instead of a line — can capture acceleration or deceleration in growth. |
| **Random Walk** | Assumes tomorrow's price equals today's price plus random noise. The hardest baseline to beat. |
| **XGBoost (time-series)** | Gradient-boosted trees trained on lagged price features. Captures non-linear patterns. |
| **LightGBM (time-series)** | Same idea as XGBoost but faster. |
| **LSTM** | Recurrent neural network that remembers past quarters. Captures long-range trends. |
| **AutoBest** | Automatically selects whichever of the above methods had the lowest holdout RMSE. |

### Long-term vs short-term — what's the difference?

| | **Long-Term (this page)** | **Short-Term (Signals page)** |
|---|---|---|
| **Input data** | Quarterly earnings, revenue, fundamentals | Daily OHLCV price + volume |
| **Horizon** | Quarters to years ahead | Days to weeks ahead |
| **Methods** | Statistical time-series + ML on fundamentals | Technical indicators + ML on price |
| **Best for** | Valuation, entry/exit windows, target prices | Timing trades, momentum, reversals |

Use long-term forecasts to decide **whether** to buy a stock; use short-term signals to decide **when**.
""")

