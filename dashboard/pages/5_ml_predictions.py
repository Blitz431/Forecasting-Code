"""Page 4 — ML Predictions.

XGBoost / LightGBM / Random Forest / Ridge / LSTM / GRU / Transformer
model results, feature importance, walk-forward validation summary.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import ml_comparison_bar, feature_importance, _empty_fig
from dashboard.components.tables import style_ml_table, style_generic
from dashboard.components.ticker_selector import render_ticker_sidebar

st.set_page_config(page_title="ML Predictions", page_icon="🤖", layout="wide")
st.title("🤖 ML Predictions")
st.caption("Classical + deep-learning models, walk-forward CV, SHAP feature importance.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Ticker selector (global — persists across pages)
# ---------------------------------------------------------------------------#

ml_results_dir = settings.data_dir / "ml" / "results"

selected = render_ticker_sidebar()
if not selected:
    st.warning("No data found. Run `python cli/scrape.py` then `python cli/ml.py --train` first.")
    st.stop()

from dashboard.components.session_cache import format_freshness
st.caption(f"📅 ML results last saved: {format_freshness(settings.data_dir / 'ml' / 'results' / f'{selected}_ml_results.parquet')}")

target_days = st.selectbox("Prediction horizon", [1, 5, 20], format_func=lambda d: f"{d}-day return")

# ---------------------------------------------------------------------------#
# Load stored ML results
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300)
def _load_ml_results(ticker: str) -> pd.DataFrame:
    fp = ml_results_dir / f"{ticker}_ml_results.parquet"
    if not fp.exists():
        return pd.DataFrame()
    return pd.read_parquet(fp)

ml_df = _load_ml_results(selected)

# ---------------------------------------------------------------------------#
# Train / predict controls
# ---------------------------------------------------------------------------#

st.divider()

# ---------------------------------------------------------------------------#
# Controls row — train button + prediction button side by side
# ---------------------------------------------------------------------------#

ctrl_col, train_info_col = st.columns([1, 3])
with ctrl_col:
    run_train = st.button("🏋️  Train All Models", type="primary",
                          help="Runs full ML pipeline — may take several minutes on GPU.")
with train_info_col:
    if not ml_df.empty:
        st.success(f"Stored ML results found ({len(ml_df)} rows). Click Train to refresh.")

if run_train:
    with st.spinner(f"Training ML models for {selected} (target_days={target_days}) …"):
        try:
            from src.ml.runner import run_ticker as ml_run_ticker, comparison_table
            results = ml_run_ticker(selected, target_days=target_days,
                                    tune=False, feature_select=True,
                                    deep_learning=True, save=True)
            ml_df = comparison_table(results)
            st.success("Training complete!")
        except Exception as e:
            st.error(f"Training failed: {e}")

# ---------------------------------------------------------------------------#
# Latest prediction — shown right after the train button
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300)
def _load_latest_price(ticker: str) -> tuple[float | None, str | None]:
    """Return (latest_close_price, date_string) from daily data."""
    fp = settings.raw_daily_dir / f"{ticker}.parquet"
    if not fp.exists():
        return None, None
    df = pd.read_parquet(fp)
    if df.empty or "Close" not in df.columns:
        return None, None
    last_row = df["Close"].dropna().iloc[-1]
    last_date = str(df["Close"].dropna().index[-1].date())
    return float(last_row), last_date

st.subheader("Latest Prediction (XGBoost)")

# Cache key — invalidate when ticker or horizon changes, or user forces refresh
_pred_key = f"ml_pred_{selected}_{target_days}"
refresh = st.button("🔄 Refresh Prediction")
if refresh or _pred_key not in st.session_state:
    with st.spinner("Running prediction …"):
        try:
            from src.ml.runner import predict_latest
            st.session_state[_pred_key] = predict_latest(
                selected, model_name="XGBoost",
                settings=settings, target_days=target_days,
            )
        except Exception as e:
            st.session_state[_pred_key] = None
            st.error(f"Prediction failed: {e}")

pred = st.session_state.get(_pred_key)
if pred is not None:
    current_price, price_date = _load_latest_price(selected)

    target_date = pd.bdate_range(
        start=pd.Timestamp.today(), periods=target_days + 1
    )[-1].strftime("%Y-%m-%d")

    st.markdown(f"**What the model is saying for {selected}:**")

    col_a, col_b, col_c = st.columns(3)

    if current_price is not None:
        predicted_price = current_price * (1 + pred)
        price_delta = predicted_price - current_price
        col_a.metric(
            f"Current price ({price_date})",
            f"${current_price:,.2f}",
        )
        col_b.metric(
            f"Predicted price by {target_date}",
            f"${predicted_price:,.2f}",
            delta=f"${price_delta:+,.2f} ({pred*100:+.2f}%)",
            delta_color="normal",
        )
    else:
        col_b.metric(
            f"Predicted {target_days}-day return",
            f"{pred*100:+.2f}%",
        )

    direction = "📈 Bullish" if pred > 0 else "📉 Bearish"
    col_c.metric("Signal", direction)

    st.caption(
        f"Interpretation: XGBoost predicts {selected} will move "
        f"**{pred*100:+.2f}%** over the next **{target_days} trading day(s)** "
        f"(by {target_date})."
        + (f" That puts the price at **${current_price*(1+pred):,.2f}**"
           f" from today's close of **${current_price:,.2f}**."
           if current_price else "")
    )
elif _pred_key in st.session_state:
    st.warning("Prediction returned None — train models first.")

# ---------------------------------------------------------------------------#
# Model detail — collapsed by default so prediction stays in focus
# ---------------------------------------------------------------------------#

if ml_df.empty:
    st.info(
        f"No ML results for **{selected}**.\n\n"
        "Click **Train All Models**, or run `python cli/ml.py --ticker {selected} --train`."
    )
    st.stop()

with st.expander("📊 Model comparison & accuracy detail", expanded=True):
    # Use comparison_table columns if available, else use raw parquet
    display_cols = ["Model", "RMSE", "MAE", "MAPE", "Test Period", "Status"]
    available_display = [c for c in display_cols if c in ml_df.columns]

    # Summary metrics
    if "RMSE" in ml_df.columns:
        numeric_rmse = pd.to_numeric(ml_df["RMSE"], errors="coerce")
        best_row = ml_df.loc[numeric_rmse.idxmin()] if not numeric_rmse.isna().all() else None
        if best_row is not None:
            col1, col2, col3 = st.columns(3)
            col1.metric("Best Model", str(best_row.get("Model", "—")))
            col2.metric("Best RMSE",  f"{float(best_row['RMSE']):.5f}")
            col3.metric("MAPE",       str(best_row.get("MAPE", "—")))

    st.subheader("Model RMSE Comparison")
    st.plotly_chart(ml_comparison_bar(ml_df[available_display] if available_display else ml_df),
                    use_container_width=True)

    st.subheader("Model Results Table")
    st.dataframe(style_ml_table(ml_df[available_display] if available_display else ml_df),
                 use_container_width=True)

with st.expander("📖 How the ML models work", expanded=False):
    st.markdown("""
### Machine Learning Models

These models are trained on historical price, volume, technical indicators, macro data, and
calendar features to predict the stock's **forward return** over 1, 5, or 20 trading days.
All models are trained on data up to a cutoff date and evaluated on the unseen period after it.

| Model | How it works | Best at |
|---|---|---|
| **XGBoost** | Builds hundreds of decision trees sequentially, each one correcting the errors of the previous. Uses GPU acceleration. | Tabular data with mixed feature types |
| **LightGBM** | Similar to XGBoost but grows trees leaf-wise (faster, less memory). Handles large feature counts well. | Speed + large datasets |
| **CatBoost** | Gradient boosting with *ordered boosting* — designed for time-series to avoid target leakage between folds. | Sequential/time-series data |
| **Random Forest** | Builds many independent decision trees in parallel and averages their predictions. Less prone to overfitting than boosting. | Noisy data, baseline robustness |
| **Ridge** | Linear regression with L2 regularisation — penalises large coefficients to prevent overfitting. | When relationships are mostly linear |
| **Lasso** | Linear regression with L1 regularisation — automatically zeroes out weak features (built-in feature selection). | Sparse signal environments |
| **ElasticNet** | Combines Ridge and Lasso penalties. Balances feature selection with stability. | When you want both effects |
| **LSTM** | A recurrent neural network that maintains a memory of past time steps. Captures sequential patterns a tree model would miss. | Momentum and trend patterns |
| **GRU** | Simplified version of LSTM with fewer parameters — trains faster, similar accuracy on shorter sequences. | Faster training, similar to LSTM |
| **Transformer** | Uses self-attention to weigh the importance of every past time step against every other. State-of-the-art for sequences. | Long-range dependencies in price history |
| **Stacking Ensemble** | Trains all base models using time-series cross-validation, then fits a Ridge meta-model on their out-of-fold predictions. | Best overall accuracy — reduces any single model's bias |

### How the walk-forward evaluation works
The data is split at a fixed date boundary (train → test). Models never see future data during
training. Metrics (RMSE, MAE, MAPE) are computed only on the held-out test period, so they
reflect real predictive accuracy — not in-sample fit.

### What the metrics mean
- **RMSE** — Root Mean Squared Error. Lower is better. Penalises large errors more than small ones.
- **MAE** — Mean Absolute Error. Average absolute prediction error in return units.
- **MAPE** — Mean Absolute Percentage Error. Same idea but as a percentage of the actual value.
""")

with st.expander("🔍 Feature importance (best model)", expanded=True):
    feat_df = pd.DataFrame()
    if "feature_importance" in ml_df.columns:
        numeric_rmse = pd.to_numeric(ml_df.get("RMSE", pd.Series()), errors="coerce")
        if not numeric_rmse.isna().all():
            best_fi_row = ml_df.loc[numeric_rmse.idxmin()]
            fi_raw = best_fi_row.get("feature_importance")
            if fi_raw and fi_raw != "None":
                try:
                    fi_dict = json.loads(fi_raw) if isinstance(fi_raw, str) else fi_raw
                    feat_df = pd.DataFrame(
                        list(fi_dict.items()),
                        columns=["Feature", "Importance"]
                    ).sort_values("Importance", ascending=False)
                except Exception:
                    pass

    if feat_df.empty:
        st.info("Feature importance not available. Re-train models to generate SHAP values.")
    else:
        st.plotly_chart(feature_importance(feat_df, top_n=25), use_container_width=True)
        with st.expander("Full feature importance table"):
            st.dataframe(style_generic(feat_df), use_container_width=True)
