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

st.set_page_config(page_title="ML Predictions", page_icon="🤖", layout="wide")
st.title("🤖 ML Predictions")
st.caption("Classical + deep-learning models, walk-forward CV, SHAP feature importance.")
st.divider()

settings = get_settings()

# ---------------------------------------------------------------------------#
# Ticker selector
# ---------------------------------------------------------------------------#

ml_results_dir = settings.data_dir / "ml" / "results"

ml_tickers = sorted([fp.stem.replace("_ml_results", "")
                      for fp in ml_results_dir.glob("*_ml_results.parquet")]) \
    if ml_results_dir.exists() else []

daily_tickers = sorted([fp.stem for fp in settings.raw_daily_dir.glob("*.parquet")]) \
    if settings.raw_daily_dir.exists() else []

all_tickers = list(dict.fromkeys(ml_tickers + daily_tickers))  # ml results first

if not all_tickers:
    st.warning("No data found. Run `python cli/scrape.py` then `python cli/ml.py --train` first.")
    st.stop()

col1, col2 = st.columns([2, 1])
with col1:
    selected = st.selectbox("Ticker", all_tickers,
                             index=all_tickers.index("AAPL") if "AAPL" in all_tickers else 0)
with col2:
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

run_col, info_col = st.columns([1, 3])
with run_col:
    run_train = st.button("🏋️  Train All Models", type="primary",
                          help="Runs full ML pipeline — may take several minutes on GPU.")
with info_col:
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
# Show results
# ---------------------------------------------------------------------------#

if ml_df.empty:
    st.info(
        f"No ML results for **{selected}**.\n\n"
        "Click **Train All Models**, or run `python cli/ml.py --ticker {selected} --train`."
    )
    st.stop()

# Use comparison_table columns if available, else use raw parquet
display_cols = ["Model", "RMSE", "MAE", "MAPE", "Test Period", "Status"]
available_display = [c for c in display_cols if c in ml_df.columns]

# Summary metrics
if "RMSE" in ml_df.columns:
    numeric_rmse = pd.to_numeric(ml_df["RMSE"], errors="coerce")
    best_row = ml_df.loc[numeric_rmse.idxmin()] if not numeric_rmse.isna().all() else None
    if best_row is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Model",       str(best_row.get("Model", "—")))
        col2.metric("Best RMSE",        f"{float(best_row['RMSE']):.5f}")
        mape = best_row.get("MAPE", "—")
        col3.metric("MAPE",             str(mape))

st.divider()

# Comparison bar chart
st.subheader("Model RMSE Comparison")
st.plotly_chart(ml_comparison_bar(ml_df[available_display] if available_display else ml_df),
                use_container_width=True)

# Comparison table
st.subheader("Model Results Table")
st.dataframe(style_ml_table(ml_df[available_display] if available_display else ml_df),
             use_container_width=True)

# ---------------------------------------------------------------------------#
# Feature importance
# ---------------------------------------------------------------------------#

st.divider()
st.subheader("Feature Importance (Best Model)")

# Try to parse feature_importance JSON column from parquet
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

# ---------------------------------------------------------------------------#
# Latest prediction
# ---------------------------------------------------------------------------#

st.divider()
st.subheader("Latest Prediction (XGBoost)")

if st.button("Get Latest Prediction"):
    with st.spinner("Running predict_latest() …"):
        try:
            from src.ml.runner import predict_latest
            pred = predict_latest(selected, model_name="XGBoost",
                                   settings=settings, target_days=target_days)
            if pred is not None:
                direction = "📈 Bullish" if pred > 0 else "📉 Bearish"
                st.metric(
                    f"Predicted {target_days}-day return",
                    f"{pred*100:+.2f}%",
                    delta=direction,
                )
            else:
                st.warning("predict_latest() returned None — train models first.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
