"""Global ticker selector — renders in the sidebar and persists across pages.

Usage in any page:
    from dashboard.components.ticker_selector import render_ticker_sidebar
    selected = render_ticker_sidebar()
    if not selected:
        st.stop()
"""

from __future__ import annotations

import streamlit as st


def _has_trained_model(ticker: str, settings) -> bool:
    """Return True if any saved model artifact exists for *ticker*."""
    model_dir = settings.data_dir / "ml" / "models" / ticker
    if not model_dir.exists():
        return False
    return any(model_dir.glob("*.pkl")) or any(model_dir.glob("*.pt"))


def render_ticker_sidebar() -> str:
    """Render the global ticker selectbox in the sidebar.

    Builds the ticker list from settings.raw_daily_dir parquet files.
    Wires selection to st.session_state["global_ticker"] so it persists
    across all pages. Returns the currently selected ticker string,
    or an empty string if no data is available.
    """
    from config.settings import get_settings
    from dashboard.components.session_cache import get_ticker_status, format_freshness

    settings = get_settings()

    daily_tickers = (
        sorted([fp.stem for fp in settings.raw_daily_dir.glob("*.parquet")])
        if settings.raw_daily_dir.exists()
        else []
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Stock Selector")

    if not daily_tickers:
        st.sidebar.warning("No ticker data found.\nRun `cli/scrape.py` first.")
        return ""

    # Manage state manually — no widget key, so Streamlit has no lifecycle to
    # interfere with. Plain session_state entries survive page navigation.
    current = st.session_state.get("global_ticker")
    if not isinstance(current, str) or current not in daily_tickers:
        current = "AAPL" if "AAPL" in daily_tickers else daily_tickers[0]

    selected = st.sidebar.selectbox(
        "Ticker",
        daily_tickers,
        index=daily_tickers.index(current),
    )
    st.session_state["global_ticker"] = selected  # write AFTER widget renders

    # --- Freshness block ---
    daily_fp = settings.raw_daily_dir / f"{selected}.parquet"
    if daily_fp.exists():
        try:
            import pandas as pd
            idx = pd.read_parquet(daily_fp, columns=[]).index
            data_through = str(idx.max().date()) if len(idx) > 0 else "?"
        except Exception:
            data_through = "?"
        st.sidebar.caption(f"Daily data through: {data_through}")

    status = get_ticker_status(selected)
    if status:
        run_time = status.get("run_at", "")[:16].replace("T", " ")
        flags = " ".join(
            f"✓ {k}" for k in ("forecast", "ml", "news", "options")
            if status.get(k)
        )
        st.sidebar.caption(f"Last analyzed: {run_time}  {flags}")

    # --- Analysis buttons ---
    st.sidebar.markdown("---")
    run_all = st.sidebar.button(
        "▶ Run All Analysis",
        type="primary",
        use_container_width=True,
        help=(
            "Runs forecast, signals, ML (trains if no saved model), "
            "news, and options for the selected ticker."
        ),
    )
    train_ml = st.sidebar.button(
        "🏋️ Train ML Models",
        use_container_width=True,
        help="Force-train all ML models for the selected ticker (overwrites existing).",
    )
    predict_only = st.sidebar.button(
        "🔮 Predict Only",
        use_container_width=True,
        help="Run ML prediction using saved model (warns if none found).",
    )

    if run_all:
        _step_label = st.sidebar.empty()
        _progress   = st.sidebar.progress(0)
        errors = _run_all_analysis(selected, settings, _step_label, _progress)
        _progress.empty()
        _step_label.empty()
        if errors:
            st.sidebar.warning(
                f"Completed with {len(errors)} issue(s):\n"
                + "\n".join(f"• {e}" for e in errors)
            )
        else:
            st.sidebar.success(f"✅ All analysis complete for {selected}!")
        st.cache_data.clear()

    elif train_ml:
        with st.sidebar:
            with st.spinner(f"Training ML models for {selected} …"):
                try:
                    from src.ml.runner import run_ticker
                    from dashboard.components.session_cache import record_run
                    run_ticker(selected, settings=settings, save=True)
                    record_run(selected, ml="trained")
                    st.sidebar.success(f"ML training complete for {selected}!")
                except Exception as exc:
                    st.sidebar.error(f"ML training failed: {exc}")
        st.cache_data.clear()

    elif predict_only:
        if not _has_trained_model(selected, settings):
            st.sidebar.warning(
                f"No saved model found for {selected}. Use '🏋️ Train ML Models' first."
            )
        else:
            with st.sidebar:
                with st.spinner(f"Running ML prediction for {selected} …"):
                    try:
                        from src.ml.runner import predict_latest
                        from dashboard.components.session_cache import record_run
                        result = predict_latest(
                            selected, model_name="XGBoost", settings=settings, target_days=5
                        )
                        if result is not None:
                            st.sidebar.success(f"Prediction: {result:+.3%}")
                        else:
                            st.sidebar.warning("Prediction returned None.")
                        record_run(selected, ml="predicted")
                    except Exception as exc:
                        st.sidebar.error(f"Prediction failed: {exc}")
            st.cache_data.clear()

    return selected


def _run_all_analysis(ticker: str, settings, _step_label=None, _progress=None) -> list[str]:
    """Run all five analysis pipelines for *ticker*. Returns list of error strings.

    *_step_label* and *_progress* are optional Streamlit sidebar placeholder / progress
    elements — when supplied they are updated as each step starts and finishes.
    """
    import pandas as pd
    from dashboard.components.session_cache import record_run

    STEPS = [
        ("1 / 5  📈  Long-term forecast",  "Forecast"),
        ("2 / 5  📡  Short-term signals",  "Signals"),
        ("3 / 5  🤖  ML models",           "ML"),
        ("4 / 5  📰  News sentiment",       "News"),
        ("5 / 5  🎯  Options flow",         "Options"),
    ]
    TOTAL = len(STEPS)

    def _tick(step_idx: int, done: bool = False) -> None:
        if _step_label is not None:
            label, _ = STEPS[step_idx]
            _step_label.markdown(f"**{label}**" if not done else f"~~{label}~~")
        if _progress is not None:
            pct = (step_idx + (1 if done else 0)) / TOTAL
            _progress.progress(pct)

    errors: list[str] = []
    ran_forecast = False
    ran_news = False
    ran_options = False
    ml_status: str | None = None

    # 1. Long-term forecast
    _tick(0)
    try:
        from src.forecasting.runner import run_all_methods
        from src.scraper.storage import get_ticker_filepath, load_dataframe

        fp = get_ticker_filepath(ticker, settings.raw_quarterly_dir)
        df = load_dataframe(fp)
        if not df.empty and "Close" in df.columns:
            series = df["Close"].dropna()
            series.name = ticker
            results = run_all_methods(
                series,
                horizons=settings.forecast_horizons,
                holdout=settings.holdout_periods,
            )
            rows = []
            for r in results:
                if r.error is None and not r.forecasts.empty:
                    for fdate, fprice in r.forecasts.items():
                        rows.append({
                            "Ticker": ticker,
                            "Method_Number": r.method_number,
                            "Method_Name": r.method_name,
                            "Forecast_Date": fdate,
                            "Forecast_Price": float(fprice),
                            "RMSE": r.rmse,
                            "MAE": r.mae,
                            "MAPE": r.mape,
                        })
            if rows:
                settings.forecasts_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_parquet(
                    settings.forecasts_dir / f"{ticker}_forecasts.parquet",
                    index=False,
                )
        ran_forecast = True
    except Exception as exc:
        errors.append(f"Forecast: {exc}")
    _tick(0, done=True)

    # 2. Short-term signals
    _tick(1)
    try:
        from src.indicators.signal_aggregator import run_and_aggregate

        price_fp = settings.raw_daily_dir / f"{ticker}.parquet"
        if price_fp.exists():
            df_daily = pd.read_parquet(price_fp)
            run_and_aggregate(df_daily, ticker)
    except Exception as exc:
        errors.append(f"Signals: {exc}")
    _tick(1, done=True)

    # 3. ML — train if no saved model exists, otherwise predict only
    _tick(2)
    try:
        from src.ml.runner import run_ticker, predict_latest

        if _has_trained_model(ticker, settings):
            predict_latest(ticker, model_name="XGBoost", settings=settings, target_days=5)
            ml_status = "predicted"
        else:
            run_ticker(ticker, settings=settings, save=True)
            ml_status = "trained"
    except Exception as exc:
        errors.append(f"ML: {exc}")
    _tick(2, done=True)

    # 4. News sentiment
    _tick(3)
    try:
        from src.news.runner import run_news_pipeline

        run_news_pipeline(
            [ticker],
            max_articles=settings.news_max_articles,
            sentiment_window_days=settings.news_sentiment_window_days,
        )
        ran_news = True
    except Exception as exc:
        errors.append(f"News: {exc}")
    _tick(3, done=True)

    # 5. Options flow
    _tick(4)
    try:
        from src.options.options_data import compute_options_metrics, save_options_metrics
        from src.options.implied_vol import compute_iv_metrics, save_iv_metrics

        save_options_metrics(compute_options_metrics(ticker), settings.options_dir)
        save_iv_metrics(compute_iv_metrics(ticker, settings.raw_daily_dir), settings.options_dir)
        ran_options = True
    except Exception as exc:
        errors.append(f"Options: {exc}")
    _tick(4, done=True)

    # Record in daily session cache
    try:
        data_through = None
        daily_fp = settings.raw_daily_dir / f"{ticker}.parquet"
        if daily_fp.exists():
            idx = pd.read_parquet(daily_fp, columns=[]).index
            if len(idx) > 0:
                data_through = str(idx.max().date())
        record_run(
            ticker,
            forecast=ran_forecast,
            ml=ml_status,
            news=ran_news,
            options=ran_options,
            data_through=data_through,
        )
    except Exception:
        pass  # session cache failure is non-fatal

    return errors
