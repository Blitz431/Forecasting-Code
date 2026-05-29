"""Page 2a — Stock Summary.

Single-page dashboard showing key signals from all 7 analysis pipelines for one
ticker. Acts as a launchpad — each section header links to the full detail page.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.ticker_selector import render_ticker_sidebar

st.set_page_config(page_title="Stock Summary", page_icon="📋", layout="wide")

settings = get_settings()

# ---------------------------------------------------------------------------#
# Ticker selector (global — persists across pages)
# ---------------------------------------------------------------------------#

selected = render_ticker_sidebar()
if not selected:
    st.warning("No price data found. Run `python cli/scrape.py --backfill 2015` first.")
    st.stop()

st.title(f"📋 Stock Summary — {selected}")
st.caption("All signals for one ticker at a glance.")

from dashboard.components.session_cache import format_freshness
_daily_fp = settings.raw_daily_dir / f"{selected}.parquet"
st.caption(f"📅 Data as of: {format_freshness(_daily_fp)}")

st.divider()

# ---------------------------------------------------------------------------#
# SECTOR_PEERS (copied inline — avoids circular import from page 16)
# ---------------------------------------------------------------------------#

SECTOR_PEERS: dict[str, list[str]] = {
    "Technology":             ["AAPL", "MSFT", "NVDA", "AVGO", "AMD", "INTC", "QCOM", "TXN", "CRM", "ORCL"],
    "Healthcare":             ["JNJ", "UNH", "LLY", "ABBV", "ABT", "MRK", "TMO", "DHR", "AMGN", "ISRG"],
    "Financials":             ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "SCHW", "USB"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TGT", "BKNG", "GM"],
    "Communication":          ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "SNAP", "TTWO"],
    "Industrials":            ["UNP", "HON", "RTX", "CAT", "BA", "DE", "MMM", "GE", "FDX", "UPS"],
    "Energy":                 ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "PXD"],
    "Consumer Staples":       ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "GIS"],
    "Utilities":              ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PEG", "XEL", "ED"],
    "Real Estate":            ["PLD", "AMT", "EQIX", "CCI", "PSA", "WELL", "SPG", "O", "DLR", "EQR"],
    "Materials":              ["LIN", "APD", "ECL", "SHW", "FCX", "NEM", "NUE", "VMC", "MLM", "ALB"],
}

# ---------------------------------------------------------------------------#
# Section 1 — Price Strip
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300)
def _load_price_strip(ticker: str) -> dict:
    fp = settings.raw_daily_dir / f"{ticker}.parquet"
    if not fp.exists():
        return {}
    df = pd.read_parquet(fp)
    if df.empty or "Close" not in df.columns:
        return {}
    closes = df["Close"].dropna()
    if len(closes) < 2:
        return {}
    latest = float(closes.iloc[-1])
    prev   = float(closes.iloc[-2])
    day_change     = latest - prev
    day_change_pct = day_change / prev * 100
    tail252        = closes.tail(252)
    high_52w       = float(tail252.max())
    low_52w        = float(tail252.min())
    range_span     = high_52w - low_52w
    range_pct      = (latest - low_52w) / range_span * 100 if range_span else 0.0
    avg_vol = None
    if "Volume" in df.columns:
        avg_vol = float(df["Volume"].dropna().tail(20).mean())
    return {
        "latest":         latest,
        "day_change":     day_change,
        "day_change_pct": day_change_pct,
        "high_52w":       high_52w,
        "low_52w":        low_52w,
        "range_pct":      range_pct,
        "avg_vol":        avg_vol,
    }


with st.container(border=True):
    price = _load_price_strip(selected)
    if not price:
        st.info("No price data — run ▶ Run All Analysis first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Price",
            f"${price['latest']:,.2f}",
            help="Latest closing price from daily OHLCV data.",
        )
        arrow = "↑" if price["day_change"] >= 0 else "↓"
        c2.metric(
            "Day Change",
            f"{price['day_change']:+,.2f} {arrow}",
            delta=f"{price['day_change_pct']:+.2f}%",
            delta_color="normal",
            help="Dollar and percent change from the prior trading day's close.",
        )
        c3.metric(
            "52-Week Range",
            f"${price['low_52w']:,.0f} — ${price['high_52w']:,.0f}",
            delta=f"Sitting at {price['range_pct']:.0f}% of range",
            delta_color="off",
            help=(
                "Lowest and highest closing price over the past 252 trading days. "
                "The % shows where today's price sits in that range — 100% = at the yearly high."
            ),
        )
        if price.get("avg_vol") is not None:
            vol = price["avg_vol"]
            vol_str = f"{vol / 1e6:.1f}M" if vol >= 1e6 else f"{vol / 1e3:.0f}K"
            c4.metric(
                "Avg Vol 20d",
                vol_str,
                help=(
                    "Average daily trading volume over the last 20 sessions. "
                    "High volume on up-days confirms bullish moves."
                ),
            )

st.divider()

# ---------------------------------------------------------------------------#
# Section 2 — Quick Trade
# ---------------------------------------------------------------------------#

_TRADE_KEY  = f"trade_panel_{selected}"
_RESULT_KEY = f"trade_result_{selected}"

with st.container(border=True):
    # Alpaca client — cached in session state (cheap to re-create if missing)
    if "alpaca_client" not in st.session_state:
        try:
            from src.trading.alpaca_client import AlpacaClient
            _c = AlpacaClient(settings)
            st.session_state["alpaca_client"] = _c if _c.connected else None
        except Exception:
            st.session_state["alpaca_client"] = None

    alpaca    = st.session_state.get("alpaca_client")
    alpaca_ok = alpaca is not None and bool(settings.alpaca.api_key)

    # Show any result banner from the previous trade (popped so it shows once)
    _pending = st.session_state.pop(_RESULT_KEY, None)
    if _pending:
        if _pending["success"]:
            st.success(_pending["msg"])
        else:
            st.error(_pending["msg"])

    # Header
    if alpaca_ok:
        _badge = "🔴 LIVE" if alpaca.is_live else "🟡 PAPER"
        st.markdown(f"**Quick Trade** &nbsp;&nbsp; {_badge}")
    else:
        st.markdown("**Quick Trade**")

    if _TRADE_KEY not in st.session_state:
        st.session_state[_TRADE_KEY] = None

    cur_action = st.session_state[_TRADE_KEY]

    # Buttons row
    _bb = st.columns(5)
    if _bb[0].button("🟢 BUY",  use_container_width=True, disabled=not alpaca_ok):
        st.session_state[_TRADE_KEY] = None if cur_action == "buy"  else "buy"
    if _bb[1].button("🔴 SELL", use_container_width=True, disabled=not alpaca_ok):
        st.session_state[_TRADE_KEY] = None if cur_action == "sell" else "sell"
    if _bb[2].button("📈 CALL", use_container_width=True, disabled=not alpaca_ok):
        st.session_state[_TRADE_KEY] = None if cur_action == "call" else "call"
    if _bb[3].button("📉 PUT",  use_container_width=True, disabled=not alpaca_ok):
        st.session_state[_TRADE_KEY] = None if cur_action == "put"  else "put"
    if _bb[4].button("✖ Close", use_container_width=True, disabled=cur_action is None):
        st.session_state[_TRADE_KEY] = None

    if not alpaca_ok:
        st.caption("Add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` to `.env` to enable trading.")

    action = st.session_state[_TRADE_KEY]

    # ── Stock order form ──────────────────────────────────────────────────
    if action in ("buy", "sell") and alpaca_ok:
        st.divider()
        lp_default = float(price["latest"]) if price else 100.0
        with st.form(f"stock_form_{selected}_{action}", clear_on_submit=True):
            fc1, fc2, fc3 = st.columns([2, 2, 2])
            qty        = fc1.number_input("Shares", min_value=0.01, value=1.0, step=1.0, format="%.2f")
            order_type = fc2.selectbox("Order type", ["market", "limit"])
            limit_px   = fc3.number_input(
                "Limit $", min_value=0.01, value=lp_default, step=0.01,
                disabled=(order_type != "limit"),
            )
            verb = "Buy" if action == "buy" else "Sell"
            if st.form_submit_button(f"Confirm {verb} {qty:.2f} × {selected}", type="primary"):
                lp = float(limit_px) if order_type == "limit" else None
                result = alpaca.place_order(selected, qty=float(qty), side=action,
                                            order_type=order_type, limit_price=lp)
                if result:
                    st.session_state[_RESULT_KEY] = {
                        "success": True,
                        "msg": f"✅ {verb} order submitted — `{result.order_id[:8]}…`  status: **{result.status.upper()}**",
                    }
                    st.session_state[_TRADE_KEY] = None
                    st.rerun()
                else:
                    st.error("❌ Order failed — check the Trading page for details.")

    # ── Option order form ─────────────────────────────────────────────────
    elif action in ("call", "put") and alpaca_ok:
        st.divider()
        current_price_val = float(price["latest"]) if price else None
        contracts_key = f"opt_{selected}_{action}"

        if contracts_key not in st.session_state:
            with st.spinner(f"Fetching {action.upper()} contracts for {selected}…"):
                st.session_state[contracts_key] = alpaca.get_option_contracts(
                    selected, action, current_price=current_price_val
                )

        contracts = st.session_state.get(contracts_key, [])

        col_refresh, _ = st.columns([1, 4])
        if col_refresh.button("🔄 Refresh contracts", key=f"refresh_{selected}_{action}"):
            st.session_state.pop(contracts_key, None)
            st.rerun()

        if not contracts:
            st.info(f"No active {action.upper()} contracts found for {selected} "
                    "in the next 7–60 days. Options may not be available for this ticker.")
        else:
            labels = [c.display_label() for c in contracts]
            with st.form(f"opt_form_{selected}_{action}", clear_on_submit=True):
                idx = st.selectbox(
                    f"{action.upper()} Contract  ({len(contracts)} available)",
                    range(len(labels)),
                    format_func=lambda i: labels[i],
                )
                n_contracts = st.number_input("Contracts (×100 shares each)", min_value=1, value=1, step=1)
                chosen = contracts[idx]
                st.caption(f"OSI symbol: `{chosen.symbol}`")
                if st.form_submit_button(
                    f"Confirm Buy {n_contracts} {action.upper()} Contract(s)", type="primary"
                ):
                    result = alpaca.place_order(
                        chosen.symbol, qty=float(n_contracts), side="buy",
                        order_type="market", time_in_force="day",
                    )
                    if result:
                        st.session_state[_RESULT_KEY] = {
                            "success": True,
                            "msg": (
                                f"✅ {action.upper()} order submitted — `{result.order_id[:8]}…`  "
                                f"status: **{result.status.upper()}**  ·  `{chosen.symbol}`"
                            ),
                        }
                        st.session_state[_TRADE_KEY] = None
                        st.session_state.pop(contracts_key, None)
                        st.rerun()
                    else:
                        st.error("❌ Option order failed — check the Trading page for details.")

st.divider()

# ---------------------------------------------------------------------------#
# Section 3 — Long-Term Forecast + Short-Term Signals (two columns)
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300)
def _load_forecast_summary(ticker: str) -> dict:
    fp = settings.forecasts_dir / f"{ticker}_forecasts.parquet"
    if not fp.exists():
        return {}
    df = pd.read_parquet(fp)
    if df.empty or "RMSE" not in df.columns:
        return {}
    best     = df.loc[df["RMSE"].idxmin()]
    method   = str(best.get("Method_Name", "—"))
    rmse     = float(best["RMSE"])
    method_rows = df[df["Method_Name"] == method].sort_values("Forecast_Date")
    fdate, fprice = None, None
    if not method_rows.empty:
        first  = method_rows.iloc[0]
        fdate  = str(first["Forecast_Date"])[:10]
        fprice = float(first["Forecast_Price"])
    # current price (for % delta and bullish-method count)
    current = None
    pfp = settings.raw_daily_dir / f"{ticker}.parquet"
    if pfp.exists():
        pdf = pd.read_parquet(pfp)
        if not pdf.empty and "Close" in pdf.columns:
            current = float(pdf["Close"].dropna().iloc[-1])
    bullish_methods = 0
    total_methods   = 0
    if current is not None and "Method_Name" in df.columns:
        for _, grp in df.groupby("Method_Name"):
            total_methods += 1
            grp_s = grp.sort_values("Forecast_Date")
            if not grp_s.empty and float(grp_s.iloc[0]["Forecast_Price"]) > current:
                bullish_methods += 1
    return {
        "method":          method,
        "rmse":            rmse,
        "fdate":           fdate,
        "fprice":          fprice,
        "current":         current,
        "bullish_methods": bullish_methods,
        "total_methods":   total_methods,
    }


@st.cache_data(ttl=300)
def _load_signals_summary(ticker: str) -> dict:
    fp = settings.raw_daily_dir / f"{ticker}.parquet"
    if not fp.exists():
        return {}
    df = pd.read_parquet(fp)
    if df.empty:
        return {}
    try:
        from src.indicators.signal_aggregator import run_and_aggregate
        agg     = run_and_aggregate(df, ticker)
        bullish = sum(1 for r in agg.results if r.error is None and int(r.signal) > 0)
        total   = sum(1 for r in agg.results if r.error is None)
        return {
            "label":   agg.signal.label(),
            "score":   float(agg.score),
            "bullish": bullish,
            "total":   total,
        }
    except Exception:
        return {}


forecast_data = _load_forecast_summary(selected)
signals_data  = _load_signals_summary(selected)

col_fc, col_sig = st.columns(2)

with col_fc:
    with st.container(border=True):
        st.page_link("pages/3_long_term_forecast.py", label="🔭 Long-Term Forecast →")
        if not forecast_data:
            st.info("No data — run ▶ Run All Analysis first.")
        else:
            m1, m2 = st.columns(2)
            m1.metric(
                "Best Model",
                forecast_data.get("method", "—"),
                delta=f"RMSE {forecast_data.get('rmse', 0):.4f}",
                delta_color="off",
                help=(
                    "The forecast method with the lowest RMSE on the holdout period. "
                    "Lower RMSE = better historical accuracy."
                ),
            )
            fprice  = forecast_data.get("fprice")
            current = forecast_data.get("current")
            fdate   = forecast_data.get("fdate")
            if fprice is not None and current:
                pct = (fprice / current - 1) * 100
                m2.metric(
                    "Next Qtr Target",
                    f"${fprice:,.2f}",
                    delta=f"{pct:+.1f}%  by {fdate}",
                    delta_color="normal",
                    help=(
                        "The forecast method with the lowest RMSE on the holdout period. "
                        "Lower RMSE = better historical accuracy."
                    ),
                )
            bm = forecast_data.get("bullish_methods", 0)
            tm = forecast_data.get("total_methods", 0)
            st.metric(
                "Bullish Methods",
                f"{bm} out of {tm}",
                help="Number of distinct forecast methods projecting a price higher than today's close.",
            )
            if fprice is not None and current:
                pct  = (fprice / current - 1) * 100
                meth = forecast_data.get("method", "—")
                st.caption(f"{meth} projects {pct:+.1f}% by {fdate}.")

with col_sig:
    with st.container(border=True):
        st.page_link("pages/4_short_term_signals.py", label="📡 Short-Term Signals →")
        if not signals_data:
            st.info("No data — run ▶ Run All Analysis first.")
        else:
            m1, m2 = st.columns(2)
            m1.metric(
                "Composite Signal",
                signals_data.get("label", "—"),
                help=(
                    "Weighted score across 9 technical indicators (RSI, MACD, Bollinger, "
                    "Stochastic, EMA/SMA, Volume, Momentum, Fundamentals, Correlation). "
                    "Range: -2 (all bearish) to +2 (all bullish)."
                ),
            )
            m2.metric(
                "Score [-2, +2]",
                f"{signals_data.get('score', 0):+.2f}",
                help=(
                    "Weighted score across 9 technical indicators (RSI, MACD, Bollinger, "
                    "Stochastic, EMA/SMA, Volume, Momentum, Fundamentals, Correlation). "
                    "Range: -2 (all bearish) to +2 (all bullish)."
                ),
            )
            b = signals_data.get("bullish", 0)
            t = signals_data.get("total", 0)
            st.metric(
                "Bullish Indicators",
                f"{b} out of {t}",
                help=(
                    "Weighted score across 9 technical indicators (RSI, MACD, Bollinger, "
                    "Stochastic, EMA/SMA, Volume, Momentum, Fundamentals, Correlation). "
                    "Range: -2 (all bearish) to +2 (all bullish)."
                ),
            )
            st.caption(f"{b} of {t} indicators bullish. Composite: {signals_data.get('label', '—')}.")

st.divider()

# ---------------------------------------------------------------------------#
# Section 4 — ML Predictions (full-width, all 3 horizons)
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300)
def _load_latest_price(ticker: str) -> tuple[float | None, str | None]:
    fp = settings.raw_daily_dir / f"{ticker}.parquet"
    if not fp.exists():
        return None, None
    df = pd.read_parquet(fp)
    if df.empty or "Close" not in df.columns:
        return None, None
    closes = df["Close"].dropna()
    return float(closes.iloc[-1]), str(closes.index[-1].date())


with st.container(border=True):
    st.page_link("pages/5_ml_predictions.py", label="🤖 ML Predictions →")

    current_price, price_date = _load_latest_price(selected)

    ml_preds: dict[int, float | None] = {}
    with st.spinner("Loading ML predictions …"):
        for days in (1, 5, 20):
            key = f"ml_pred_{selected}_{days}"
            if key not in st.session_state:
                try:
                    from src.ml.runner import predict_latest
                    st.session_state[key] = predict_latest(
                        selected, model_name="XGBoost", settings=settings, target_days=days
                    )
                except Exception:
                    st.session_state[key] = None
            ml_preds[days] = st.session_state.get(key)

    if all(v is None for v in ml_preds.values()):
        st.info("No ML predictions — train models first via ▶ Run All Analysis.")
    else:
        _HORIZON_HELP = {
            1:  (
                "XGBoost trained on 40+ features (price momentum, RSI, MACD, volume ratios, macro). "
                "Predicts % return from today's close to tomorrow's close."
            ),
            5:  (
                "Same XGBoost model predicting cumulative % return over the next 5 trading days "
                "(one calendar week)."
            ),
            20: (
                "Same model predicting cumulative % return over the next 20 trading days "
                "(approx. one calendar month)."
            ),
        }
        _HORIZON_LABELS = {1: "1-Day Forecast", 5: "5-Day Forecast", 20: "20-Day Forecast"}

        ml_cols   = st.columns(3)
        valid_sig: list[tuple[int, float]] = []

        for i, days in enumerate((1, 5, 20)):
            pred = ml_preds[days]
            with ml_cols[i]:
                if pred is None:
                    st.metric(_HORIZON_LABELS[days], "—", help=_HORIZON_HELP[days])
                    continue
                target_date = pd.bdate_range(
                    start=pd.Timestamp.today(), periods=days + 1
                )[-1].strftime("%Y-%m-%d")
                if current_price is not None:
                    tgt       = current_price * (1 + pred)
                    delta_str = f"${tgt - current_price:+,.2f} ({pred * 100:+.2f}%)"
                    st.metric(
                        _HORIZON_LABELS[days],
                        f"${tgt:,.2f}",
                        delta=delta_str,
                        delta_color="normal",
                        help=_HORIZON_HELP[days],
                    )
                    st.caption(f"Now: ${current_price:,.2f}  ·  By: {target_date}")
                else:
                    st.metric(
                        _HORIZON_LABELS[days],
                        f"{pred * 100:+.2f}%",
                        help=_HORIZON_HELP[days],
                    )
                    st.caption(f"By: {target_date}")
                valid_sig.append((days, pred))

        if valid_sig:
            all_bull  = all(p > 0 for _, p in valid_sig)
            all_bear  = all(p < 0 for _, p in valid_sig)
            strongest = max(valid_sig, key=lambda x: abs(x[1]))
            tone      = "Bullish" if all_bull else ("Bearish" if all_bear else "Mixed")
            st.caption(
                f"XGBoost {tone.lower()} signal across all 3 horizons. "
                f"Strongest signal on {strongest[0]}-day ({strongest[1] * 100:+.2f}%)."
            )

st.divider()

# ---------------------------------------------------------------------------#
# Section 5 — News Sentiment + Political & Insider (two columns)
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300)
def _load_news_summary(ticker: str) -> dict:
    try:
        from src.news.aggregator import get_sentiment_summary
        return get_sentiment_summary(ticker, settings.news_articles_dir, days=7)
    except Exception:
        return {}


@st.cache_data(ttl=300)
def _load_political_summary(ticker: str) -> dict:
    try:
        from src.political.congress_tracker import get_congress_signal
        from src.political.insider_tracker import get_insider_signal
        return {
            "congress": get_congress_signal(ticker, settings.political_congress_dir, window_days=90),
            "insider":  get_insider_signal(ticker, settings.political_insider_dir, window_days=90),
        }
    except Exception:
        return {}


news_data = _load_news_summary(selected)
pol_data  = _load_political_summary(selected)

col_news, col_pol = st.columns(2)

with col_news:
    with st.container(border=True):
        st.page_link("pages/7_news_sentiment.py", label="📰 News Sentiment →")
        if not news_data:
            st.info("No data — run ▶ Run All Analysis first.")
        else:
            m1, m2 = st.columns(2)
            sentiment_label = news_data.get("sentiment", "neutral").title()
            m1.metric(
                "Sentiment",
                sentiment_label,
                help="FinBERT NLP label for the 7-day article window: Bullish, Bearish, or Neutral.",
            )
            m2.metric(
                "Avg Sentiment Score",
                f"{news_data.get('avg_score', 0.0):+.3f}",
                help=(
                    "FinBERT NLP score averaged across all articles in the window. "
                    "Range: -1 (very negative) to +1 (very positive). Above +0.1 = bullish tone."
                ),
            )
            count = news_data.get("article_count", 0)
            st.metric(
                "Articles (7-day)",
                count,
                help="Number of articles analyzed in the 7-day sentiment window.",
            )
            st.caption(f"{count} articles. Tone {sentiment_label.lower()}.")

with col_pol:
    with st.container(border=True):
        st.page_link("pages/8_political_insider.py", label="🏛️ Political & Insider →")
        if not pol_data:
            st.info("No data — run ▶ Run All Analysis first.")
        else:
            c_sig = pol_data.get("congress", {})
            i_sig = pol_data.get("insider", {})
            m1, m2 = st.columns(2)
            c_val   = c_sig.get("congress_signal", 0.0)
            i_val   = i_sig.get("insider_signal", 0.0)
            c_label = "Bullish" if c_val > 0 else ("Bearish" if c_val < 0 else "Neutral")
            i_label = "Bullish" if i_val > 0 else ("Bearish" if i_val < 0 else "Neutral")
            m1.metric(
                "Congress Signal",
                c_label,
                delta=f"{c_val:+.3f}",
                delta_color="normal",
                help=(
                    "Net congressional buy/sell activity over the lookback window. "
                    "Positive = more purchases than sales by members of Congress (STOCK Act disclosures)."
                ),
            )
            m2.metric(
                "Insider Signal",
                i_label,
                delta=f"{i_val:+.3f}",
                delta_color="normal",
                help=(
                    "Net insider buy/sell from SEC Form 4 filings. "
                    "Insiders buying their own stock is typically a bullish signal; "
                    "heavy selling may indicate caution."
                ),
            )
            net_buys = c_sig.get("congress_net_buys", 0)
            net_str  = f"{net_buys:+}" if isinstance(net_buys, (int, float)) else str(net_buys)
            st.metric(
                "Net Congress Buys (90d)",
                net_str,
                help=(
                    "Net congressional buy/sell activity over the lookback window. "
                    "Positive = more purchases than sales by members of Congress (STOCK Act disclosures)."
                ),
            )
            st.caption(f"{net_str} net congress buys (90d). Insider: {i_label.lower()}.")

st.divider()

# ---------------------------------------------------------------------------#
# Section 6 — Options Flow (full-width)
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300)
def _load_options_summary(ticker: str) -> dict:
    try:
        from src.options.options_data import get_options_signal
        from src.options.implied_vol import get_iv_signal
        return {
            "options": get_options_signal(ticker, settings.options_dir),
            "iv":      get_iv_signal(ticker, settings.options_dir),
        }
    except Exception:
        return {}


with st.container(border=True):
    st.page_link("pages/9_options_flow.py", label="🎯 Options Flow →")
    options_data = _load_options_summary(selected)
    if not options_data:
        st.info("No data — run ▶ Run All Analysis first.")
    else:
        opt = options_data.get("options", {})
        iv  = options_data.get("iv", {})
        c1, c2, c3 = st.columns(3)
        pc = opt.get("put_call_ratio")
        c1.metric(
            "P/C Ratio",
            f"{pc:.2f}" if pc is not None else "—",
            help=(
                "Put/Call volume ratio. Below 1.0 = more calls than puts = bullish sentiment. "
                "Above 1.2 = elevated put buying = bearish hedge activity."
            ),
        )
        flow       = opt.get("flow_signal", 0.0)
        flow_label = "Bullish" if flow > 0 else ("Bearish" if flow < 0 else "Neutral")
        c2.metric(
            "Flow Signal",
            flow_label,
            delta=f"{flow:+.3f}",
            delta_color="normal",
            help="Net directional signal from options flow: call volume vs put volume weighted by open interest.",
        )
        iv_spike = iv.get("iv_spike", False)
        c3.metric(
            "IV Spike",
            "Yes ⚠️" if iv_spike else "No ✅",
            help=(
                "Implied Volatility spike flag. Triggers when IV exceeds Historical Volatility "
                "by more than 20 percentage points — often signals expected news event or elevated fear."
            ),
        )
        pc_note = (
            "More calls than puts." if (pc is not None and pc < 1.0)
            else ("More puts than calls." if pc is not None else "")
        )
        iv_note = "IV is normal." if not iv_spike else "IV spike detected."
        st.caption(f"{pc_note} {iv_note}".strip())

st.divider()

# ---------------------------------------------------------------------------#
# Section 7 — Sector Position (full-width)
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300)
def _load_sector_summary(ticker: str) -> dict:
    sector_name: str | None = None
    peers: list[str] = []
    for sname, slist in SECTOR_PEERS.items():
        if ticker in slist:
            sector_name = sname
            peers = slist
            break
    if sector_name is None:
        return {}

    LOOKBACK = 63  # ≈ 3 trading months
    ticker_return: float | None = None
    peer_returns:  list[float]  = []

    fp = settings.raw_daily_dir / f"{ticker}.parquet"
    if fp.exists():
        df = pd.read_parquet(fp)
        if not df.empty and "Close" in df.columns:
            s = df["Close"].dropna().tail(LOOKBACK + 1)
            if len(s) >= 2:
                ticker_return = (s.iloc[-1] / s.iloc[0] - 1) * 100

    for peer in peers:
        if peer == ticker:
            continue
        pfp = settings.raw_daily_dir / f"{peer}.parquet"
        if not pfp.exists():
            continue
        try:
            pdf = pd.read_parquet(pfp)
            if not pdf.empty and "Close" in pdf.columns:
                ps = pdf["Close"].dropna().tail(LOOKBACK + 1)
                if len(ps) >= 2:
                    peer_returns.append((ps.iloc[-1] / ps.iloc[0] - 1) * 100)
        except Exception:
            continue

    peer_avg       = sum(peer_returns) / len(peer_returns) if peer_returns else None
    delta_vs_peers = (
        ticker_return - peer_avg
        if ticker_return is not None and peer_avg is not None
        else None
    )
    peer_names = [p for p in peers if p != ticker]
    return {
        "sector":          sector_name,
        "ticker_return":   ticker_return,
        "peer_avg":        peer_avg,
        "delta_vs_peers":  delta_vs_peers,
        "peer_names":      peer_names,
        "n_peers":         len(peer_returns),
    }


sector_data = _load_sector_summary(selected)

with st.container(border=True):
    sector_name = sector_data.get("sector", "Unknown")
    st.page_link("pages/11_sector_analysis.py", label=f"🗂️ Sector Position — {sector_name} →")
    if not sector_data:
        st.info("No data — run ▶ Run All Analysis first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Sector",
            sector_data.get("sector", "Unknown"),
            help="GICS sector classification based on the curated peer map.",
        )
        tr = sector_data.get("ticker_return")
        dv = sector_data.get("delta_vs_peers")
        c2.metric(
            f"{selected} 3M Return",
            f"{tr:+.1f}%" if tr is not None else "—",
            delta=f"vs peers: {dv:+.1f}%" if dv is not None else None,
            delta_color="normal",
            help=(
                "Compares the ticker's 63-trading-day price return against the average return "
                "of its sector peer group. Positive delta = outperforming peers."
            ),
        )
        pa      = sector_data.get("peer_avg")
        n_peers = sector_data.get("n_peers", 0)
        c3.metric(
            "Peer Avg 3M",
            f"{pa:+.1f}%" if pa is not None else "—",
            delta=f"{n_peers} peers with data",
            delta_color="off",
            help=(
                "Compares the ticker's 63-trading-day price return against the average return "
                "of its sector peer group. Positive delta = outperforming peers."
            ),
        )
        if tr is not None and dv is not None and pa is not None:
            direction = "outperforming" if dv > 0 else "underperforming"
            st.caption(
                f"{selected} is {direction} its {n_peers} {sector_name} peers "
                f"by {abs(dv):.1f}% over 3 months "
                f"({tr:+.1f}% vs peer avg {pa:+.1f}%)."
            )
        peer_names = sector_data.get("peer_names", [])
        if peer_names:
            st.caption("Peers: " + "  ·  ".join(peer_names[:8]))
