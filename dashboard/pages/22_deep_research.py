"""Page 22 — Deep Research.

Standalone control surface for the supplier-discovery agent (Ollama + SearXNG). Pick a target
ticker, an endpoint/model and a round budget, hit Start, and review discovered suppliers with their
source links before promoting them into the supply-chain graph (page 21). Scoped to supplier
discovery — it grows the same data/supply_chain.json the map reads.

Connections:
  - src/research/runner.py: run_supplier_research() drives the agent
  - src/research/ollama_client.py, search_client.py: endpoint health + model list
  - src/research/research_store.py + dashboard/components/research_panel.py: review + approve
  - src/analytics/supply_chain.py: approved findings become graph edges (source="research")
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from config.settings import get_settings
from dashboard.components.research_panel import render_findings_review
from dashboard.components.ticker_selector import render_ticker_sidebar
from src.research.ollama_client import OllamaClient
from src.research.research_store import ResearchStore
from src.research.runner import run_supplier_research
from src.research.search_client import SearxClient
from src.utils.input_sanitize import clean_ticker

st.set_page_config(page_title="Deep Research", page_icon="🔬", layout="wide")

from dashboard.components.market_clock import render_market_clock
render_market_clock()

st.title("🔬 Deep Research")
st.caption(
    "Local-LLM web research to discover a stock's suppliers. Powered by your Ollama models + SearXNG, "
    "with difficulty-scaled rounds. Findings are staged for review — approve to add them to the map."
)
st.divider()

settings = get_settings()
selected = render_ticker_sidebar()
store = ResearchStore(settings)


# ---------------------------------------------------------------------------#
# Backend health
# ---------------------------------------------------------------------------#

ollama = OllamaClient(settings, use_cache=False)
searx = SearxClient(settings, use_cache=False)

ollama_ok, ollama_msg = ollama.health()
searx_ok, searx_msg = searx.health()

hc1, hc2 = st.columns(2)
with hc1:
    (st.success if ollama_ok else st.error)(f"**Ollama** — {ollama_msg}")
with hc2:
    (st.success if searx_ok else st.error)(f"**SearXNG** — {searx_msg}")

if not ollama_ok or not searx_ok:
    st.info(
        "Both services must be running. Configure endpoints with `OLLAMA_BASE_URL` "
        f"(now `{settings.ollama.base_url}`) and `SEARX_BASE_URL` (now `{settings.searx.base_url}`) "
        "in your `.env`."
    )

models = ollama.list_models() if ollama_ok else []


# ---------------------------------------------------------------------------#
# Research controls (mirrors the Deep Research panel: target, rounds, engine, endpoint, model)
# ---------------------------------------------------------------------------#

with st.form("deep_research"):
    fc1, fc2 = st.columns([2, 1])
    with fc1:
        target_raw = st.text_input("Target ticker", value=selected,
                                   help="The company whose suppliers you want to discover.")
    with fc2:
        round_choice = st.selectbox(
            "Rounds", ["Auto"] + [str(i) for i in range(1, settings.research_max_rounds + 1)],
            help="Auto scales the number of research rounds to the difficulty of the target.",
        )

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.text_input("Search engine", value=f"SearXNG · {settings.searx.base_url}", disabled=True)
    with sc2:
        st.text_input("Endpoint", value=settings.ollama.base_url, disabled=True)
    with sc3:
        model = st.selectbox("Model", models or ["(none installed)"], disabled=not ollama_ok)

    start = st.form_submit_button("▶ Start", type="primary", disabled=not (ollama_ok and searx_ok))

if start:
    target = clean_ticker(target_raw)
    if not target:
        st.error(f"Invalid ticker: {target_raw!r}")
    else:
        if model and model != "(none installed)":
            settings.ollama.model = model
        rounds = "auto" if round_choice == "Auto" else int(round_choice)

        progress = st.progress(0.0, text="Starting…")

        def _tick(step: int, total: int, label: str) -> None:
            pct = min(1.0, step / total) if total else 0.0
            progress.progress(pct, text=label)

        with st.spinner(f"Researching suppliers of {target}…"):
            result = run_supplier_research(target, settings, rounds=rounds, progress_cb=_tick)
        progress.empty()

        if result.get("error"):
            st.error(result["error"])
        else:
            st.success(
                f"Found {len(result['findings'])} candidate supplier(s) for {target} over "
                f"{result.get('rounds_used', 0)} round(s)."
            )
            st.cache_data.clear()


# ---------------------------------------------------------------------------#
# Review — the target's staged findings, plus any prior runs
# ---------------------------------------------------------------------------#

st.divider()
staged_tickers = store.list_tickers()

target_clean = clean_ticker(target_raw) or selected
if target_clean and target_clean in staged_tickers:
    st.subheader(f"Review — {target_clean}")
    render_findings_review(target_clean, store, key_prefix="page22")

other = [t for t in staged_tickers if t != target_clean]
if other:
    st.subheader("Other staged research")
    pick = st.selectbox("Ticker", other)
    if pick:
        render_findings_review(pick, store, key_prefix="page22_other")
elif not staged_tickers:
    st.caption("No staged research yet. Run the agent above to generate a review list.")
