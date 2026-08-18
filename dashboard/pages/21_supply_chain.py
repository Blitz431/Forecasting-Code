"""Page 21 — Supply Chain.

Interactive buyer-supplier network for the tickers we track. Arrows point from
the supplier to the buyer, nodes are sized by market cap and coloured by sector.
Backed by src/analytics/supply_chain.py (Phase 9 analytics) and grown over time
through the in-page "Add relationship" form and the "Research suppliers" agent.

Connections:
  - src/analytics/supply_chain.py: SupplyChainGraph load/add/remove + RELATIONSHIP_TYPES
  - dashboard/components/charts.py: supply_chain_network() figure
  - src/research/runner.py: run_supplier_research() for the selected ticker (Ollama + SearXNG)
  - src/research/research_store.py + dashboard/components/research_panel.py: review + approve findings
  - src/analytics/company_meta.py: market cap / sector for node styling
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from dashboard.components.charts import normalize_sector, sector_legend_items, supply_chain_network
from dashboard.components.research_panel import render_findings_review
from dashboard.components.ticker_selector import render_ticker_sidebar
from src.analytics.company_meta import CompanyMeta
from src.analytics.supply_chain import RELATIONSHIP_TYPES, Relationship, SupplyChainGraph
from src.research.ollama_client import OllamaClient
from src.research.research_store import ResearchStore
from src.research.runner import run_supplier_research
from src.utils.input_sanitize import clean_ticker

st.set_page_config(page_title="Supply Chain", page_icon="🔗", layout="wide")

from dashboard.components.market_clock import render_market_clock
render_market_clock()

st.title("🔗 Supply Chain Map")
st.caption(
    "Who buys from whom. An arrow **supplier → buyer** means the buyer depends on that supplier — "
    "`TSM → AAPL` reads *AAPL buys from TSM*."
)
st.divider()

settings = get_settings()
selected = render_ticker_sidebar()

chain = SupplyChainGraph(settings)


# ---------------------------------------------------------------------------#
# Loaders
# ---------------------------------------------------------------------------#

@st.cache_data(ttl=300, show_spinner="Loading supply chain …")
def _load_relationships() -> pd.DataFrame:
    """Relationships as a DataFrame — cached so reruns don't re-read the JSON."""
    rels = SupplyChainGraph(get_settings()).load()
    cols = ["Supplier", "Buyer", "Type", "Dependency %", "Source", "Source URL", "Added"]
    if not rels:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([
        {
            "Supplier": r.supplier,
            "Buyer": r.buyer,
            "Type": r.type,
            "Dependency %": r.dependency_pct,
            "Source": r.source,
            "Source URL": r.source_url,
            "Added": r.added,
        }
        for r in rels
    ])


@st.cache_data(ttl=300, show_spinner=False)
def _load_meta(tickers: tuple[str, ...]) -> dict:
    """Cached company metadata. Only reads what is already on disk — no fetching."""
    cm = CompanyMeta(get_settings())
    cached = cm.load()
    return {t: cached[t] for t in tickers if t in cached}


rel_df = _load_relationships()

if rel_df.empty:
    st.warning(
        "No relationships found. Expected `data/supply_chain.json` — "
        "add one below to get started."
    )

all_tickers = sorted(set(rel_df["Supplier"]) | set(rel_df["Buyer"])) if not rel_df.empty else []
meta = _load_meta(tuple(all_tickers))
scraped = chain.scraped_tickers()

missing_meta = [t for t in all_tickers if t not in meta]


# ---------------------------------------------------------------------------#
# Filters
# ---------------------------------------------------------------------------#

fc1, fc2, fc3 = st.columns([1.2, 2, 1.2])

with fc1:
    focus_mode = st.radio(
        "View",
        ["Whole graph", f"{selected} neighbourhood"],
        index=0,
        help="Neighbourhood shows only the selected ticker and its direct suppliers and buyers.",
    )

with fc2:
    known_sectors = sorted({normalize_sector((meta.get(t) or {}).get("sector")) for t in all_tickers})
    sector_filter = st.multiselect(
        "Sectors", known_sectors, default=[],
        help="Empty means all sectors.",
    )

with fc3:
    min_dep = st.slider(
        "Min dependency %", 0, 50, 0,
        help="Hide relationships below this estimated revenue dependency.",
    )

# Apply filters
view = rel_df.copy()

if not view.empty and min_dep > 0:
    view = view[view["Dependency %"].fillna(0) >= min_dep]

if not view.empty and focus_mode != "Whole graph":
    # Both ends must be in the neighbourhood. Matching on either end would pull in
    # suppliers-of-suppliers, showing companies with no direct link to the ticker.
    hood = chain.neighbourhood(selected)
    view = view[view["Supplier"].isin(hood) & view["Buyer"].isin(hood)]

if not view.empty and sector_filter:
    def _sector_of(t: str) -> str:
        return normalize_sector((meta.get(t) or {}).get("sector"))
    view = view[
        view["Supplier"].map(_sector_of).isin(sector_filter)
        | view["Buyer"].map(_sector_of).isin(sector_filter)
    ]


# ---------------------------------------------------------------------------#
# Graph
# ---------------------------------------------------------------------------#

if view.empty:
    st.info("No relationships match the current filters.")
else:
    filtered_rels = [
        Relationship(
            supplier=row["Supplier"],
            buyer=row["Buyer"],
            type=row["Type"],
            dependency_pct=None if pd.isna(row["Dependency %"]) else float(row["Dependency %"]),
            source=row.get("Source", "curated") or "curated",
            source_url=row.get("Source URL", "") or "",
        )
        for _, row in view.iterrows()
    ]

    g = chain.build_graph(filtered_rels)
    pos = chain.layout(g)

    focus = selected if focus_mode != "Whole graph" else None
    fig = supply_chain_network(g, pos, meta=meta, focus=focus)
    st.plotly_chart(fig, use_container_width=True)

    # Sector legend — the figure hides its own legend to keep the canvas clean.
    legend_sectors = {(meta.get(n) or {}).get("sector") for n in g.nodes()}
    chips = " &nbsp; ".join(
        f"<span style='color:{color}'>●</span> <span style='font-size:12px'>{name}</span>"
        for name, color in sector_legend_items(legend_sectors)
    )
    st.markdown(chips, unsafe_allow_html=True)

    st.caption(
        "Dependency % is an **estimate** of the share of the *supplier's* revenue coming from that "
        "buyer. Figures are approximate, not sourced from filings."
    )

    # ----------------------------------------------------------------------#
    # Metrics
    # ----------------------------------------------------------------------#

    st.divider()
    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Relationships", g.number_of_edges())
    m2.metric("Companies", g.number_of_nodes())

    if g.number_of_nodes():
        busiest = max(g.nodes(), key=lambda n: g.in_degree(n) + g.out_degree(n))
        m3.metric(
            "Most connected", busiest,
            help=f"{g.in_degree(busiest)} suppliers, {g.out_degree(busiest)} buyers",
        )

    concentrated = int((view["Dependency %"].fillna(0) >= 20).sum())
    m4.metric(
        "High-concentration links", concentrated,
        help="Relationships where one buyer is 20%+ of the supplier's revenue.",
    )


# ---------------------------------------------------------------------------#
# Company metadata
# ---------------------------------------------------------------------------#

with st.expander("🏷️ Company data (market cap & sector)", expanded=bool(missing_meta)):
    if missing_meta:
        st.info(
            f"{len(missing_meta)} of {len(all_tickers)} companies have no cached market cap, "
            "so they render at a default size. Fetch it below (one yfinance call each)."
        )
    else:
        st.success(f"All {len(all_tickers)} companies have cached metadata.")

    mc1, mc2 = st.columns(2)
    with mc1:
        if st.button("⬇️ Fetch missing company data", disabled=not missing_meta):
            with st.spinner(f"Fetching {len(missing_meta)} companies from yfinance …"):
                CompanyMeta(settings).ensure(missing_meta)
            st.cache_data.clear()
            st.rerun()
    with mc2:
        if st.button("🔄 Refresh all company data", disabled=not all_tickers):
            with st.spinner(f"Refreshing {len(all_tickers)} companies …"):
                CompanyMeta(settings).refresh(all_tickers)
            st.cache_data.clear()
            st.rerun()


# ---------------------------------------------------------------------------#
# Add relationship
# ---------------------------------------------------------------------------#

with st.expander("➕ Add relationship", expanded=False):
    st.caption(
        "Add a supplier → buyer link. Tickers outside our scraped universe are allowed — "
        "they render as nodes without market cap."
    )

    universe = sorted(scraped) if scraped else all_tickers

    with st.form("add_relationship", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            supplier_pick = st.selectbox("Supplier (sells)", ["— type below —"] + universe)
            supplier_free = st.text_input("…or type a supplier ticker", placeholder="e.g. TSM")
        with c2:
            buyer_pick = st.selectbox("Buyer (purchases)", ["— type below —"] + universe)
            buyer_free = st.text_input("…or type a buyer ticker", placeholder="e.g. AAPL")

        c3, c4 = st.columns(2)
        with c3:
            rel_type = st.selectbox("Relationship type", RELATIONSHIP_TYPES)
        with c4:
            dep_pct = st.number_input(
                "Est. dependency % (supplier's revenue from this buyer)",
                min_value=0.0, max_value=100.0, value=0.0, step=1.0,
            )

        submitted = st.form_submit_button("Add relationship", type="primary")

    if submitted:
        supplier = clean_ticker(supplier_free) or (
            supplier_pick if supplier_pick != "— type below —" else None
        )
        buyer = clean_ticker(buyer_free) or (
            buyer_pick if buyer_pick != "— type below —" else None
        )

        if not supplier or not buyer:
            st.error("Pick or type both a supplier and a buyer.")
        elif supplier == buyer:
            st.error("A company cannot supply itself.")
        elif chain.add(
            supplier, buyer, rel_type,
            dependency_pct=dep_pct if dep_pct > 0 else None,
            source="manual",
        ):
            st.success(f"Added {supplier} → {buyer}.")
            st.cache_data.clear()
            st.rerun()
        else:
            st.info(f"{supplier} → {buyer} already exists.")


# ---------------------------------------------------------------------------#
# Research suppliers (Ollama + SearXNG agent)
# ---------------------------------------------------------------------------#

store = ResearchStore(settings)
_staged = store.load_findings(selected)

with st.expander(
    f"🔬 Research suppliers of {selected}"
    + (f" · {len(_staged)} awaiting review" if _staged else ""),
    expanded=bool(_staged),
):
    st.caption(
        "Uses your local Ollama models + SearXNG to search the web for who supplies "
        f"**{selected}**. Discovered suppliers land in a review list below with their source "
        "links — nothing hits the graph until you approve it."
    )

    try:
        _models = OllamaClient(settings, use_cache=False).list_models()
        _ollama_ok = True
    except Exception as exc:
        _models, _ollama_ok = [], False
        st.error(
            f"Ollama unavailable: {exc}\n\nStart Ollama or set `OLLAMA_BASE_URL` "
            f"(currently `{settings.ollama.base_url}`)."
        )

    rc1, rc2, rc3 = st.columns([2, 1.3, 1.2])
    with rc1:
        model = st.selectbox("Model", _models or ["(none installed)"], disabled=not _ollama_ok)
    with rc2:
        round_choice = st.selectbox(
            "Rounds", ["Auto"] + [str(i) for i in range(1, settings.research_max_rounds + 1)],
            help="Auto scales rounds to the difficulty of the target.",
        )
    with rc3:
        st.write("")
        st.write("")
        run_research = st.button("▶ Start research", type="primary", disabled=not _ollama_ok)

    if run_research:
        # Persist the chosen model for this run via settings override.
        if model and model != "(none installed)":
            settings.ollama.model = model
        rounds = "auto" if round_choice == "Auto" else int(round_choice)

        progress = st.progress(0.0, text="Starting…")

        def _tick(step: int, total: int, label: str) -> None:
            pct = min(1.0, step / total) if total else 0.0
            progress.progress(pct, text=label)

        with st.spinner(f"Researching suppliers of {selected}…"):
            result = run_supplier_research(selected, settings, rounds=rounds, progress_cb=_tick)
        progress.empty()

        if result.get("error"):
            st.error(result["error"])
        else:
            st.success(
                f"Found {len(result['findings'])} candidate supplier(s) over "
                f"{result.get('rounds_used', 0)} round(s). Review below."
            )
            st.cache_data.clear()
            _staged = store.load_findings(selected)

    if _staged:
        st.divider()
        st.markdown(f"**Review — {selected} suppliers**")
        render_findings_review(selected, store, key_prefix="page21")


# ---------------------------------------------------------------------------#
# Relationship table
# ---------------------------------------------------------------------------#

st.divider()
st.subheader("Relationships")

if view.empty:
    st.caption("Nothing to list under the current filters.")
else:
    table = view.sort_values(["Supplier", "Buyer"]).reset_index(drop=True)
    st.dataframe(
        table,
        use_container_width=True,
        height=340,
        hide_index=True,
        column_config={
            "Dependency %": st.column_config.NumberColumn("Dependency %", format="%.0f"),
            "Source URL": st.column_config.LinkColumn(
                "Source", help="Evidence link for research-discovered edges", display_text="open ↗"
            ),
        },
    )

    rc1, rc2 = st.columns([3, 1])
    with rc1:
        pairs = [f"{r.Supplier} → {r.Buyer}" for r in table.itertuples(index=False)]
        to_remove = st.selectbox("Remove a relationship", ["—"] + pairs)
    with rc2:
        st.write("")
        if st.button("🗑️ Remove", disabled=to_remove == "—"):
            supplier, buyer = to_remove.split(" → ")
            if chain.remove(supplier, buyer):
                st.success(f"Removed {to_remove}.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.warning("That relationship was not found.")
