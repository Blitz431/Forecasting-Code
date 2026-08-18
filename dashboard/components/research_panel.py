from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from src.research.research_store import ResearchStore
from src.research.sources import tier_label

"""
Purpose: Shared Streamlit panel that renders staged supplier findings for review, showing the SOURCE
         link behind every claim, and promotes the ones a user approves into the supply-chain graph.

Connections:
  - src/research/research_store.py: loads data/research/<ticker>.json and approve()/discard()
  - src/research/sources.py: trust-tier label per evidence link
  - dashboard/pages/21_supply_chain.py, 22_deep_research.py: both call render_findings_review()
  - on approval, edges land in data/supply_chain.json (source="research") → page 21 graph

In:  a ticker and a ResearchStore
Out: interactive review table; side effect = graph writes + cache clear + rerun on approve/discard
"""


def render_findings_review(ticker: str, store: ResearchStore, key_prefix: str = "") -> None:
    """Render the review table for one ticker's staged findings, with approve/discard controls."""
    findings = store.load_findings(ticker)
    if not findings:
        st.info(f"No staged supplier findings for {ticker}. Run the research agent to generate some.")
        return

    st.caption(
        "Each row is an **estimate** the agent extracted from the web. Check the **Source** link before "
        "approving — approved rows are written to the graph as `source=\"research\"` with that link attached."
    )

    rows = []
    for f in findings:
        primary = f.evidence_urls[0] if f.evidence_urls else ""
        rows.append({
            "Approve": False,
            "Supplier": f.supplier_name,
            "Ticker": f.supplier_ticker or "—",
            "Type": f.rel_type,
            "Dep %": f.dependency_pct,
            "Confidence": round(f.confidence, 2),
            "Sources": len(f.evidence_urls),
            "Trust": tier_label(primary) if primary else "—",
            "Source": primary,
            "Rationale": f.rationale,
        })
    df = pd.DataFrame(rows)

    edited = st.data_editor(
        df,
        hide_index=True,
        use_container_width=True,
        key=f"review_editor_{ticker}_{key_prefix}",
        column_config={
            "Approve": st.column_config.CheckboxColumn("✓", help="Select rows to approve/discard", width="small"),
            "Dep %": st.column_config.NumberColumn("Dep %", help="Est. % of supplier revenue from this buyer", format="%.0f"),
            "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
            "Source": st.column_config.LinkColumn("Source", help="Evidence page the claim came from", display_text="open ↗"),
            "Rationale": st.column_config.TextColumn("Rationale", width="medium"),
        },
        disabled=["Supplier", "Ticker", "Type", "Dep %", "Confidence", "Sources", "Trust", "Source", "Rationale"],
    )

    selected_keys = [f.key for f, take in zip(findings, edited["Approve"].tolist()) if take]

    c1, c2, c3 = st.columns([1.4, 1.2, 3])
    with c1:
        if st.button("✅ Approve selected", type="primary", disabled=not selected_keys,
                     key=f"approve_{ticker}_{key_prefix}"):
            res = store.approve(ticker, selected_keys)
            if res["added"]:
                st.success(f"Added to graph: {', '.join(res['added'])}")
            if res["skipped"]:
                st.info(f"Already present (skipped): {', '.join(res['skipped'])}")
            st.cache_data.clear()
            st.rerun()
    with c2:
        if st.button("🗑️ Discard selected", disabled=not selected_keys,
                     key=f"discard_{ticker}_{key_prefix}"):
            n = store.discard(ticker, selected_keys)
            st.warning(f"Discarded {n} finding(s).")
            st.cache_data.clear()
            st.rerun()
    with c3:
        st.caption(f"{len(findings)} staged · {len(selected_keys)} selected")
