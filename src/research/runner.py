from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Optional

from src.research import ResearchError
from src.research.research_store import ResearchStore
from src.research.supply_chain_agent import SupplyChainAgent
from src.utils.input_sanitize import clean_ticker
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Orchestrate one supplier-research run — invoke the agent, stage its findings in the review
         queue, and return a result dict. Never raises; unreachable backends surface as an "error" key.

Connections:
  - src/research/supply_chain_agent.py: runs the multi-round discovery loop
  - src/research/research_store.py: persists findings to data/research/<ticker>.json
  - dashboard/pages/21,22 + cli/research.py: the callers
  - mirrors the result-dict shape of src/news/runner.py

In:  ticker, settings, rounds ("auto" | int), optional progress callback
Out: {"ticker", "findings": [dict…], "rounds_used": int, "error": None|str}
"""

ProgressCb = Callable[[int, int, str], None]


def run_supplier_research(
    ticker: str,
    settings,
    rounds="auto",
    progress_cb: Optional[ProgressCb] = None,
) -> dict:
    """Research suppliers of `ticker`, stage them for review, and return a result dict."""
    clean = (clean_ticker(ticker) or "").upper()
    result = {"ticker": clean, "findings": [], "rounds_used": 0, "error": None}
    if not clean:
        result["error"] = f"Invalid ticker: {ticker!r}"
        return result

    try:
        agent = SupplyChainAgent(settings)
        findings = agent.run(clean, rounds=rounds, progress_cb=progress_cb)
        ResearchStore(settings).save_findings(clean, findings)
        result["findings"] = [asdict(f) for f in findings]
        result["rounds_used"] = agent.last_rounds_used
        logger.info("Research for %s: %d finding(s) over %d round(s)",
                    clean, len(findings), agent.last_rounds_used)
    except ResearchError as exc:
        result["error"] = str(exc)
        logger.warning("Research for %s failed: %s", clean, exc)
    except Exception as exc:  # never let a research run take down the caller
        result["error"] = f"Unexpected error: {exc}"
        logger.exception("Unexpected error researching %s", clean)
    return result
