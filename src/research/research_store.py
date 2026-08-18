from __future__ import annotations

import datetime
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

from src.analytics.supply_chain import SupplyChainGraph
from src.research.supply_chain_agent import SupplierFinding
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Review-and-approve queue for agent-discovered suppliers — stages findings as JSON per ticker
         and, on approval, promotes them into the supply-chain graph with their evidence link.

Connections:
  - config/settings.py: research_dir for the per-ticker queue files
  - src/research/supply_chain_agent.py: SupplierFinding is the staged record
  - src/analytics/supply_chain.py: approve() writes edges via SupplyChainGraph.add(source="research", source_url=…)
  - dashboard/components/research_panel.py, cli/research.py: read the queue and trigger approval

In:  a ticker + its list of SupplierFinding (from the agent)
Out: data/research/<ticker>.json review files; approved rows become supply_chain.json edges

Storage mirrors src/analytics/supply_chain.py: {"ticker", "updated", "findings":[…]}, one file per ticker.
"""

_SCHEMA_VERSION = 1


class ResearchStore:
    """Persistent per-ticker review queue of SupplierFindings."""

    def __init__(self, settings=None, research_dir: Optional[Path] = None):
        self._settings = settings
        if research_dir is not None:
            self._dir = research_dir
        elif settings is not None:
            self._dir = settings.research_dir
        else:
            self._dir = Path("data") / "research"

    def _path(self, ticker: str) -> Path:
        return self._dir / f"{ticker.upper().strip()}.json"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_findings(self, ticker: str, findings: list[SupplierFinding]) -> None:
        ticker = ticker.upper().strip()
        payload = {
            "version": _SCHEMA_VERSION,
            "ticker": ticker,
            "updated": datetime.datetime.now().isoformat(timespec="seconds"),
            "findings": [asdict(f) for f in findings],
        }
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path(ticker).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_findings(self, ticker: str) -> list[SupplierFinding]:
        path = self._path(ticker)
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Could not parse research queue %s", path)
            return []
        out: list[SupplierFinding] = []
        for row in raw.get("findings", []):
            if not isinstance(row, dict):
                continue
            try:
                out.append(SupplierFinding(
                    supplier_name=str(row.get("supplier_name", "")),
                    buyer_ticker=str(row.get("buyer_ticker", ticker)).upper(),
                    supplier_ticker=row.get("supplier_ticker") or None,
                    rel_type=str(row.get("rel_type") or "Components"),
                    dependency_pct=row.get("dependency_pct"),
                    confidence=float(row.get("confidence", 0.5)),
                    evidence_urls=list(row.get("evidence_urls") or []),
                    rationale=str(row.get("rationale") or ""),
                ))
            except Exception:
                continue
        return out

    def list_tickers(self) -> list[str]:
        """Tickers that currently have a staged review file."""
        if not self._dir.exists():
            return []
        return sorted(p.stem.upper() for p in self._dir.glob("*.json"))

    # ------------------------------------------------------------------
    # Approval → graph
    # ------------------------------------------------------------------

    def approve(self, ticker: str, keys: Iterable[str]) -> dict:
        """Promote the findings whose `key` is in `keys` into the supply-chain graph.

        Each approved finding becomes an edge supplier→buyer with source="research" and the
        top-tier evidence URL as source_url. Approved (and no-longer-actionable) rows are dropped
        from the queue. Idempotent: re-approving an already-written edge is a no-op.
        Returns {"added": [...], "skipped": [...]}.
        """
        ticker = ticker.upper().strip()
        keyset = {k.upper() for k in keys}
        findings = self.load_findings(ticker)
        graph = SupplyChainGraph(self._settings)

        added, skipped, remaining = [], [], []
        for f in findings:
            if f.key not in keyset:
                remaining.append(f)
                continue
            supplier_node = (f.supplier_ticker or f.supplier_name).strip()
            ok = graph.add(
                supplier=supplier_node,
                buyer=f.buyer_ticker,
                rel_type=f.rel_type,
                dependency_pct=f.dependency_pct,
                source="research",
                source_url=f.evidence_urls[0] if f.evidence_urls else "",
            )
            (added if ok else skipped).append(f"{supplier_node} → {f.buyer_ticker}")
            # Either way the finding is resolved — don't leave it in the queue.

        if len(remaining) != len(findings):
            self.save_findings(ticker, remaining)
        return {"added": added, "skipped": skipped}

    def discard(self, ticker: str, keys: Iterable[str]) -> int:
        """Remove findings from the queue without writing them. Returns count removed."""
        ticker = ticker.upper().strip()
        keyset = {k.upper() for k in keys}
        findings = self.load_findings(ticker)
        remaining = [f for f in findings if f.key not in keyset]
        removed = len(findings) - len(remaining)
        if removed:
            self.save_findings(ticker, remaining)
        return removed
