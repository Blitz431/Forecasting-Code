from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Buyer-supplier relationship store and directed graph builder for the supply chain map.

Connections:
  - src/analytics/company_meta.py: node sizing (market cap) and colouring (sector) come from there
  - config/settings.py: data_dir for the JSON store, raw_daily_dir to tell scraped from external tickers
  - dashboard/pages/21_supply_chain.py: loads the graph, adds/removes relationships via the UI
  - dashboard/components/charts.py: supply_chain_network() consumes the DiGraph + layout produced here
  - src/research/research_store.py: approved agent findings are written here via add(source="research",
    source_url=...), so a relationship carries a link back to the web page it was discovered from

In:  data/supply_chain.json — curated, manually added, and research-discovered relationships
Out: list[Relationship], networkx.DiGraph (edge attrs: type, dependency_pct, source, source_url),
     spring layout positions

Edge direction: supplier -> buyer. "TSM -> AAPL" reads "AAPL buys from TSM".
dependency_pct: estimated % of the SUPPLIER's revenue that comes from the buyer
    (supplier-side revenue concentration, e.g. "AAPL is ~20% of AVGO's revenue").
    These are estimates, not sourced filings figures.
source_url: for research-discovered edges, the evidence page the claim came from (else "").
"""

_DEFAULT_SUPPLY_CHAIN_FILE = "supply_chain.json"
_SCHEMA_VERSION = 1

# Controlled vocabulary for relationship types — keeps the UI dropdown and the
# curated seed data from drifting apart.
RELATIONSHIP_TYPES = [
    "Semiconductors",
    "Cloud Services",
    "Components",
    "Raw Materials",
    "Logistics",
    "Distribution",
    "Payment Processing",
    "Software",
]


@dataclass
class Relationship:
    """One directed supplier -> buyer edge."""

    supplier: str
    buyer: str
    type: str = "Components"
    dependency_pct: float | None = None
    source: str = "curated"      # "curated" (seed), "manual" (UI), or "research" (agent)
    added: str = ""
    source_url: str = ""         # evidence link for research-discovered edges; "" otherwise

    @property
    def key(self) -> tuple[str, str]:
        """Identity of the relationship — the (supplier, buyer) pair."""
        return (self.supplier, self.buyer)


class SupplyChainGraph:
    """Persistent buyer-supplier relationships, loaded as a directed graph."""

    def __init__(self, settings=None, path: Path | None = None):
        self._settings = settings
        if path is not None:
            self._path = path
        elif settings is not None:
            self._path = settings.data_dir / _DEFAULT_SUPPLY_CHAIN_FILE
        else:
            self._path = Path(_DEFAULT_SUPPLY_CHAIN_FILE)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> list[Relationship]:
        """Return every stored relationship. Never raises.

        Falls back to the checked-in seed set when no JSON file exists yet, so a
        fresh clone opens with a populated map instead of an empty canvas
        (all of data/ is gitignored). The first add/remove writes the file, and
        from then on the file is authoritative.
        """
        if not self._path.exists():
            return self.seed_relationships()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Could not parse %s — treating the supply chain as empty", self._path)
            return []

        rows = raw.get("relationships", []) if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            return []

        out: list[Relationship] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            supplier = str(row.get("supplier", "")).upper().strip()
            buyer = str(row.get("buyer", "")).upper().strip()
            if not supplier or not buyer or supplier == buyer:
                continue
            pct = row.get("dependency_pct")
            try:
                pct = float(pct) if pct is not None else None
            except (TypeError, ValueError):
                pct = None
            out.append(Relationship(
                supplier=supplier,
                buyer=buyer,
                type=str(row.get("type") or "Components"),
                dependency_pct=pct,
                source=str(row.get("source") or "curated"),
                added=str(row.get("added") or ""),
                source_url=str(row.get("source_url") or ""),
            ))
        return out

    @staticmethod
    def seed_relationships() -> list[Relationship]:
        """The curated seed set, as Relationship objects."""
        from src.analytics.supply_chain_seed import SEED_RELATIONSHIPS

        return [
            Relationship(
                supplier=s, buyer=b, type=t, dependency_pct=pct,
                source="curated", added="",
            )
            for s, b, t, pct in SEED_RELATIONSHIPS
        ]

    def save(self, relationships: list[Relationship]) -> None:
        """Persist relationships, deduplicated on (supplier, buyer) and sorted."""
        seen: dict[tuple[str, str], Relationship] = {}
        for rel in relationships:
            seen[rel.key] = rel
        ordered = sorted(seen.values(), key=lambda r: (r.supplier, r.buyer))

        payload = {
            "version": _SCHEMA_VERSION,
            "updated": datetime.date.today().isoformat(),
            "relationships": [asdict(r) for r in ordered],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        supplier: str,
        buyer: str,
        rel_type: str = "Components",
        dependency_pct: float | None = None,
        source: str = "manual",
        source_url: str = "",
    ) -> bool:
        """Add one relationship. Returns False if the pair already exists or is invalid."""
        s = str(supplier).upper().strip()
        b = str(buyer).upper().strip()
        if not s or not b or s == b:
            return False

        existing = self.load()
        if any(r.key == (s, b) for r in existing):
            return False

        existing.append(Relationship(
            supplier=s,
            buyer=b,
            type=rel_type,
            dependency_pct=dependency_pct,
            source=source,
            added=datetime.date.today().isoformat(),
            source_url=source_url,
        ))
        self.save(existing)
        return True

    def remove(self, supplier: str, buyer: str) -> bool:
        """Remove one relationship. Returns False if it was not present."""
        s = str(supplier).upper().strip()
        b = str(buyer).upper().strip()
        existing = self.load()
        kept = [r for r in existing if r.key != (s, b)]
        if len(kept) == len(existing):
            return False
        self.save(kept)
        return True

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def tickers(self) -> set[str]:
        """Every ticker appearing on either side of a relationship."""
        out: set[str] = set()
        for rel in self.load():
            out.add(rel.supplier)
            out.add(rel.buyer)
        return out

    def scraped_tickers(self) -> set[str]:
        """Tickers we hold daily price data for — used to flag external companies."""
        if self._settings is None:
            return set()
        try:
            return {p.stem.upper() for p in self._settings.raw_daily_dir.glob("*.parquet")}
        except Exception:
            return set()

    def build_graph(self, relationships: list[Relationship] | None = None) -> nx.DiGraph:
        """Build the directed graph. Edges point supplier -> buyer."""
        rels = self.load() if relationships is None else relationships
        g = nx.DiGraph()
        for rel in rels:
            g.add_edge(
                rel.supplier,
                rel.buyer,
                type=rel.type,
                dependency_pct=rel.dependency_pct,
                source=rel.source,
                source_url=rel.source_url,
            )
        return g

    @staticmethod
    def layout(g: nx.DiGraph, seed: int = 42) -> dict[str, tuple[float, float]]:
        """Spring layout positions.

        The seed is deliberate: without it the layout is re-randomised on every
        Streamlit rerun and the graph appears to jump around for no reason.
        """
        if g.number_of_nodes() == 0:
            return {}
        k = 1.6 / (g.number_of_nodes() ** 0.5) if g.number_of_nodes() > 1 else None
        return nx.spring_layout(g, seed=seed, k=k, iterations=80)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def upstream(self, ticker: str) -> set[str]:
        """Companies that supply the given ticker."""
        t = str(ticker).upper().strip()
        return {r.supplier for r in self.load() if r.buyer == t}

    def downstream(self, ticker: str) -> set[str]:
        """Companies that buy from the given ticker."""
        t = str(ticker).upper().strip()
        return {r.buyer for r in self.load() if r.supplier == t}

    def neighbourhood(self, ticker: str) -> set[str]:
        """The ticker plus its direct suppliers and buyers."""
        t = str(ticker).upper().strip()
        return {t} | self.upstream(t) | self.downstream(t)
