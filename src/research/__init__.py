"""
Purpose: Deep-research agent package — discovers supplier→buyer relationships for a stock by driving
         local Ollama models + a SearXNG search engine in a multi-round, difficulty-scaled loop.

Connections:
  - config/settings.py: OllamaSettings / SearxSettings / research_* tunables
  - src/analytics/supply_chain.py: approved findings are written here as graph edges (source="research")
  - src/analytics/company_meta.py, src/utils/tickers.py: name↔ticker resolution
  - dashboard/pages/21_supply_chain.py, 22_deep_research.py + cli/research.py: entry points

In:  a target ticker
Out: staged SupplierFinding review queue (data/research/<ticker>.json); approved edges → supply_chain.json

ResearchError is defined here (no imports) so every submodule can raise/catch it without cycles.
"""


class ResearchError(RuntimeError):
    """Raised when a research backend (Ollama or SearXNG) is unreachable or misbehaves."""
