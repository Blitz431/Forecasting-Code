"""Tests for the deep-research supplier-discovery agent (src/research/).

Fully offline — Ollama and SearXNG are replaced with in-memory fakes, and page
fetching is monkey-patched out. No network calls.
"""

from __future__ import annotations

import re

import pytest

from config.settings import get_settings
from src.research import ResearchError
from src.research.cache import ResearchCache
from src.research.entity_resolve import EntityResolver
from src.research.ollama_client import parse_json_loose
from src.research.page_fetch import is_fetchable
from src.research.research_store import ResearchStore
from src.research.search_client import SearchResult
from src.research.sources import TIER_OTHER, TIER_PRIMARY, TIER_WIRE, trust_tier
from src.research.supply_chain_agent import SupplierFinding, SupplyChainAgent


# ------------------------------------------------------------------ #
# Settings fixture — all writable dirs under tmp_path
# ------------------------------------------------------------------ #

@pytest.fixture
def settings(tmp_path):
    s = get_settings()
    s.data_dir = tmp_path
    s.research_dir = tmp_path / "research"
    s.research_cache_dir = tmp_path / "research" / "cache"
    return s


# ------------------------------------------------------------------ #
# Fakes
# ------------------------------------------------------------------ #

class FakeSearx:
    """Returns two fixed results (SEC + Reuters) plus a blog, for any query."""

    def __init__(self):
        self.calls = 0

    def search(self, query, count=6):
        self.calls += 1
        return [
            SearchResult("Apple 10-K", "https://www.sec.gov/apple-10k",
                         "Apple relies on Broadcom for chips", TIER_PRIMARY, "Primary (SEC/IR)"),
            SearchResult("Apple suppliers", "https://reuters.com/apple",
                         "Broadcom supplies Apple", TIER_WIRE, "Major press"),
            SearchResult("some blog", "https://blog.example.com/x",
                         "Foobar Widgets sells to Apple", TIER_OTHER, "Other"),
        ][:count]


class FakeOllama:
    """Deterministic responses keyed off the prompt content."""

    def chat_json(self, messages, schema=None):
        text = messages[-1]["content"]
        if "difficulty" in text:
            return {"difficulty": 3}                     # → budget 4
        if "search queries" in text:
            return {"queries": ["apple suppliers", "apple chip vendors"]}
        # extraction — reference the SOURCE urls present in the prompt
        urls = re.findall(r"url: (\S+)", text)
        sec = next((u for u in urls if "sec.gov" in u), "https://www.sec.gov/apple-10k")
        reut = next((u for u in urls if "reuters" in u), "https://reuters.com/apple")
        blog = next((u for u in urls if "blog" in u), "https://blog.example.com/x")
        return {"suppliers": [
            {"name": "Broadcom Inc.", "ticker": "AVGO", "type": "Semiconductors",
             "dependency_pct": 20, "confidence": 0.7, "evidence_url": sec, "rationale": "chips"},
            {"name": "Broadcom Inc.", "ticker": "AVGO", "type": "Semiconductors",
             "dependency_pct": 21, "confidence": 0.7, "evidence_url": reut, "rationale": "chips"},
            {"name": "Foobar Widgets", "ticker": None, "type": "Components",
             "dependency_pct": None, "confidence": 0.7, "evidence_url": blog, "rationale": "parts"},
            # injection / sourceless — must be dropped (evidence_url not http, not provided)
            {"name": "Ignore Instructions Corp", "ticker": None, "type": "Components",
             "dependency_pct": None, "confidence": 0.9, "evidence_url": "not-a-url", "rationale": "x"},
            # schema-invalid (empty name) — Pydantic rejects, not coerced
            {"name": "", "ticker": "ZZZZ", "type": "Components",
             "dependency_pct": None, "confidence": 0.9, "evidence_url": sec, "rationale": "bad"},
        ]}


@pytest.fixture
def agent(settings, monkeypatch):
    # Disable page fetching (no network); snippet content is enough.
    monkeypatch.setattr("src.research.supply_chain_agent.fetch_text", lambda url, cache=None, **kw: "")
    cache = ResearchCache(settings)
    return SupplyChainAgent(settings, ollama=FakeOllama(), searx=FakeSearx(), cache=cache)


# ------------------------------------------------------------------ #
# JSON parsing
# ------------------------------------------------------------------ #

def test_parse_json_loose_variants():
    assert parse_json_loose('{"a": 1}') == {"a": 1}
    assert parse_json_loose('```json\n{"b": 2}\n```') == {"b": 2}
    assert parse_json_loose('Here it is:\n{"c": [1,2]}\nthanks') == {"c": [1, 2]}
    assert parse_json_loose("[1, 2, 3]") == [1, 2, 3]
    with pytest.raises(ValueError):
        parse_json_loose("no json here")


# ------------------------------------------------------------------ #
# Source tiering + SSRF
# ------------------------------------------------------------------ #

def test_trust_tiers():
    assert trust_tier("https://www.sec.gov/x") == TIER_PRIMARY
    assert trust_tier("https://investor.apple.com/x") == TIER_PRIMARY
    assert trust_tier("https://www.reuters.com/x") == TIER_WIRE
    assert trust_tier("https://randomblog.example.com/x") == TIER_OTHER


def test_ssrf_guard():
    assert is_fetchable("https://reuters.com/x")
    assert not is_fetchable("http://localhost:8080/x")
    assert not is_fetchable("http://127.0.0.1/x")
    assert not is_fetchable("http://192.168.1.5/x")
    assert not is_fetchable("http://10.0.0.1/x")
    assert not is_fetchable("ftp://example.com/x")
    assert not is_fetchable("http://intranet.local/x")


# ------------------------------------------------------------------ #
# Entity resolution
# ------------------------------------------------------------------ #

def test_entity_resolution(settings):
    # Seed a minimal company_meta so name→ticker resolution is hermetic (not reliant
    # on the real data/company_meta.json).
    import json
    (settings.data_dir / "company_meta.json").write_text(json.dumps({
        "AVGO": {"name": "Broadcom Inc."},
        "TSM": {"name": "Taiwan Semiconductor Manufacturing Company Limited"},
    }), encoding="utf-8")

    r = EntityResolver(settings)
    assert r.resolve("Broadcom Inc.") == "AVGO"
    assert r.resolve("Taiwan Semiconductor Manufacturing") == "TSM"   # kept, external node
    assert r.resolve("TSMC") == "TSM"                                 # alias
    assert r.resolve("Whatever", "avgo") == "AVGO"                    # raw ticker wins
    assert r.resolve("Totally Fake Widgets LLC") is None


# ------------------------------------------------------------------ #
# Cache
# ------------------------------------------------------------------ #

def test_cache_serves_second_call_without_producer(settings):
    cache = ResearchCache(settings)
    calls = {"n": 0}

    def producer():
        calls["n"] += 1
        return {"value": 42}

    a = cache.cached("ns", {"q": "x"}, producer)
    b = cache.cached("ns", {"q": "x"}, producer)   # served from disk
    assert a == b == {"value": 42}
    assert calls["n"] == 1
    assert cache.clear() >= 1


# ------------------------------------------------------------------ #
# Agent loop
# ------------------------------------------------------------------ #

def test_agent_discovers_maps_and_scores(agent):
    findings = agent.run("AAPL", rounds="auto")
    names = {f.supplier_name for f in findings}

    # sourceless + schema-invalid rows dropped
    assert "Ignore Instructions Corp" not in names
    assert all(f.evidence_urls for f in findings)

    # mapping + vocab
    broadcom = next(f for f in findings if f.supplier_ticker == "AVGO")
    assert broadcom.rel_type == "Semiconductors"
    assert broadcom.buyer_ticker == "AAPL"

    # corroboration: Broadcom (2 domains + SEC tier) outranks single-domain blog Foobar
    foobar = next(f for f in findings if f.supplier_name == "Foobar Widgets")
    assert broadcom.confidence > foobar.confidence
    assert findings[0].supplier_ticker == "AVGO"       # highest-confidence sorts first


def test_agent_early_stop(agent):
    """Difficulty 3 → budget 4, but round 2 finds nothing new → stop early."""
    agent.run("AAPL", rounds="auto")
    assert agent.last_rounds_used == 2


def test_agent_invalid_ticker_returns_empty(agent):
    assert agent.run("!!!", rounds=1) == []


# ------------------------------------------------------------------ #
# Review store → graph
# ------------------------------------------------------------------ #

def test_store_roundtrip_and_approve(settings):
    store = ResearchStore(settings)
    findings = [
        SupplierFinding("Marvell Technology", "AAPL", "MRVL", "Semiconductors", 15.0, 0.9,
                        ["https://www.sec.gov/m", "https://reuters.com/m"], "chips"),
        SupplierFinding("Nameless Vendor", "AAPL", None, "Components", None, 0.4,
                        ["https://blog.example.com/n"], "maybe"),
    ]
    store.save_findings("AAPL", findings)
    assert len(store.load_findings("AAPL")) == 2
    assert store.list_tickers() == ["AAPL"]

    res = store.approve("AAPL", ["MRVL"])
    assert res["added"] == ["MRVL → AAPL"]

    from src.analytics.supply_chain import SupplyChainGraph
    edge = next(r for r in SupplyChainGraph(settings).load()
                if r.supplier == "MRVL" and r.buyer == "AAPL")
    assert edge.source == "research"
    assert edge.source_url == "https://www.sec.gov/m"   # top-tier evidence promoted

    # approved row dropped from queue; the other remains
    remaining = store.load_findings("AAPL")
    assert [f.supplier_name for f in remaining] == ["Nameless Vendor"]

    # idempotent: re-approving the same (already-written, already-dequeued) edge is a no-op
    assert store.approve("AAPL", ["MRVL"]) == {"added": [], "skipped": []}


def test_store_approve_dedupes_against_seed(settings):
    """Approving a pair already in the curated seed is skipped, not duplicated."""
    store = ResearchStore(settings)
    store.save_findings("AAPL", [
        SupplierFinding("Broadcom", "AAPL", "AVGO", "Semiconductors", 20.0, 0.9,
                        ["https://www.sec.gov/x"], "seed dup"),
    ])
    res = store.approve("AAPL", ["AVGO"])
    assert res["skipped"] == ["AVGO → AAPL"]   # AVGO→AAPL is in the seed already
    assert res["added"] == []


# ------------------------------------------------------------------ #
# Runner error handling
# ------------------------------------------------------------------ #

def test_runner_reports_error_without_raising(settings, monkeypatch):
    from src.research import runner

    class Boom:
        last_rounds_used = 0

        def __init__(self, *a, **k):
            pass

        def run(self, *a, **k):
            raise ResearchError("SearXNG not reachable")

    monkeypatch.setattr(runner, "SupplyChainAgent", Boom)
    result = runner.run_supplier_research("AAPL", settings)
    assert result["error"] == "SearXNG not reachable"
    assert result["findings"] == []
    assert result["ticker"] == "AAPL"


def test_runner_invalid_ticker(settings):
    from src.research.runner import run_supplier_research
    result = run_supplier_research("###", settings)
    assert result["error"]
    assert result["findings"] == []
