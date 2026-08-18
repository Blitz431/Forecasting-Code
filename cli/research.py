import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.research import ResearchError
from src.research.ollama_client import OllamaClient
from src.research.runner import run_supplier_research
from src.utils.logging import setup_logger

logger = setup_logger("cli.research")

"""
Purpose: CLI for the deep-research supplier-discovery agent — runs the Ollama+SearXNG agent for a
         ticker and stages discovered suppliers in the review queue.

Connections:
  - src/research/runner.py: run_supplier_research() does the work
  - src/research/ollama_client.py: --list-models sanity-checks the endpoint
  - config/settings.py: ollama/searx endpoints, research_* tunables
  - writes data/research/<ticker>.json (review queue), read by dashboard pages 21/22

In:  --ticker (target), --rounds (auto|N), or --list-models
Out: staged review queue file; a findings table printed to stdout (no graph writes — approval is manual)
"""


def _print_findings(result: dict) -> None:
    """Print the staged findings as an ASCII table (Windows-console safe — no arrows)."""
    if result.get("error"):
        print(f"\nERROR: {result['error']}\n")
        return
    findings = result.get("findings", [])
    print(f"\nSuppliers of {result['ticker']} — {len(findings)} candidate(s), "
          f"{result.get('rounds_used', 0)} round(s). Review/approve in the dashboard.\n")
    print(f"{'Supplier':<26} {'Ticker':<8} {'Type':<16} {'Dep%':>5} {'Conf':>5}  Source")
    print("-" * 92)
    for f in findings:
        supplier = (f.get("supplier_name") or "")[:25]
        ticker = f.get("supplier_ticker") or "-"
        rtype = (f.get("rel_type") or "")[:15]
        pct = f.get("dependency_pct")
        pct_s = f"{pct:.0f}" if isinstance(pct, (int, float)) else "-"
        conf = f.get("confidence", 0.0)
        urls = f.get("evidence_urls") or []
        src = urls[0] if urls else "(none)"
        print(f"{supplier:<26} {ticker:<8} {rtype:<16} {pct_s:>5} {conf:>5.2f}  {src}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoStockAnalyzer — deep-research supplier discovery (Ollama + SearXNG)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ticker", type=str, default=None, help="Target ticker to research suppliers for")
    parser.add_argument("--rounds", type=str, default="auto",
                        help='Research rounds: "auto" (difficulty-scaled) or an integer (default: auto)')
    parser.add_argument("--list-models", action="store_true",
                        help="List installed Ollama models and exit (endpoint sanity check)")
    args = parser.parse_args()

    settings = get_settings()

    if args.list_models:
        try:
            models = OllamaClient(settings, use_cache=False).list_models(refresh=True)
        except ResearchError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        print(f"\nOllama models at {settings.ollama.base_url}:")
        for m in models:
            print(f"  - {m}")
        print()
        return

    if not args.ticker:
        parser.error("--ticker is required (or use --list-models)")

    rounds = "auto"
    if args.rounds != "auto":
        try:
            rounds = int(args.rounds)
        except ValueError:
            parser.error("--rounds must be 'auto' or an integer")

    logger.info("Researching suppliers of %s (rounds=%s)", args.ticker, rounds)

    def _progress(step: int, total: int, label: str) -> None:
        logger.info("[%d/%d] %s", step, total, label)

    result = run_supplier_research(args.ticker, settings, rounds=rounds, progress_cb=_progress)
    _print_findings(result)
    if result.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
