import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence, TypeVar

from src.utils.logging import setup_logger

"""
Purpose: Single shared concurrency primitive for scraper-style workloads —
bounded thread pool with order-preserving results and per-item failure
isolation.

Connections:
  - src/scraper/*.py, src/news/scraper.py: import thread_map instead of
    creating their own ThreadPoolExecutor
  - src/utils/logging.py: setup_logger(__name__) for warning-level logging
    of per-item failures

In:  func, items (sequence), max_workers, label, stagger
Out: list[R | None] in the same order as `items`; failed items are None
"""

logger = setup_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def thread_map(
    func: Callable[[T], R],
    items: Sequence[T],
    max_workers: int = 8,
    label: str = "",
    stagger: float = 0.0,
) -> list[R | None]:
    """Run func over items on a bounded thread pool.

    Returns results in the SAME ORDER as `items`. An item whose call raises
    gets `None` in its slot and the exception is logged as a warning with
    `label` for context — one bad ticker must never abort the run.
    `stagger` seconds are slept between task submissions (0 = none), used to
    bound outbound request rate. max_workers <= 1 runs inline with no pool.
    Empty `items` returns [] without creating a pool.
    """
    if not items:
        return []

    def _truncated_repr(item: T) -> str:
        return repr(item)[:40]

    if max_workers <= 1:
        results: list[R | None] = []
        for i, item in enumerate(items):
            if stagger and i > 0:
                time.sleep(stagger)
            try:
                results.append(func(item))
            except Exception as exc:
                logger.warning(
                    f"[{label}] item {_truncated_repr(item)} failed: {exc}"
                )
                results.append(None)
        return results

    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, item in enumerate(items):
            if stagger and i > 0:
                time.sleep(stagger)
            futures[executor.submit(func, item)] = i

        for future, i in futures.items():
            try:
                results[i] = future.result()
            except Exception as exc:
                logger.warning(
                    f"[{label}] item {_truncated_repr(items[i])} failed: {exc}"
                )
                results[i] = None

    return results
