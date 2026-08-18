from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: SHA-keyed on-disk cache for the deep-research agent — memoizes slow Ollama and SearXNG
         round-trips so identical prompts/queries (within a run and across re-runs) return instantly.

Connections:
  - config/settings.py: research_cache_dir for the cache location
  - src/research/ollama_client.py, search_client.py, page_fetch.py: wrap their HTTP calls in cached()
  - writes only under data/research/cache/ — no other project data is touched

In:  a namespace string, a JSON-serialisable payload (the cache key material), and a producer fn
Out: the produced value (dict/list/str), transparently served from disk on a hit
"""


class ResearchCache:
    """Tiny JSON file cache keyed by SHA-256 of (namespace + payload).

    Each entry is one file `cache_dir/<namespace>_<hash>.json` holding
    `{"created": <epoch>, "value": <payload>}`. Values must be JSON-serialisable
    (strings, dicts, lists) — which covers every LLM/search response we cache.
    """

    def __init__(self, settings=None, cache_dir: Optional[Path] = None, ttl_seconds: Optional[int] = None):
        if cache_dir is not None:
            self._dir = cache_dir
        elif settings is not None:
            self._dir = settings.research_cache_dir
        else:
            self._dir = Path("data") / "research" / "cache"
        self._ttl = ttl_seconds  # None ⇒ entries never expire

    # ------------------------------------------------------------------
    # Keying
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(namespace: str, payload: Any) -> str:
        """Deterministic hash of the payload — order-stable via sort_keys."""
        blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
        digest = hashlib.sha256(f"{namespace}:{blob}".encode("utf-8")).hexdigest()[:32]
        return digest

    def _path(self, namespace: str, key: str) -> Path:
        safe_ns = "".join(c for c in namespace if c.isalnum() or c in "-_") or "cache"
        return self._dir / f"{safe_ns}_{key}.json"

    # ------------------------------------------------------------------
    # Get / set
    # ------------------------------------------------------------------

    def get(self, namespace: str, key: str) -> Optional[Any]:
        path = self._path(namespace, key)
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if self._ttl is not None:
            created = entry.get("created", 0)
            if time.time() - created > self._ttl:
                return None
        return entry.get("value")

    def set(self, namespace: str, key: str, value: Any) -> None:
        path = self._path(namespace, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"created": time.time(), "value": value}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:  # cache write failures must never break a run
            logger.warning("Could not write cache entry %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Memoize helper
    # ------------------------------------------------------------------

    def cached(self, namespace: str, payload: Any, producer: Callable[[], Any]) -> Any:
        """Return a cached value for `payload`, else run `producer()`, store, and return it.

        `producer` is only called on a miss — that's what saves the HTTP round-trip.
        A `None` result from the producer is returned but not cached (so transient
        failures are retried next time rather than remembered).
        """
        key = self.make_key(namespace, payload)
        hit = self.get(namespace, key)
        if hit is not None:
            logger.debug("cache hit  [%s] %s", namespace, key)
            return hit
        value = producer()
        if value is not None:
            self.set(namespace, key, value)
        return value

    def clear(self) -> int:
        """Delete all cache files. Returns the number removed."""
        if not self._dir.exists():
            return 0
        n = 0
        for f in self._dir.glob("*.json"):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
        return n
