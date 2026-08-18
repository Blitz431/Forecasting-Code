from __future__ import annotations

import json
import re
from typing import Any, Optional

import requests

from src.research import ResearchError
from src.research.cache import ResearchCache
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

"""
Purpose: Thin HTTP client for a local Ollama server — lists installed models and runs chat
         completions, including schema-constrained JSON output for the research agent.

Connections:
  - config/settings.py: settings.ollama.{base_url, model, request_timeout, num_ctx}
  - src/research/cache.py: chat responses are memoised (identical prompts skip the LLM round-trip)
  - src/research/supply_chain_agent.py: calls chat_json() for query generation + supplier extraction
  - dashboard pages 21/22 + cli/research.py: call list_models() to populate model dropdowns

In:  chat messages (list of {role, content}); an optional model name and JSON schema
Out: assistant text (chat) or a parsed dict (chat_json); ResearchError if the server is unreachable
"""


def parse_json_loose(text: str) -> Any:
    """Best-effort parse of an LLM string into JSON.

    Handles the common ways models wrap JSON: ```json fences, leading prose, or a
    trailing explanation. Returns the parsed object, or raises ValueError.
    """
    if text is None:
        raise ValueError("empty response")
    s = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced {...} or [...] blob in the text.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        end = s.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(s[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in model response")


class OllamaClient:
    """Minimal Ollama REST client (POST /api/chat, GET /api/tags)."""

    def __init__(self, settings, cache: Optional[ResearchCache] = None, use_cache: bool = True):
        self._settings = settings
        self._base = settings.ollama.base_url.rstrip("/")
        self._timeout = settings.ollama.request_timeout
        self._default_model = settings.ollama.model
        self._num_ctx = settings.ollama.num_ctx
        self._cache = cache if cache is not None else (ResearchCache(settings) if use_cache else None)
        self._model_cache: list[str] | None = None

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def list_models(self, refresh: bool = False) -> list[str]:
        """Return installed model names (GET /api/tags). Raises ResearchError if unreachable."""
        if self._model_cache is not None and not refresh:
            return self._model_cache
        try:
            resp = requests.get(f"{self._base}/api/tags", timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise ResearchError(
                f"Ollama not reachable at {self._base} ({exc}). "
                f"Start Ollama or set OLLAMA_BASE_URL."
            ) from exc
        models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        self._model_cache = models
        return models

    def resolve_model(self, model: Optional[str] = None) -> str:
        """Pick the model to use: explicit arg → settings default → first installed."""
        if model:
            return model
        if self._default_model:
            return self._default_model
        installed = self.list_models()
        if not installed:
            raise ResearchError(
                f"No models installed on Ollama at {self._base}. Run e.g. `ollama pull llama3.1`."
            )
        return installed[0]

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        format: Any = None,
        options: Optional[dict] = None,
    ) -> str:
        """Run a non-streaming chat completion and return the assistant text.

        Cached by (model, messages, format, options) — identical calls are free.
        """
        resolved = self.resolve_model(model)
        opts = {"num_ctx": self._num_ctx}
        if options:
            opts.update(options)

        payload = {
            "model": resolved,
            "messages": messages,
            "stream": False,
            "options": opts,
        }
        if format is not None:
            payload["format"] = format

        def _produce() -> str:
            try:
                resp = requests.post(
                    f"{self._base}/api/chat", json=payload, timeout=self._timeout
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                raise ResearchError(
                    f"Ollama chat failed at {self._base} ({exc})."
                ) from exc
            return (data.get("message") or {}).get("content", "") or ""

        if self._cache is None:
            return _produce()
        # format may be a dict (schema) — json.dumps in the cache key handles it.
        return self._cache.cached("ollama_chat", payload, _produce)

    def chat_json(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        schema: Any = None,
    ) -> Any:
        """Chat with JSON output. Passes a JSON schema to Ollama when given, else format="json".

        Returns the parsed object. Raises ValueError if the response isn't parseable JSON
        (callers validate the shape with Pydantic).
        """
        fmt = schema if schema is not None else "json"
        text = self.chat(messages, model=model, format=fmt)
        return parse_json_loose(text)

    def health(self) -> tuple[bool, str]:
        """(reachable, message) — used by the dashboard to show a friendly error."""
        try:
            models = self.list_models(refresh=True)
        except ResearchError as exc:
            return False, str(exc)
        if not models:
            return False, f"Ollama is up at {self._base} but has no models installed."
        return True, f"{len(models)} model(s) available at {self._base}."
