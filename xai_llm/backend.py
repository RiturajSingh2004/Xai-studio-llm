"""Model backends. A backend answers one question:

    "Given this prompt, what are the most likely FIRST output tokens and their
     log-probabilities?"

`OllamaBackend` asks a local Ollama server (POST /api/generate with
logprobs=true, num_predict=1). Results are cached per prompt and requests run
in parallel threads.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from xai_llm.config import DEFAULT_HOST, DEFAULT_MODEL


class BackendError(RuntimeError):
    """Raised with a message that tells the user how to fix the problem."""


def token_key(token: str, raw_bytes=None) -> str:
    """Stable identity of a token (its bytes, so ' Paris' != 'Paris')."""
    if raw_bytes:
        try:
            return bytes(raw_bytes).hex()
        except (TypeError, ValueError):
            pass
    return token.encode("utf-8", "replace").hex()


@dataclass
class TopK:
    """Top-N next-token distribution, sorted by descending log-probability."""

    keys: List[str]
    tokens: List[str]
    logprobs: np.ndarray

    def as_dict(self) -> Dict[str, float]:
        return dict(zip(self.keys, self.logprobs.tolist()))

    @property
    def floor(self) -> float:
        """Any token NOT in the list has log-probability <= this value."""
        return float(self.logprobs.min())


def make_topk(entries: Sequence[tuple]) -> TopK:
    """entries: (token, logprob, bytes_or_None). De-duplicates and sorts."""
    best: Dict[str, tuple] = {}
    for tok, lp, by in entries:
        k = token_key(tok, by)
        if k not in best or lp > best[k][1]:
            best[k] = (tok, float(lp))
    items = sorted(best.items(), key=lambda kv: -kv[1][1])
    return TopK(
        keys=[k for k, _ in items],
        tokens=[v[0] for _, v in items],
        logprobs=np.array([v[1] for _, v in items], dtype=float),
    )


def parse_ollama_logprobs(resp: dict) -> TopK:
    lps = resp.get("logprobs")
    if not lps:
        raise BackendError(
            "Ollama answered but returned no log-probabilities. Update Ollama to a "
            "recent release (logprobs support is recent) and make sure the model is "
            "served by it, then retry."
        )
    first = lps[0]
    entries = [(t["token"], t["logprob"], t.get("bytes")) for t in (first.get("top_logprobs") or [])]
    entries.append((first["token"], first["logprob"], first.get("bytes")))
    return make_topk(entries)


class Backend:
    """Shared caching + parallel querying. Subclasses implement `_fetch`."""

    name = "backend"

    def __init__(self, workers: int = 4):
        self.workers = max(1, int(workers))
        self._cache: Dict[str, TopK] = {}
        self._lock = threading.Lock()
        self.requests = 0
        self.cache_hits = 0

    def _fetch(self, prompt: str) -> TopK:  # pragma: no cover - interface
        raise NotImplementedError

    def query(self, prompt: str, use_cache: bool = True) -> TopK:
        if use_cache:
            with self._lock:
                hit = self._cache.get(prompt)
                if hit is not None:
                    self.cache_hits += 1
                    return hit
        tk = self._fetch(prompt)
        with self._lock:
            self.requests += 1
            self._cache[prompt] = tk
        return tk

    def query_many(
        self, prompts: Sequence[str], on_progress: Optional[Callable[[int], None]] = None
    ) -> List[TopK]:
        counts: Dict[str, int] = {}
        for p in prompts:
            counts[p] = counts.get(p, 0) + 1

        results: Dict[str, TopK] = {}
        todo: List[str] = []
        cached_n = 0
        with self._lock:
            for p, c in counts.items():
                if p in self._cache:
                    results[p] = self._cache[p]
                    self.cache_hits += c
                    cached_n += c
                else:
                    todo.append(p)
        if on_progress and cached_n:
            on_progress(cached_n)

        if todo:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(self.query, p): p for p in todo}
                for fut in as_completed(futures):
                    p = futures[fut]
                    results[p] = fut.result()
                    if on_progress:
                        on_progress(counts[p])
        return [results[p] for p in prompts]


class OllamaBackend(Backend):
    """Local Ollama server. `mode` is 'complete' (raw text continuation, no chat
    template) or 'chat' (Ollama applies the model's chat template, thinking off)."""

    name = "ollama"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        mode: str = "complete",
        top_logprobs: int = 20,
        workers: int = 4,
        timeout: float = 180.0,
        seed: int = 0,
    ):
        super().__init__(workers)
        if mode not in ("complete", "chat"):
            raise ValueError("mode must be 'complete' or 'chat'")
        self.model = model
        self.host = host.rstrip("/")
        self.mode = mode
        self.top_logprobs = int(top_logprobs)
        self.timeout = timeout
        self.seed = seed

    def _payload(self, prompt: str, top_n: int) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "logprobs": True,
            "top_logprobs": top_n,
            "options": {"temperature": 0, "seed": self.seed, "num_predict": 1},
        }
        if self.mode == "complete":
            payload["raw"] = True  # no template: the model just continues the text
        else:
            payload["think"] = False  # answer directly, no reasoning block first
        return payload

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _fetch(self, prompt: str) -> TopK:
        top_n = self.top_logprobs
        last_err: Optional[Exception] = None
        for _ in range(4):
            try:
                return parse_ollama_logprobs(self._post(self._payload(prompt, top_n)))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                if e.code == 404:
                    raise BackendError(
                        f"Ollama has no model called '{self.model}'. Run:  ollama pull {self.model}"
                    ) from e
                if e.code == 400 and "logprob" in body.lower() and top_n > 1:
                    top_n = max(1, top_n // 2)  # this server allows fewer alternatives
                    self.top_logprobs = top_n
                    last_err = e
                    continue
                raise BackendError(f"Ollama returned HTTP {e.code}: {body[:300]}") from e
            except urllib.error.URLError as e:
                raise BackendError(
                    f"Cannot reach Ollama at {self.host}. Is it running? Start it with `ollama serve` "
                    f"(or open the Ollama app). Details: {e.reason}"
                ) from e
            except TimeoutError as e:
                last_err = e
        raise BackendError(f"Ollama did not answer in time: {last_err}")
