"""A deterministic fake "LLM" with PLANTED word effects.

Used for (1) unit tests, which check that the statistics recover the planted
effects, and (2) a preview mode that needs no Ollama. Results from it say
nothing about any real model, and the app labels them as simulated.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Optional

import numpy as np

from xai_llm.backend import Backend, TopK, make_topk
from xai_llm.segment import segment


class SimulatedBackend(Backend):
    name = "simulated"

    def __init__(
        self,
        base_logits: Dict[str, float],
        effects: Dict[str, Dict[str, float]],
        noise_sd: float = 0.0,
        n_filler: int = 60,
        top_n: int = 20,
        workers: int = 4,
    ):
        super().__init__(workers)
        self.base = dict(base_logits)
        self.effects = {w.lower(): d for w, d in effects.items()}
        self.noise_sd = noise_sd
        self.top_n = top_n
        self.filler = {f"<f{i}>": -7.0 - 0.1 * i for i in range(n_filler)}

    def _fetch(self, prompt: str) -> TopK:
        units, _ = segment(prompt)
        present = {u.text.lower() for u in units}
        logits = dict(self.base)
        for w in present:
            for tok, delta in self.effects.get(w, {}).items():
                logits[tok] = logits.get(tok, 0.0) + delta
        logits.update(self.filler)
        toks = list(logits)
        arr = np.array([logits[t] for t in toks], dtype=float)
        if self.noise_sd > 0:
            seed = int(hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8], 16)
            arr = arr + np.random.default_rng(seed).normal(0, self.noise_sd, arr.size)
        m = arr.max()
        logp = arr - (m + np.log(np.exp(arr - m).sum()))
        order = np.argsort(-logp)[: self.top_n]
        return make_topk([(toks[i], float(logp[i]), None) for i in order])


def demo_backend(noise_sd: float = 0.0) -> SimulatedBackend:
    """Capital-city toy model: 'France' pushes ' Paris', 'capital' pushes any city."""
    cities = [" Paris", " London", " Berlin", " Rome", " Madrid"]
    base = {c: 0.0 for c in cities}
    base.update({" the": 1.6, " a": 1.1, " located": 0.4, " known": 0.3})
    effects = {
        "capital": {**{c: 1.2 for c in cities}, " the": -0.9, " a": -0.7},
        "france": {" Paris": 4.0, " London": -0.4, " Berlin": -0.4},
        "french": {" Paris": 1.5},
        "england": {" London": 4.0},
        "germany": {" Berlin": 4.0},
        "italy": {" Rome": 4.0},
        "spain": {" Madrid": 4.0},
    }
    return SimulatedBackend(base, effects, noise_sd=noise_sd)
