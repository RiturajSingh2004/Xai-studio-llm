"""Turn a next-token distribution into one number per experiment.

Single mode:   score = log P(target)
Contrast mode: score = log P(target) - log P(foil)   ("why A rather than B?")

COVERAGE. The server only reports the top-N tokens. If the target (or foil) is
not among them we do not know its probability, only that it is <= the smallest
reported one. We then use that bound as the value (this can only UNDER-state a
drop, never invent one) and mark the observation as not covered. Coverage = the
share of experiments where everything needed was observed; it is reported
everywhere and the sampler adapts to keep it high.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from xai_llm.backend import TopK


@dataclass(frozen=True)
class TokenInfo:
    key: str
    token: str


@dataclass
class Observation:
    score: float      # log P(target)  or  log-odds(target vs foil)
    prob: float       # P(target)      or  P(target | target or foil)
    covered: bool     # True if everything needed was actually observed
    top1_key: str
    top1_token: str
    top1_prob: float


def _sigmoid(x: float) -> float:
    x = max(min(x, 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-x))


def observe(tk: TopK, target_key: str, foil_key: Optional[str] = None) -> Observation:
    d = tk.as_dict()
    floor = tk.floor
    lp_t = d.get(target_key)

    if foil_key is None:
        covered = lp_t is not None
        score = lp_t if covered else floor
        prob = math.exp(score)
    else:
        lp_f = d.get(foil_key)
        if lp_t is not None and lp_f is not None:
            score, covered = lp_t - lp_f, True
        elif lp_t is None and lp_f is not None:
            score, covered = floor - lp_f, False   # upper bound
        elif lp_t is not None and lp_f is None:
            score, covered = lp_t - floor, False   # lower bound
        else:
            score, covered = 0.0, False
        prob = _sigmoid(score)

    return Observation(
        score=float(score),
        prob=float(prob),
        covered=bool(covered),
        top1_key=tk.keys[0],
        top1_token=tk.tokens[0],
        top1_prob=float(math.exp(tk.logprobs[0])),
    )
