"""Plain-English summary of an Explanation (what a reader should take away)."""
from __future__ import annotations

import math
from typing import List

import numpy as np

from xai_llm.explain import Explanation


def tok(t: str) -> str:
    """Quoted token with visible whitespace: ' Paris' -> "␣Paris" is avoided; quotes show the space."""
    return '"' + t.replace("\n", "⏎") + '"'


def mult(beta: float) -> str:
    m = math.exp(max(min(beta, 12.0), -12.0))
    if m >= 100:
        return f"×{m:.0f}"
    if m >= 10:
        return f"×{m:.1f}"
    if m >= 1:
        return f"×{m:.2f}"
    if m >= 0.1:
        return f"×{m:.2f}"
    return f"×{m:.3f}"


def prob_name(e: Explanation) -> str:
    if e.contrast:
        return f"the chance of {tok(e.target.token)} over {tok(e.foil.token)}"
    return f"P({tok(e.target.token)})"


def _words(e: Explanation, idx) -> str:
    return ", ".join(f"{e.units[i]} ({mult(e.beta[i])})" for i in idx)


def summary(e: Explanation) -> List[str]:
    n = len(e.units)
    p = e.base_obs.prob
    lines: List[str] = []

    if e.contrast:
        lines.append(
            f"The model prefers {tok(e.target.token)} over {tok(e.foil.token)}: "
            f"{p:.0%} vs {1 - p:.0%} when only these two are compared."
        )
    else:
        lines.append(f"The model gives {tok(e.target.token)} a probability of {p:.0%} as the next token.")

    pos = [i for i in np.argsort(-e.beta) if e.sig[i] and e.beta[i] > 0]
    neg = [i for i in np.argsort(e.beta) if e.sig[i] and e.beta[i] < 0]
    what = "the odds of" if e.contrast else "the probability of"
    if pos:
        lines.append(f"Words that push the model toward {tok(e.target.token)} (they multiply {what} it): {_words(e, pos)}.")
    else:
        lines.append(f"No single word has a reliable effect that pushes the model toward {tok(e.target.token)}.")
    if neg:
        lines.append(f"Words that push it away: {_words(e, neg)}.")

    n_sig = int(e.sig.sum())
    min_eff = max(e.cfg.min_effect, 3 * e.noise_sd)
    lines.append(
        f"{n_sig} of {n} words have a statistically reliable effect "
        f"({e.cfg.ci:.0%} bootstrap interval excludes zero, false-discovery rate ≤ {e.cfg.fdr:.0%}, "
        f"effect at least {mult(min_eff)}). The others show no reliable effect on their own."
    )

    d = e.deletion
    kh = d["k_headline"]
    top_p, med = d["top_prob"][kh - 1], d["rand_med"][kh - 1]
    verdict = (
        "the top-ranked words matter clearly more than the words left out"
        if d["p_auc"] < 0.05
        else "the top-ranked words do not matter clearly more than the words left out"
    )
    lines.append(
        f"Word-removal test: removing the {kh} word{'s' if kh > 1 else ''} ranked most important leaves "
        f"{top_p:.0%} (random words from outside the top {d['ks'][-1]}: median {med:.0%}). "
        f"Over 1–{d['ks'][-1]} words, {verdict} (permutation p = {d['p_auc']:.3f}, {d['n_random']} random draws)."
    )
    if n < 6:
        lines.append("Note: with so few words the test has little power (few random alternatives exist); add context to the prompt for a stronger test.")

    lines.append(
        f"Trust checks: coverage {e.coverage['overall']:.0%}; the word-by-word explanation reproduces "
        + (f"{max(e.r2_cv, 0):.0%}" if not math.isnan(e.r2_cv) else "n/a")
        + " of held-out behaviour; the two estimation methods correlate at r = "
        + (f"{e.r_methods:.2f}" if not math.isnan(e.r_methods) else "n/a")
        + "."
    )
    for w in e.warnings:
        lines.append("⚠ " + w)
    return lines
