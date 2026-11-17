"""The attribution engine.

Question answered:  "Which words of my prompt made the model output THIS token?"

Because Ollama exposes no gradients, we run controlled experiments on the model
and test the results statistically:

 1. Baseline          the unmodified prompt (+ a few repeats to measure noise).
 2. Remove-one        drop each word alone  -> exact, easy-to-read counterfactuals.
 3. Random removals   drop random subsets of words; regress the target score on
                      which words were kept -> the average effect of each word
                      (a Shapley-style estimate) with bootstrap confidence
                      intervals, p-values and false-discovery-rate control.
                      The fraction of words dropped adapts so the target stays
                      observable in the server's top-N list (coverage).
 4. Faithfulness      remove the words the explanation ranks highest / lowest
                      and compare with removing random words (permutation test).
 5. Trust checks      coverage, cross-validated R^2 of the linear explanation,
                      agreement between remove-one and regression (Spearman).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from xai_llm import stats
from xai_llm.backend import Backend
from xai_llm.config import MAX_UNITS
from xai_llm.scoring import Observation, TokenInfo, observe
from xai_llm.segment import render, segment


@dataclass
class ExplainConfig:
    n_samples: int = 300      # random word-removal experiments (minimum; scales with prompt length)
    n_random: int = 30        # random controls per k in the faithfulness test
    n_noise: int = 5          # repeats of the baseline to estimate run-to-run noise
    n_boot: int = 500         # bootstrap resamples
    ci: float = 0.95
    fdr: float = 0.10         # false discovery rate for "reliable effect"
    min_effect: float = 0.10  # ignore effects smaller than this many nats (~10%)
    drop_frac: float = 0.35   # starting max share of words removed per experiment
    min_coverage: float = 0.95
    max_k: int = 6            # max number of words removed in the faithfulness test
    seed: int = 0


@dataclass
class Explanation:
    prompt: str
    units: List[str]
    backend_name: str
    model: str
    mode: str
    target: TokenInfo
    foil: Optional[TokenInfo]
    cfg: ExplainConfig
    base_obs: Observation
    baseline_top: List[Tuple[str, float]]
    noise_sd: float
    beta: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    p: np.ndarray
    q: np.ndarray
    sig: np.ndarray
    intercept: float
    r2_cv: float
    y_cv: np.ndarray
    yhat_cv: np.ndarray
    loo_effect: np.ndarray
    loo_obs: List[Observation]
    r_methods: float
    censor_rho: float
    coverage: Dict[str, float]
    drop_frac_used: float
    n_experiments: int
    deletion: Dict[str, object]
    n_requests: int
    n_cache_hits: int
    seconds: float
    warnings: List[str] = field(default_factory=list)

    @property
    def contrast(self) -> bool:
        return self.foil is not None


def baseline_top(backend: Backend, prompt: str, k: int = 10) -> List[Tuple[TokenInfo, float]]:
    """The model's top-k next tokens for the unmodified prompt (for choosing a target)."""
    tk = backend.query(prompt)
    return [
        (TokenInfo(key, tok), float(np.exp(lp)))
        for key, tok, lp in zip(tk.keys[:k], tk.tokens[:k], tk.logprobs[:k])
    ]


class _Tracker:
    def __init__(self, total: int, cb):
        self.total, self.done, self.cb = max(1, total), 0, cb

    def tick(self, k: int, label: str):
        self.done = min(self.total, self.done + k)
        if self.cb:
            self.cb(self.done / self.total, label)


def _sample_masks(rng, m: int, n: int, frac: float) -> np.ndarray:
    kmax = min(max(1, int(round(frac * n))), n - 1)
    ks = rng.integers(1, kmax + 1, size=m)
    masks = np.ones((m, n), dtype=bool)
    for i, k in enumerate(ks):
        masks[i, rng.choice(n, int(k), replace=False)] = False
    return masks


def explain(
    backend: Backend,
    prompt: str,
    target: Optional[TokenInfo] = None,
    foil: Optional[TokenInfo] = None,
    cfg: Optional[ExplainConfig] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Explanation:
    cfg = cfg or ExplainConfig()
    t0 = time.perf_counter()
    req0, hit0 = backend.requests, backend.cache_hits

    units, trailing = segment(prompt)
    n = len(units)
    if n < 2:
        raise ValueError("Write at least two words: with one word there is nothing to compare.")
    if n > MAX_UNITS:
        raise ValueError(f"The prompt has {n} words/punctuation marks; the limit is {MAX_UNITS}.")
    rng = np.random.default_rng(cfg.seed)

    base_tk = backend.query(prompt)
    base_dict = base_tk.as_dict()
    tgt = target or TokenInfo(base_tk.keys[0], base_tk.tokens[0])
    if tgt.key not in base_dict:
        raise ValueError("The chosen output token is not among the model's top predictions for this prompt.")
    if foil is not None and foil.key == tgt.key:
        foil = None
    if foil is not None and foil.key not in base_dict:
        raise ValueError("The comparison token is not among the model's top predictions for this prompt.")
    foil_key = foil.key if foil else None

    def obs_of(tk):
        return observe(tk, tgt.key, foil_key)

    base_obs = obs_of(base_tk)

    K = min(cfg.max_k, max(1, n // 2))   # top-K words are tested against K words from OUTSIDE them
    m_main = max(cfg.n_samples, 8 * (n + 1))
    total = 1 + cfg.n_noise + n + m_main + K * (max(cfg.n_random, 39) + 2)
    tracker = _Tracker(total, progress)
    tracker.tick(1, "Reading the model's prediction")

    def run(masks, label):
        prompts = [render(units, m, trailing) for m in masks]
        tks = backend.query_many(prompts, on_progress=lambda k: tracker.tick(k, label))
        return [obs_of(tk) for tk in tks]

    # 1. noise floor ---------------------------------------------------------
    noise = [obs_of(backend.query(prompt, use_cache=False)).score for _ in range(cfg.n_noise)]
    tracker.tick(cfg.n_noise, "Measuring run-to-run noise")
    noise_sd = float(np.std(noise, ddof=1)) if len(noise) > 1 else 0.0

    # 2. remove-one counterfactuals --------------------------------------------
    loo_masks = ~np.eye(n, dtype=bool)
    loo_obs = run(loo_masks, "Removing each word on its own")
    loo_effect = np.array([base_obs.score - o.score for o in loo_obs])

    # 3. random removals, adaptive so the target stays observable -------------
    frac = cfg.drop_frac
    pilot_n = min(max(30, 2 * n), m_main)
    X_parts = [np.ones((1, n), dtype=bool), loo_masks]
    obs_all: List[Observation] = [base_obs] + list(loo_obs)
    n_pilot_rows = 0
    for _ in range(4):
        masks = _sample_masks(rng, pilot_n, n, frac)
        ob = run(masks, "Random word removals (pilot)")
        X_parts.append(masks)
        obs_all += ob
        n_pilot_rows += pilot_n
        if np.mean([o.covered for o in ob]) >= cfg.min_coverage or frac <= 0.12:
            break
        frac = max(0.1, frac * 0.6)
    remaining = max(0, m_main - n_pilot_rows)
    if remaining:
        masks = _sample_masks(rng, remaining, n, frac)
        ob = run(masks, "Random word removals")
        X_parts.append(masks)
        obs_all += ob

    X = np.vstack(X_parts).astype(float)
    y = np.array([o.score for o in obs_all])
    covered = np.array([o.covered for o in obs_all])

    fit = stats.fit_effects(X, y, n_boot=cfg.n_boot, ci=cfg.ci, rng=rng)
    q = stats.benjamini_hochberg(fit.p)
    r2_cv, yhat = stats.cross_val_r2(X, y, rng=rng)
    min_eff = max(cfg.min_effect, 3.0 * noise_sd)
    sig = (q <= cfg.fdr) & (np.abs(fit.beta) >= min_eff) & ((fit.lo > 0) | (fit.hi < 0))
    # Agreement of the two estimates in their common unit (nats). Pearson, not rank
    # correlation: ranks among words with ~zero effect are pure noise.
    rho = stats.pearson(fit.beta, loo_effect)

    censor_rho = float("nan")
    if not covered.all() and covered.sum() >= 3 * (n + 1):
        censor_rho = stats.pearson(fit.beta, stats.ols(X[covered], y[covered])[1:])

    # 4. faithfulness test -----------------------------------------------------
    # A permutation p-value cannot be smaller than 1/(R+1); R >= 39 lets it reach 0.025,
    # so the test is able to pass at the 0.05 level.
    R = max(cfg.n_random, 39)
    order_top = np.argsort(-fit.beta, kind="stable")
    order_low = np.argsort(np.abs(fit.beta), kind="stable")
    del_masks = []
    for order in (order_top, order_low):
        for k in range(1, K + 1):
            mk = np.ones(n, dtype=bool)
            mk[order[:k]] = False
            del_masks.append(mk)
    # Random control = R random orderings of the words the explanation did NOT put in
    # its top-K. Question tested: "do the words it ranks highest matter more than the
    # words it leaves out?"  (Drawing controls from ALL words would let a random draw
    # pick the decisive word by luck, capping the best possible p-value at about 1/n
    # for a one-word explanation; the removal experiments are fresh, not the data the
    # ranking was fitted on, so this comparison is not circular.)
    outside = np.setdiff1d(np.arange(n), order_top[:K])
    for _ in range(R):
        perm = rng.permutation(outside)
        for k in range(1, K + 1):
            mk = np.ones(n, dtype=bool)
            mk[perm[:k]] = False
            del_masks.append(mk)
    del_obs = run(np.array(del_masks), "Testing the explanation by removing words")
    top_obs, low_obs = del_obs[:K], del_obs[K : 2 * K]
    rand = np.array([o.prob for o in del_obs[2 * K :]]).reshape(R, K).T  # (K, R)
    top_p = np.array([o.prob for o in top_obs])
    perm_p = np.array([(1 + np.sum(rand[i] <= top_p[i])) / (R + 1) for i in range(K)])
    # Whole-curve test: is the average probability over k=1..K lower when removing the
    # explanation's top words than when removing random outside words?
    # (single pre-specified statistic, so no cherry-picking over k)
    top_auc, rand_auc = float(top_p.mean()), rand.mean(axis=0)
    p_auc = float((1 + np.sum(rand_auc <= top_auc)) / (R + 1))
    deletion = {
        "ks": list(range(1, K + 1)),
        "top_prob": top_p.tolist(),
        "top_covered": [o.covered for o in top_obs],
        "low_prob": [o.prob for o in low_obs],
        "low_covered": [o.covered for o in low_obs],
        "rand_lo": np.quantile(rand, 0.05, axis=1).tolist(),
        "rand_med": np.median(rand, axis=1).tolist(),
        "rand_hi": np.quantile(rand, 0.95, axis=1).tolist(),
        "p_values": perm_p.tolist(),
        "p_auc": p_auc,
        "top_auc": top_auc,
        "rand_auc_med": float(np.median(rand_auc)),
        "k_headline": int(np.clip(round(0.25 * n), 1, K)),
        "top_words": [units[i].text for i in order_top[:K]],
        "low_words": [units[i].text for i in order_low[:K]],
        "n_random": R,
    }

    coverage = {
        "remove-one": float(np.mean([o.covered for o in loo_obs])),
        "random removals": float(covered[1 + n :].mean()) if len(covered) > 1 + n else 1.0,
        "word-removal test": float(np.mean([o.covered for o in del_obs])),
    }
    coverage["overall"] = float(np.mean(list(covered) + [o.covered for o in del_obs]))

    warnings: List[str] = []
    if coverage["overall"] < cfg.min_coverage:
        warnings.append(
            f"Coverage is {coverage['overall']:.0%}: in some experiments the chosen token fell out of the "
            "server's top predictions, so its probability was only bounded. Try the 'compare against' option "
            "(it stays observable more often) or a token with a higher probability."
        )
    if not np.isnan(r2_cv) and r2_cv < 0.5:
        warnings.append(
            f"The word-by-word explanation only reproduces {max(r2_cv, 0):.0%} of the model's behaviour on held-out "
            "experiments: words interact strongly here. Rely on the remove-one table and the word-removal test."
        )
    if not np.isnan(rho) and rho < 0.5:
        warnings.append(
            f"Remove-one and the random-removal estimate disagree (correlation r = {rho:.2f}); "
            "the words probably overlap in what they contribute."
        )
    if noise_sd > 0.05:
        warnings.append(f"The model's answer fluctuates between identical runs (SD {noise_sd:.3f} nats); small effects are not reliable.")

    return Explanation(
        prompt=prompt,
        units=[u.text for u in units],
        backend_name=backend.name,
        model=getattr(backend, "model", backend.name),
        mode=getattr(backend, "mode", "n/a"),
        target=tgt,
        foil=foil,
        cfg=cfg,
        base_obs=base_obs,
        baseline_top=[(t, float(np.exp(lp))) for t, lp in zip(base_tk.tokens[:10], base_tk.logprobs[:10])],
        noise_sd=noise_sd,
        beta=fit.beta,
        ci_lo=fit.lo,
        ci_hi=fit.hi,
        p=fit.p,
        q=q,
        sig=sig,
        intercept=fit.intercept,
        r2_cv=float(r2_cv),
        y_cv=y,
        yhat_cv=yhat,
        loo_effect=loo_effect,
        loo_obs=loo_obs,
        r_methods=float(rho),
        censor_rho=float(censor_rho),
        coverage=coverage,
        drop_frac_used=float(frac),
        n_experiments=int(len(y) + len(del_obs)),
        deletion=deletion,
        n_requests=backend.requests - req0,
        n_cache_hits=backend.cache_hits - hit0,
        seconds=time.perf_counter() - t0,
        warnings=warnings,
    )
