"""End-to-end tests on a simulated model whose true word effects are known."""
import numpy as np
import pytest

from xai_llm.explain import ExplainConfig, baseline_top, explain
from xai_llm.simulate import SimulatedBackend, demo_backend

PROMPT = "Please tell me quickly what the capital of France is, today"
CFG = ExplainConfig(n_samples=300, n_random=30, n_boot=300)


@pytest.fixture(scope="module")
def result():
    return explain(demo_backend(), PROMPT, cfg=CFG)


def sig_words(e):
    return {e.units[i] for i in np.flatnonzero(e.sig)}


def test_recovers_the_planted_words_and_nothing_else(result):
    assert sig_words(result) == {"capital", "France"}
    i = result.units.index("France")
    assert result.beta[i] > 1.5 and result.ci_lo[i] > 1.0   # planted +4 logit, diluted by normalisation
    assert 0 < result.beta[result.units.index("capital")] < 0.6


def test_trust_diagnostics_are_healthy(result):
    assert result.coverage["overall"] == 1.0
    assert result.r2_cv > 0.9
    assert result.r_methods > 0.9
    assert result.warnings == []


def test_faithfulness_test_passes_with_enough_words(result):
    d = result.deletion
    assert d["p_auc"] < 0.05
    assert d["top_prob"][0] < 0.3 < d["rand_med"][0]       # removing "France" first kills the prediction
    assert min(d["low_prob"]) > 0.9                        # removing the least important words does nothing


def test_remove_one_agrees_with_planted_truth(result):
    i = result.units.index("France")
    assert result.loo_obs[i].prob < 0.3 and result.loo_effect[i] > 1.5
    assert all(o.prob > 0.9 for j, o in enumerate(result.loo_obs) if result.units[j] not in ("France", "capital"))


def test_contrast_mode_cancels_the_shared_effect():
    be = demo_backend()
    top = {t.token: t for t, _ in baseline_top(be, PROMPT, 10)}
    e = explain(be, PROMPT, target=top[" Paris"], foil=top[" London"], cfg=CFG)
    # "capital" boosts every city equally, so it cannot explain Paris-vs-London; "France" can.
    assert sig_words(e) == {"France"}
    assert e.contrast and e.coverage["overall"] == 1.0


def test_survives_unmodelled_noise_with_few_false_positives():
    be = demo_backend(noise_sd=0.15)   # deterministic per-prompt jitter = interactions the model cannot explain
    e = explain(be, PROMPT, cfg=CFG)
    assert "France" in sig_words(e)
    assert len(sig_words(e) - {"France", "capital"}) <= 1   # FDR control: at most one slip among 10 fillers


def test_low_coverage_triggers_adaptive_sampling_and_a_warning():
    # " Paris" is only observable (top-3) while "France" is present.
    base = {"A": 0.0, "B": 0.1, "C": 0.2, "D": 0.3, "E": 0.4, " Paris": -0.2}
    be = SimulatedBackend(base, {"france": {" Paris": 5.0}}, top_n=3)
    prompt = "I really would like to know the capital of France"
    top = baseline_top(be, prompt, 3)
    e = explain(be, prompt, target=top[0][0], cfg=ExplainConfig(n_samples=150, n_random=10, n_boot=200))
    assert e.drop_frac_used < CFG.drop_frac                 # the sampler backed off
    assert e.coverage["overall"] < 1.0
    assert any("Coverage" in w for w in e.warnings)
    assert "France" in sig_words(e)                         # still finds the one word that matters


def test_faithfulness_test_can_always_reach_significance():
    # Even if a caller asks for few random draws, the p-value floor must stay below 0.05.
    e = explain(demo_backend(), PROMPT, cfg=ExplainConfig(n_samples=120, n_random=5, n_boot=100))
    assert e.deletion["n_random"] >= 39 and e.deletion["p_auc"] < 0.05


def test_input_validation():
    be = demo_backend()
    with pytest.raises(ValueError, match="at least two"):
        explain(be, "Paris", cfg=CFG)
    with pytest.raises(ValueError, match="empty|two"):
        explain(be, "   ", cfg=CFG)
    with pytest.raises(ValueError, match="top predictions"):
        from xai_llm.scoring import TokenInfo
        explain(be, PROMPT, target=TokenInfo("00", "nope"), cfg=CFG)


def test_two_word_prompt_does_not_crash():
    e = explain(demo_backend(), "France capital", cfg=ExplainConfig(n_samples=60, n_random=10, n_boot=100))
    assert len(e.units) == 2 and np.isfinite(e.beta).all()
