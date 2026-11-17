import numpy as np

from xai_llm import stats


def _design(rng, m=400, n=6):
    return (rng.random((m, n)) > 0.35).astype(float)


def test_ols_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = _design(rng)
    true = np.array([1.5, 0.0, -0.8, 0.0, 0.3, 0.0])
    y = 2.0 + X @ true + rng.normal(0, 0.05, len(X))
    b = stats.ols(X, y)
    assert abs(b[0] - 2.0) < 0.1
    assert np.allclose(b[1:], true, atol=0.1)


def test_bootstrap_ci_contains_truth_and_separates_null():
    rng = np.random.default_rng(1)
    X = _design(rng)
    true = np.array([1.5, 0.0, -0.8, 0.0, 0.3, 0.0])
    y = X @ true + rng.normal(0, 0.3, len(X))
    fit = stats.fit_effects(X, y, n_boot=400, rng=rng)
    assert np.all(fit.lo <= true + 1e-9) and np.all(true <= fit.hi + 1e-9)
    assert fit.p[0] < 1e-6 and fit.p[2] < 1e-6      # strong effects: not capped at the bootstrap floor
    assert fit.p[1] > 0.05 and fit.p[3] > 0.05      # nulls look null


def test_benjamini_hochberg_known_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    q = stats.benjamini_hochberg(p)
    assert np.all(q >= p - 1e-12) and np.all(q <= 1)
    assert q[0] < 0.05 and q[-1] > 0.2
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)  # monotone in p


def test_rank_and_correlations():
    assert np.allclose(stats.rankdata([3, 1, 2, 2]), [4, 1, 2.5, 2.5])
    a = np.arange(10.0)
    assert abs(stats.spearman(a, a**3) - 1) < 1e-9
    assert abs(stats.pearson(a, 2 * a + 1) - 1) < 1e-9
    assert np.isnan(stats.pearson(a, np.ones(10)))


def test_cross_val_r2_distinguishes_signal_from_noise():
    rng = np.random.default_rng(2)
    X = _design(rng, 300, 5)
    good = X @ np.array([1, -1, 0.5, 0, 0]) + rng.normal(0, 0.1, 300)
    noise = rng.normal(0, 1, 300)
    assert stats.cross_val_r2(X, good, rng=rng)[0] > 0.9
    assert stats.cross_val_r2(X, noise, rng=rng)[0] < 0.1
    assert np.isnan(stats.cross_val_r2(X, np.ones(300), rng=rng)[0])
