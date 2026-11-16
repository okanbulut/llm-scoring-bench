import numpy as np

from scoringbench.fusion import fit_fusion
from test_strategies import _study


def _grm_probs(theta, a, b, rng, noise=0.0):
    """Soft score distributions from a graded response model (b: K-1 increasing thresholds)."""
    t = theta + rng.normal(0, noise, len(theta))
    s = 1 / (1 + np.exp(-(a * t[:, None] - b[None, :])))
    S = np.hstack([np.ones((len(t), 1)), s, np.zeros((len(t), 1))])
    return S[:, :-1] - S[:, 1:]


def test_fusion_recovers_latent_better_than_one_indicator():
    rng = np.random.default_rng(2)
    n = 300
    theta = rng.normal(0, 1, n)
    harsh = _grm_probs(theta, 1.5, np.array([-1.5, -0.5, 0.5, 1.5, 2.5]), rng, noise=0.8)  # severe, noisy
    trait1 = _grm_probs(theta, 1.2, np.array([-2, -1, 0, 1, 2]), rng, noise=0.8)
    trait2 = _grm_probs(theta, 1.0, np.array([-2.5, -1, 0, 1, 1.5]), rng, noise=0.8)
    pairs = []
    for _ in range(900):
        i, j = rng.choice(n, 2, replace=False)
        d = theta[i] - theta[j] + rng.normal(0, 0.8)
        pairs.append((i, j, float(1 / (1 + np.exp(-1.5 * d)))))
    fit = fit_fusion({"holistic": harsh, "t1": trait1, "t2": trait2}, pairs, n)
    r_fused = np.corrcoef(fit["theta"], theta)[0, 1]
    r_single = np.corrcoef((harsh * np.arange(6)).sum(1), theta)[0, 1]
    assert fit["params"]["converged"]
    assert r_fused > r_single + 0.05
    assert r_fused > 0.85
    # the harsh indicator's thresholds sit above the lenient one's
    assert fit["params"]["holistic"]["b"][2] > fit["params"]["t2"]["b"][2]
    # SEs are positive and smaller than the prior SD
    assert (fit["se"] > 0).all() and (fit["se"] < 1).all()


def test_fusion_scores_only_and_missing_rows():
    rng = np.random.default_rng(3)
    theta = rng.normal(0, 1, 100)
    p = _grm_probs(theta, 1.5, np.array([-2, -1, 0, 1, 2]), rng, noise=0.3)
    p[5] = np.nan  # unparsed reply
    fit = fit_fusion({"holistic": p}, [], 100)
    ok = np.arange(100) != 5
    assert np.corrcoef(fit["theta"][ok], theta[ok])[0, 1] > 0.85
    assert abs(fit["theta"][5]) < 0.2 and fit["se"][5] > 0.9  # no data: prior mean, prior SD


def test_fusion_strategy_end_to_end(tmp_path):
    res = _study(tmp_path, "[pairwise, {name: fusion, k: 6}, {name: fusion, components: [vanilla, mts], label: fusion-abs}]")
    m = res["metrics"].set_index("strategy")
    assert m.loc["fusion", "r"] > 0.75
    assert m.loc["fusion", "calls_per_essay"] > m.loc["fusion-abs", "calls_per_essay"]
    preds = res["predictions"]
    assert preds.loc[preds.strategy == "fusion", "se"].gt(0).all()
    import json
    meta = json.loads((tmp_path / "o" / "run.json").read_text())
    assert "fake | fusion" in meta["fusion_params"]
