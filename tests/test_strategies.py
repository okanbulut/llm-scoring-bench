import numpy as np
import pandas as pd
import pytest

from scoringbench.config import load_config
from scoringbench.runner import run_study
from scoringbench.strategies import ab_probability, bradley_terry


def test_bradley_terry_recovers_order():
    rng = np.random.default_rng(0)
    true = np.linspace(-2, 2, 30)
    pairs = []
    for _ in range(300):
        i, j = rng.choice(30, 2, replace=False)
        pairs.append((i, j, float(1 / (1 + np.exp(-(true[i] - true[j]))))))
    theta = bradley_terry(30, pairs)
    assert np.corrcoef(theta, true)[0, 1] > 0.98


def test_ab_probability():
    import math
    assert ab_probability({"first_token_logprobs": {"A": math.log(0.7), " B": math.log(0.3)}}) == pytest.approx(0.7)
    assert ab_probability({"content": "B"}) == 0.0
    assert ab_probability({"content": "neither"}) == 0.5


def _study(tmp_path, strategies: str, n=80):
    rng = np.random.default_rng(5)
    true = rng.integers(1, 7, n)
    h1 = np.clip(true + rng.integers(-1, 2, n), 1, 6)
    h2 = np.clip(true + rng.integers(-1, 2, n), 1, 6)
    pd.DataFrame({"id": range(n), "essay": [f"E{i} [true={t}]" for i, t in enumerate(true)], "r1": h1, "r2": h2}) \
        .to_csv(tmp_path / "d.csv", index=False)
    (tmp_path / "s.yaml").write_text(f"""
name: strat
data: {{loader: table, path: {tmp_path / 'd.csv'}, score_range: [1, 6], columns: {{id: id, text: essay, human: [r1, r2]}}}}
task: {{rubric: "r"}}
models: [{{name: fake, backend: fake}}]
strategies: {strategies}
cache: {tmp_path / 'c.sqlite'}
""")
    return run_study(load_config(tmp_path / "s.yaml"), tmp_path / "o", progress=False)


def test_mts_and_pairwise_run(tmp_path):
    res = _study(tmp_path, "[vanilla, mts, {name: pairwise, k: 8}]")
    m = res["metrics"].set_index("strategy")
    assert m.loc["mts", "calls_per_essay"] == 3
    # min-max scaling spans the full range, so MTS is far less harsh than vanilla
    assert abs(m.loc["mts", "mean_diff"]) < abs(m.loc["vanilla", "mean_diff"])
    # both-order averaging removes the fake rater's pull towards "A", so pairwise ranks well
    assert m.loc["pairwise", "r"] > 0.75
    assert 3.5 <= m.loc["pairwise", "calls_per_essay"] <= 8.5
    preds = res["predictions"]
    assert preds.loc[preds.strategy == "mts", "traits"].notna().all()


def test_pairwise_with_calibration(tmp_path):
    m = _study(tmp_path, "[{name: pairwise, k: 6, calibrate: quantile, anchors: 20}]")["metrics"].iloc[0]
    assert m["n"] == 60 and abs(m["mean_diff"]) < 0.5 and m["qwk"] > 0.6


def test_anchored_ladder(tmp_path):
    res = _study(tmp_path, "[{name: pairwise, k: 6, calibrate: quantile, anchors: 20},"
                           " {name: anchored, rounds: 3, calibrate: quantile, anchors: 20},"
                           " {name: anchored, rounds: 3, peer_k: 4, calibrate: quantile, anchors: 20, label: anch-peer}]",
                 n=120)
    m = res["metrics"].set_index("strategy")
    a = m.loc["anchored+quantile20"]
    assert a["r"] > 0.75
    # about rounds * 2 calls per essay plus the anchor chain; peers add more
    assert 4 <= a["calls_per_essay"] <= 9
    assert m.loc["anch-peer", "calls_per_essay"] > a["calls_per_essay"]


def test_anchored_needs_anchors(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="needs anchors"):
        _study(tmp_path, "[{name: anchored}]")


def test_hybrid_embeddings_and_pairwise(tmp_path):
    import json
    res = _study(tmp_path, "[{name: hybrid, calibrate: quantile, anchors: 30, base: none, label: emb-only},"
                           " {name: hybrid, calibrate: quantile, anchors: 30, base_options: {k: 6}, label: hyb}]",
                 n=150)
    m = res["metrics"].set_index("strategy")
    assert m.loc["emb-only", "r"] > 0.5 and m.loc["emb-only", "calls_per_essay"] == 0
    assert m.loc["hyb", "r"] > 0.75 and m.loc["hyb", "calls_per_essay"] > 0
    params = json.loads((tmp_path / "o" / "run.json").read_text())["fusion_params"]
    hyb = params["fake | hyb"]
    assert 0 <= hyb["weight_embedding"] <= 1 and "0.0" in hyb["weight_table"]
    # embeddings were cached: a rerun gives identical predictions
    again = _study(tmp_path, "[{name: hybrid, calibrate: quantile, anchors: 30, base: none, label: emb-only}]", n=150)
    a = res["predictions"].query("strategy == 'emb-only'")["score_cont"].to_numpy()
    b = again["predictions"].query("strategy == 'emb-only'")["score_cont"].to_numpy()
    assert np.allclose(a, b)


def test_ridge_loo_matches_refits():
    from scoringbench.embeddings import ridge_loo
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 50)); y = X[:, 0] + rng.normal(0, .5, 30)
    r = ridge_loo(X, y, (10.0,))
    for i in (0, 7, 29):
        keep = np.arange(30) != i
        sub = ridge_loo(X[keep], y[keep], (10.0,))
        assert abs(sub["predict"](X[i:i + 1])[0] - r["oof"][i]) < 1e-8
