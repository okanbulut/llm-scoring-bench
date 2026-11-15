import numpy as np
import pytest

from scoringbench.metrics import agreement, prmse, qwk, smd


def test_qwk_perfect_and_symmetric():
    a = [1, 2, 3, 4, 5, 6]
    assert qwk(a, a) == pytest.approx(1.0)
    b = [1, 3, 3, 4, 6, 5]
    assert qwk(a, b) == pytest.approx(qwk(b, a))


def test_qwk_known_value():
    # hand-computed: obs off-diagonal at (0,1) and (1,0); k=3
    a = [0, 0, 1, 1, 2, 2]
    b = [0, 1, 0, 1, 2, 2]
    # weighted observed disagreement = 2 * (1/4) = 0.5; expected uses marginals (2,2,2)x(2,2,2)/6
    exp = sum(((i - j) ** 2 / 4) * (2 * 2 / 6) for i in range(3) for j in range(3))
    assert qwk(a, b) == pytest.approx(1 - 0.5 / exp)


def test_qwk_matches_sklearn_if_available():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    a = rng.integers(1, 7, 300)
    b = np.clip(a + rng.integers(-1, 2, 300), 1, 6)
    assert qwk(a, b) == pytest.approx(sk.cohen_kappa_score(a, b, weights="quadratic"))


def test_prmse_recovers_known_value():
    rng = np.random.default_rng(1)
    n = 40000
    t = rng.normal(0, 1, n)
    h1, h2 = t + rng.normal(0, 0.6, n), t + rng.normal(0, 0.6, n)
    system = t + rng.normal(0, 0.3, n)
    # true PRMSE = 1 - 0.09 / 1
    assert prmse(system, h1, h2) == pytest.approx(0.91, abs=0.015)
    # an independent third human rater (error SD .6) has true PRMSE 1 - .36 = .64
    h3 = t + rng.normal(0, 0.6, n)
    assert prmse(h3, h1, h2) == pytest.approx(0.64, abs=0.02)


def test_agreement_and_smd():
    assert agreement([1, 2, 3], [1, 3, 5]) == {"exact": pytest.approx(1 / 3), "adjacent": pytest.approx(2 / 3)}
    assert smd([2, 3, 4], [2, 3, 4]) == pytest.approx(0.0)
    assert smd([3, 4, 5], [2, 3, 4]) > 0
