import math

import numpy as np
import pytest

from scoringbench.calibrate import QuantileCalibrator
from scoringbench.parse import score_distribution


def test_distribution_from_logprobs():
    reply = {"content": "2", "first_token_logprobs": {"2": math.log(0.6), "1": math.log(0.3), "3": math.log(0.08),
                                                      "<|im_end|>": math.log(0.02)}}
    d = score_distribution(reply, 1, 6)
    assert d["source"] == "logprobs"
    assert d["mode"] == 2
    assert d["coverage"] == pytest.approx(0.98)
    assert d["expected"] == pytest.approx((0.3 * 1 + 0.6 * 2 + 0.08 * 3) / 0.98)


def test_text_fallback_and_unparsed():
    assert score_distribution({"content": "Score: 4"}, 1, 6)["expected"] == 4
    assert score_distribution({"content": "12", "first_token_logprobs": {"1": 0.0}}, 0, 30)["expected"] == 12
    assert score_distribution({"content": "great essay"}, 1, 6)["source"] == "unparsed"


def test_quantile_calibration_undoes_severity():
    rng = np.random.default_rng(0)
    human = rng.integers(1, 7, 400).astype(float)
    model = 0.5 * human + 0.3  # harsh and compressed, same order
    cal = QuantileCalibrator().fit(model[:100], human[:100])
    out = cal.transform(model[100:])
    assert abs(out.mean() - human[100:].mean()) < 0.3
    assert np.std(out) == pytest.approx(np.std(human[100:]), rel=0.15)


def test_anchor_split_matches_population_distribution():
    import pandas as pd
    from scoringbench.data import split_anchors
    h = [1] * 2 + [2] * 10 + [3] * 40 + [4] * 200 + [5] * 120 + [6] * 28
    df = pd.DataFrame({"h1": h})
    anchors, rest = split_anchors(df, 50, seed=1)
    assert len(anchors) == 50 and len(rest) == 350
    assert set(anchors.index).isdisjoint(rest.index)
    share = anchors["h1"].value_counts(normalize=True).reindex(range(1, 7), fill_value=0)
    pop = df["h1"].value_counts(normalize=True).reindex(range(1, 7), fill_value=0)
    assert (share - pop).abs().max() < 0.03


def test_context_appears_in_prompt():
    from scoringbench.config import StrategyConfig, TaskConfig
    from scoringbench.strategies import Vanilla
    v = Vanilla(StrategyConfig("vanilla"), TaskConfig(rubric="r", context="Grade 8 students"), (1, 6))
    msg = v.messages({"text": "essay"})[-1]["content"]
    assert "Grade 8 students" in msg and msg.index("Grade 8") < msg.index("Essay:")
