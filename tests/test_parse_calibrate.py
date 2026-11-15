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
