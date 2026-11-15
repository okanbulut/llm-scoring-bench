"""Turn a model reply into a score distribution over the rater scale."""

from __future__ import annotations

import math
import re

import numpy as np


def score_distribution(reply: dict, lo: int, hi: int, min_coverage: float = 0.5) -> dict:
    """Distribution over lo..hi from first-token log-probabilities when the scale is single-digit.

    Falls back to the integer in the reply text (a one-hot distribution) when log-probs are missing,
    the scale has two-digit scores, or the digits carry less than `min_coverage` of the probability.
    """
    levels = np.arange(lo, hi + 1)
    lp = reply.get("first_token_logprobs")
    if lp and hi <= 9:
        mass = np.zeros(len(levels))
        for tok, logp in lp.items():
            t = tok.strip()
            if t.isdigit() and lo <= int(t) <= hi:
                mass[int(t) - lo] += math.exp(logp)
        cov = float(mass.sum())
        if cov >= min_coverage:
            p = mass / cov
            return _summary(levels, p, cov, "logprobs")
    hit = re.search(r"-?\d+", reply.get("content") or "")
    if hit and lo <= int(hit.group()) <= hi:
        p = (levels == int(hit.group())).astype(float)
        return _summary(levels, p, 0.0, "text")
    return {"probs": [float("nan")] * len(levels), "expected": float("nan"), "mode": float("nan"),
            "coverage": 0.0, "source": "unparsed"}


def _summary(levels: np.ndarray, p: np.ndarray, cov: float, source: str) -> dict:
    return {"probs": p.round(6).tolist(), "expected": float((levels * p).sum()), "mode": float(levels[int(p.argmax())]),
            "coverage": round(cov, 6), "source": source}
