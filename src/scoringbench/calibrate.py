"""Map model scores onto the human score distribution using a few scored anchor essays."""

from __future__ import annotations

import numpy as np


class QuantileCalibrator:
    """Equipercentile linking: a model score's percentile rank among the anchors' model scores is mapped
    to the same percentile of the anchors' human scores. Fixes severity and scale compression; it keeps
    the model's ranking of essays unchanged."""

    def fit(self, model_scores, human_scores) -> "QuantileCalibrator":
        m = np.asarray(model_scores, float)
        h = np.asarray(human_scores, float)
        ok = ~(np.isnan(m) | np.isnan(h))
        if ok.sum() < 2:
            raise ValueError("Need at least two anchors with scores to calibrate")
        self.m_sorted = np.sort(m[ok])
        self.h_sorted = np.sort(h[ok])
        return self

    def transform(self, model_scores) -> np.ndarray:
        x = np.asarray(model_scores, float)
        n = len(self.m_sorted)
        grid = (np.arange(n) + 0.5) / n
        # mid-rank percentile of x among anchor model scores (ties share a rank), then human quantile
        lo = np.searchsorted(self.m_sorted, x, side="left")
        hi = np.searchsorted(self.m_sorted, x, side="right")
        pct = np.clip((lo + hi) / (2 * n), grid[0], grid[-1])
        out = np.interp(pct, grid, self.h_sorted)
        out[np.isnan(x)] = np.nan
        return out


CALIBRATORS = {"quantile": QuantileCalibrator}
