"""Joint latent-variable model for LLM judgments ("psychometric fusion").

Indicators
- Score indicators j (holistic score, each trait score): the LLM gives a probability vector p_ij over K
  ordered categories. A graded response model links it to the essay's latent quality theta_i:
      P(Y_ij >= k | theta_i) = logistic(a_j * theta_i - b_jk),  k = 2..K,  b_j2 < ... < b_jK.
  The LLM's distribution enters as soft data: sum_k p_ijk * log P(Y_ij = k | theta_i). Thresholds absorb each
  indicator's severity and scale compression; a_j says how strongly it reflects the common quality.
- Comparisons (i, j, q): q = P(essay i judged better), averaged over both orders. Bradley-Terry:
      q * log logistic(g * (theta_i - theta_j)) + (1 - q) * log logistic(-g * (theta_i - theta_j)).

Priors: theta_i ~ N(0, 1); log a_j, log g ~ N(0, 1); first threshold and log threshold gaps ~ N(0, 3^2).
Estimation: joint MAP with L-BFGS (analytic gradient). SE(theta_i): Laplace approximation from the diagonal
of the Hessian with the other parameters held at their estimates.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class _Layout:
    def __init__(self, n: int, ks: list[int], has_pairs: bool):
        self.n, self.ks = n, ks
        self.slices = []
        pos = n
        for k in ks:
            alpha, c, d = pos, pos + 1, slice(pos + 2, pos + 2 + (k - 2))
            self.slices.append((alpha, c, d))
            pos += 2 + (k - 2)
        self.g = pos if has_pairs else None
        self.size = pos + (1 if has_pairs else 0)


def _thresholds(x, c_idx, d_sl):
    gaps = np.exp(x[d_sl])
    return np.concatenate([[x[c_idx]], x[c_idx] + np.cumsum(gaps)])


def _score_part(theta, a, b, p, w):
    """Log-lik of one score indicator and its gradients. p: (n, K) soft data, w: (n,) 0/1 present."""
    K = p.shape[1]
    eta = a * theta[:, None] - b[None, :]  # (n, K-1) for k = 2..K
    s = _sig(eta)
    S = np.hstack([np.ones((len(theta), 1)), s, np.zeros((len(theta), 1))])  # S_1..S_{K+1}
    P = np.clip(S[:, :-1] - S[:, 1:], 1e-12, None)  # (n, K)
    ll = (w[:, None] * p * np.log(P)).sum()
    r = w[:, None] * p / P  # (n, K)
    deta = s * (1 - s) * (r[:, 1:] - r[:, :-1])  # d ll / d eta_k, k = 2..K
    g_theta = a * deta.sum(1)
    g_a = (theta[:, None] * deta).sum()
    g_b = -deta.sum(0)  # (K-1,)
    return ll, g_theta, g_a, g_b


def _objective(x, lay: _Layout, data, pairs, prior_sd_thr=3.0, only_scores=False):
    n = lay.n
    theta = x[:n]
    grad = np.zeros_like(x)
    ll = -0.5 * (theta ** 2).sum()
    grad[:n] -= theta
    for (alpha_i, c_i, d_sl), (p, w) in zip(lay.slices, data):
        a = np.exp(x[alpha_i])
        b = _thresholds(x, c_i, d_sl)
        l, gt, ga, gb = _score_part(theta, a, b, p, w)
        ll += l
        grad[:n] += gt
        grad[alpha_i] += ga * a - x[alpha_i]
        ll -= 0.5 * x[alpha_i] ** 2
        # thresholds: b_2 = c, b_k = c + sum_{m<k} exp(d_m)
        grad[c_i] += gb.sum() - x[c_i] / prior_sd_thr ** 2
        ll -= 0.5 * (x[c_i] / prior_sd_thr) ** 2
        if gb.size > 1:
            gaps = np.exp(x[d_sl])
            tail = np.cumsum(gb[::-1])[::-1][1:]  # sum of g_b over thresholds after each gap
            grad[d_sl] += tail * gaps - x[d_sl] / prior_sd_thr ** 2
            ll -= 0.5 * ((x[d_sl] / prior_sd_thr) ** 2).sum()
    if lay.g is not None and not only_scores:
        I, J, Q = pairs
        gam = np.exp(x[lay.g])
        diff = theta[I] - theta[J]
        z = gam * diff
        ll += (Q * np.log(_sig(z) + 1e-12) + (1 - Q) * np.log(_sig(-z) + 1e-12)).sum()
        resid = Q - _sig(z)
        np.add.at(grad, I, gam * resid)
        np.add.at(grad, J, -gam * resid)
        grad[lay.g] += (diff * resid).sum() * gam - x[lay.g]
        ll -= 0.5 * x[lay.g] ** 2
    return -ll, -grad


def fit_fusion(indicators: dict[str, np.ndarray], pairs: list[tuple[int, int, float]], n: int) -> dict:
    names = list(indicators)
    data = []
    for nm in names:
        p = np.asarray(indicators[nm], float)
        w = (~np.isnan(p).any(1)).astype(float)
        p = np.where(np.isnan(p), 0.0, p)
        data.append((p, w))
    ks = [d[0].shape[1] for d in data]
    lay = _Layout(n, ks, bool(pairs))
    pair_arr = None
    if pairs:
        pair_arr = (np.array([q[0] for q in pairs]), np.array([q[1] for q in pairs]), np.array([q[2] for q in pairs]))

    # start: theta from the standardised mean expected score; thresholds from marginal cumulative logits
    x0 = np.zeros(lay.size)
    if data:
        exps = []
        for p, w in data:
            e = (p * np.arange(p.shape[1])).sum(1)
            exps.append(np.where(w > 0, e, np.nan))
        stack = np.vstack(exps)
        cnt = (~np.isnan(stack)).sum(0)
        m = np.where(cnt > 0, np.nansum(stack, 0) / np.maximum(cnt, 1), np.nan)
        m = np.where(np.isnan(m), np.nanmean(m), m)
        x0[:n] = (m - m.mean()) / (m.std() + 1e-9)
    for (alpha_i, c_i, d_sl), (p, w) in zip(lay.slices, data):
        marg = (p * w[:, None]).sum(0) / max(w.sum(), 1)
        cum = np.clip(1 - np.cumsum(marg)[:-1], 0.02, 0.98)  # P(Y >= k), k = 2..K
        b = -np.log(cum / (1 - cum))
        b = np.maximum.accumulate(b + np.arange(len(b)) * 1e-3)
        x0[c_i] = b[0]
        if len(b) > 1:
            x0[d_sl] = np.log(np.maximum(np.diff(b), 0.05))

    res = minimize(_objective, x0, args=(lay, data, pair_arr), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-6})
    x = res.x
    theta = x[:n]

    # Laplace SE: score part is separable in theta, so a simultaneous finite difference gives its diagonal;
    # the pairwise and prior parts are added analytically.
    h = 1e-4
    xp, xm = x.copy(), x.copy()
    xp[:n] += h; xm[:n] -= h
    gp = -_objective(xp, lay, data, pair_arr, only_scores=True)[1][:n]
    gm = -_objective(xm, lay, data, pair_arr, only_scores=True)[1][:n]
    diag = (gp - gm) / (2 * h)  # includes the -1 prior term
    if pairs:
        I, J, _ = pair_arr
        gam = np.exp(x[lay.g])
        z = gam * (theta[I] - theta[J])
        wpair = gam ** 2 * _sig(z) * (1 - _sig(z))
        np.add.at(diag, I, -wpair)
        np.add.at(diag, J, -wpair)
    se = 1 / np.sqrt(np.maximum(-diag, 1e-9))

    params = {"converged": bool(res.success), "iterations": int(res.nit), "neg_log_post": float(res.fun)}
    for nm, (alpha_i, c_i, d_sl) in zip(names, lay.slices):
        params[nm] = {"a": float(np.exp(x[alpha_i])), "b": _thresholds(x, c_i, d_sl).round(3).tolist()}
    if lay.g is not None:
        params["pairwise_gamma"] = float(np.exp(x[lay.g]))
    return {"theta": theta, "se": se, "params": params}
