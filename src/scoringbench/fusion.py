"""Joint latent-variable model for LLM judgments ("psychometric fusion").

Indicators
- Score indicators j (holistic score, each trait score): the LLM gives a probability vector p_ij over K
  ordered categories. A graded response model links it to the essay's latent quality theta_i:
      P(Y_ij >= k | theta_i) = logistic(a_j * theta_i - b_jk),  k = 2..K,  b_j2 < ... < b_jK.
  The LLM's distribution enters as soft data: sum_k p_ijk * log P(Y_ij = k | theta_i). Thresholds absorb each
  indicator's severity and scale compression; a_j says how strongly it reflects the common quality.
- Comparisons (i, j, q): q = P(essay i judged better), averaged over both orders. Bradley-Terry:
      q * log logistic(g * (theta_i - theta_j)) + (1 - q) * log logistic(-g * (theta_i - theta_j)).

- Method factor (v2): the score indicators all come from the same model reading the same essay, so their
  errors are correlated (halo). With method_sd > 0 they load on theta_i + u_i, u_i ~ N(0, method_sd^2), a
  per-essay term shared by every score indicator but not by the comparisons. The score indicators then count
  as one correlated source: they pin theta + u, and how much of that reaches theta depends on method_sd.
  method_sd = 0 gives the v1 model. method_sd is not estimated by MAP (a hierarchical MAP collapses to 0);
  choose it on human-scored anchors with tune_method_sd, or fix it.

Scale (v2): one loading is fixed at 1 (gamma when there are comparisons, else the first indicator's a), as in
IRT identification. v1 left all loadings free, and the joint MAP then shrank theta towards 0 while the
loadings grew without bound (a and gamma of 10-43) and the Laplace SEs became too small.
Priors: theta_i ~ N(0, THETA_SD^2); log a_j, log g ~ N(0, 1); first threshold and log threshold gaps ~ N(0, 3^2).
Estimation: joint MAP with L-BFGS (analytic gradient). SE(theta_i): Laplace approximation from the per-essay
(theta_i, u_i) block of the Hessian with the other parameters held at their estimates.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


THETA_SD = 2.0


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class _Layout:
    def __init__(self, n: int, ks: list[int], has_pairs: bool, method: bool = False):
        self.n, self.ks = n, ks
        self.slices = []
        pos = n
        self.u = slice(pos, pos + n) if method else None
        pos += n if method else 0
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


def _objective(x, lay: _Layout, data, pairs, prior_sd_thr=3.0, only_scores=False, method_sd=0.0):
    n = lay.n
    theta = x[:n]
    grad = np.zeros_like(x)
    ll = 0.0
    if not only_scores:
        ll -= 0.5 * (theta ** 2).sum() / THETA_SD ** 2
        grad[:n] -= theta / THETA_SD ** 2
    eff = theta
    if lay.u is not None:
        u = x[lay.u]
        eff = theta + u
        if not only_scores:
            ll -= 0.5 * (u ** 2).sum() / method_sd ** 2
            grad[lay.u] -= u / method_sd ** 2
    for (alpha_i, c_i, d_sl), (p, w) in zip(lay.slices, data):
        a = np.exp(x[alpha_i])
        b = _thresholds(x, c_i, d_sl)
        l, gt, ga, gb = _score_part(eff, a, b, p, w)
        ll += l
        grad[:n] += gt
        if lay.u is not None:
            grad[lay.u] += gt
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


def fit_fusion(indicators: dict[str, np.ndarray], pairs: list[tuple[int, int, float]], n: int,
               method_sd: float = 0.0) -> dict:
    """Fit the joint model. method_sd > 0 adds the per-essay method factor shared by the score indicators
    (only identified apart from theta when there are comparisons; without them it is ignored)."""
    names = list(indicators)
    method = bool(method_sd > 0 and pairs and indicators)
    data = []
    for nm in names:
        p = np.asarray(indicators[nm], float)
        w = (~np.isnan(p).any(1)).astype(float)
        p = np.where(np.isnan(p), 0.0, p)
        data.append((p, w))
    ks = [d[0].shape[1] for d in data]
    lay = _Layout(n, ks, bool(pairs), method)
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
    if pairs and (method or not data):
        # comparisons carry the ranking that the method factor separates from the score indicators, so
        # start theta there; the score part then only has to explain theta + u
        from .strategies import bradley_terry
        bt = np.asarray(bradley_terry(n, pairs), float)
        x0[:n] = (bt - bt.mean()) / (bt.std() + 1e-9)
        if method and data:
            x0[lay.u] = np.clip(((m - m.mean()) / (m.std() + 1e-9)) - x0[:n], -3 * method_sd, 3 * method_sd)
    for (alpha_i, c_i, d_sl), (p, w) in zip(lay.slices, data):
        marg = (p * w[:, None]).sum(0) / max(w.sum(), 1)
        cum = np.clip(1 - np.cumsum(marg)[:-1], 0.02, 0.98)  # P(Y >= k), k = 2..K
        b = -np.log(cum / (1 - cum))
        b = np.maximum.accumulate(b + np.arange(len(b)) * 1e-3)
        x0[c_i] = b[0]
        if len(b) > 1:
            x0[d_sl] = np.log(np.maximum(np.diff(b), 0.05))

    bounds = [(None, None)] * lay.size
    if lay.g is not None:
        bounds[lay.g] = (0.0, 0.0)  # gamma = 1
    elif lay.slices:
        bounds[lay.slices[0][0]] = (0.0, 0.0)  # first a = 1
    for i, bd in enumerate(bounds):
        if bd[0] is not None:
            x0[i] = 0.0
    res = minimize(_objective, x0, args=(lay, data, pair_arr, 3.0, False, method_sd), jac=True,
                   method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000, "gtol": 1e-6})
    x = res.x
    theta = x[:n]

    # Laplace SE. The score part depends on essay i only through eta_i = theta_i (+ u_i), so a simultaneous
    # finite difference in theta gives its curvature s_i (<= 0) for every essay at once. Pairwise information
    # P_i is added analytically (diagonal). Per essay, the negative Hessian over (theta_i, u_i) is
    #   [[-s + 1 + P, -s], [-s, -s + 1/method_sd^2]]   and Var(theta_i) is its inverse's (0, 0) element.
    h = 1e-4
    xp, xm = x.copy(), x.copy()
    xp[:n] += h; xm[:n] -= h
    gp = -_objective(xp, lay, data, pair_arr, only_scores=True)[1][:n]
    gm = -_objective(xm, lay, data, pair_arr, only_scores=True)[1][:n]
    s_curv = np.minimum((gp - gm) / (2 * h), 0.0)
    info_p = np.zeros(n)
    if pairs:
        I, J, _ = pair_arr
        gam = np.exp(x[lay.g])
        z = gam * (theta[I] - theta[J])
        wpair = gam ** 2 * _sig(z) * (1 - _sig(z))
        np.add.at(info_p, I, wpair)
        np.add.at(info_p, J, wpair)
    if method:
        h11, h12, h22 = -s_curv + THETA_SD ** -2 + info_p, -s_curv, -s_curv + 1 / method_sd ** 2
        var = h22 / (h11 * h22 - h12 ** 2)
    else:
        var = 1 / (-s_curv + THETA_SD ** -2 + info_p)
    se = np.sqrt(np.maximum(var, 1e-12))

    # model-implied reliability of theta: share of the prior variance the data explain
    rel = float(np.var(theta) / (np.var(theta) + np.mean(se ** 2)))
    params = {"converged": bool(res.success), "iterations": int(res.nit), "neg_log_post": float(res.fun),
              "method_sd": float(method_sd if method else 0.0), "reliability_model": rel}
    if method:
        params["method_u_sd"] = float(np.std(x[lay.u]))
    for nm, (alpha_i, c_i, d_sl) in zip(names, lay.slices):
        params[nm] = {"a": float(np.exp(x[alpha_i])), "b": _thresholds(x, c_i, d_sl).round(3).tolist()}
    if lay.g is not None:
        params["pairwise_gamma"] = float(np.exp(x[lay.g]))
    return {"theta": theta, "se": se, "params": params}


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    # average ranks for ties in b (human scores are integers)
    for v in np.unique(b):
        m = b == v
        rb[m] = rb[m].mean()
    return float(np.corrcoef(ra, rb)[0, 1])


METHOD_SD_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def tune_method_sd(indicators, pairs, n, anchor_idx, anchor_scores, grid=METHOD_SD_GRID) -> dict:
    """Pick method_sd by the Spearman correlation of theta with the human scores of the anchor essays.
    The anchors are only used to choose this one number; theta is still fitted on all essays without human
    scores. Returns the chosen fit with params['tuning'] = {method_sd: anchor correlation}."""
    anchor_idx = np.asarray(anchor_idx, int)
    anchor_scores = np.asarray(anchor_scores, float)
    best, table = None, {}
    for sd in grid:
        fit = fit_fusion(indicators, pairs, n, method_sd=sd)
        rho = _spearman(fit["theta"][anchor_idx], anchor_scores)
        table[str(sd)] = round(rho, 4)
        if best is None or rho > best[0] + 1e-9:
            best = (rho, fit)
    best[1]["params"]["tuning"] = table
    return best[1]
