"""Agreement and accuracy metrics used in operational automated scoring.

PRMSE follows Haberman (2008) and Loukina et al. (2020), "Using PRMSE to evaluate automated scoring
systems in the presence of label noise", as implemented in ETS's rsmtool: it estimates how well system
scores predict the human TRUE score, using the disagreement between two human raters to estimate
rater error variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(*arrays):
    arrs = [np.asarray(a, float) for a in arrays]
    ok = np.ones(len(arrs[0]), bool)
    for a in arrs:
        ok &= ~np.isnan(a)
    return [a[ok] for a in arrs]


def qwk(a, b, lo: int | None = None, hi: int | None = None) -> float:
    """Quadratic weighted kappa between two integer score vectors (non-integers are rounded)."""
    a, b = _clean(a, b)
    if len(a) == 0:
        return float("nan")
    a, b = np.rint(a).astype(int), np.rint(b).astype(int)
    lo = min(a.min(), b.min()) if lo is None else lo
    hi = max(a.max(), b.max()) if hi is None else hi
    k = hi - lo + 1
    if k <= 1:
        return float("nan")
    obs = np.zeros((k, k))
    np.add.at(obs, (np.clip(a, lo, hi) - lo, np.clip(b, lo, hi) - lo), 1)
    w = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    den = (w * exp).sum()
    return float(1 - (w * obs).sum() / den) if den > 0 else float("nan")


def human_error_variance(human: np.ndarray) -> float:
    """Mean within-essay variance of the human ratings, over essays with 2+ ratings."""
    h = np.asarray(human, float)
    multi = (~np.isnan(h)).sum(1) >= 2
    if not multi.any():
        return float("nan")
    return float(np.nanmean(np.nanvar(h[multi], axis=1, ddof=1)))


def true_score_variance(human: np.ndarray, var_e: float | None = None) -> float:
    h = np.asarray(human, float)
    var_e = human_error_variance(h) if var_e is None else var_e
    n = (~np.isnan(h)).sum(1)
    keep = n > 0
    h, n = h[keep], n[keep]
    mean_i = np.nanmean(h, 1)
    mu = np.nansum(h) / n.sum()
    num = (n * (mean_i - mu) ** 2).sum() - (len(n) - 1) * var_e
    den = (n.sum() ** 2 - (n ** 2).sum()) / n.sum()
    return float(num / den)


def prmse(system, h1, h2) -> float:
    """Proportional reduction in mean squared error for predicting the human true score."""
    s = np.asarray(system, float)
    human = np.column_stack([np.asarray(h1, float), np.asarray(h2, float)])
    ok = ~np.isnan(s) & ((~np.isnan(human)).sum(1) > 0)
    s, human = s[ok], human[ok]
    var_e = human_error_variance(human)
    if np.isnan(var_e):
        return float("nan")
    n = (~np.isnan(human)).sum(1)
    mean_i = np.nanmean(human, 1)
    mse_true = ((n * (mean_i - s) ** 2).sum() - len(n) * var_e) / n.sum()
    vt = true_score_variance(human, var_e)
    return float(1 - mse_true / vt) if vt > 0 else float("nan")


def agreement(a, b) -> dict:
    a, b = _clean(a, b)
    a, b = np.rint(a), np.rint(b)
    d = np.abs(a - b)
    return {"exact": float((d == 0).mean()), "adjacent": float((d <= 1).mean())} if len(a) else {"exact": np.nan, "adjacent": np.nan}


def smd(system, human) -> float:
    """Standardised mean difference, (mean system - mean human) / pooled SD."""
    s, h = _clean(system, human)
    if len(s) < 2:
        return float("nan")
    sd = np.sqrt((s.var(ddof=1) + h.var(ddof=1)) / 2)
    return float((s.mean() - h.mean()) / sd) if sd > 0 else float("nan")


def evaluate(df: pd.DataFrame, cont_col: str, int_col: str, lo: int, hi: int, groups: list[str]) -> dict:
    """All metrics for one prediction set. df has h1, optional h2, and the two prediction columns."""
    out = {
        "n": int(df[int_col].notna().sum()),
        "qwk": qwk(df[int_col], df["h1"], lo, hi),
        **agreement(df[int_col], df["h1"]),
        "mean_diff": float(np.nanmean(df[cont_col] - df["h1"])),
        "smd": smd(df[cont_col], df["h1"]),
        "sd_ratio": float(np.nanstd(df[cont_col], ddof=1) / np.nanstd(df["h1"], ddof=1)),
        "r": float(pd.Series(df[cont_col]).corr(df["h1"])),
    }
    if "h2" in df and df["h2"].notna().any():
        out["qwk_h2"] = qwk(df[int_col], df["h2"], lo, hi)
        out["prmse"] = prmse(df[cont_col], df["h1"], df["h2"])
        out["prmse_int"] = prmse(df[int_col], df["h1"], df["h2"])
    for g in groups:
        for level, sub in df.groupby(g, dropna=True):
            out[f"smd[{g}={level}]"] = smd(sub[cont_col], sub["h1"])
    return out


def human_benchmark(df: pd.DataFrame, lo: int, hi: int) -> dict:
    if "h2" not in df or not df["h2"].notna().any():
        return {}
    return {"n": int(df["h2"].notna().sum()), "qwk": qwk(df["h1"], df["h2"], lo, hi), **agreement(df["h1"], df["h2"]),
            "smd": smd(df["h2"], df["h1"])}
