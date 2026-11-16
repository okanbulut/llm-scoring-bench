"""Repeated anchor draws: how stable is a comparison between strategies?

For each anchor count and each draw (seed), the anchors are re-split, strategies that use anchors are re-fitted
(hybrid, anchored, fusion with tuning, finetune), and every strategy is evaluated on the essays that are not
anchors in that draw. Strategies that do not use anchors are scored once and only their calibration is
re-fitted per draw. Outputs: seeds_runs.csv (one row per model x strategy x anchors x seed), seeds_summary.csv
(mean and SD) and, with a baseline, seeds_diff.csv (paired difference to the baseline and wins).

cache_only=True turns every uncached model call into an error, so an analysis can never spend API calls.
Strategy option `seeds: k` limits an expensive strategy (e.g. finetune) to the first k draws.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backends import make_backend
from .cache import Cache
from .calibrate import CALIBRATORS
from .config import StudyConfig
from .data import load_essays, split_anchors
from .metrics import evaluate
from .runner import _asker
from .strategies import make_strategy

METRICS = ["qwk", "qwk_h2", "prmse", "r", "sd_ratio", "exact", "adjacent", "smd"]


class CacheMiss(RuntimeError):
    pass




def run_seeds(cfg: StudyConfig, anchors: list[int], n_seeds: int = 20, baseline: str | None = None,
              out_dir: str | Path | None = None, cache_only: bool = True, progress: bool = True) -> dict:
    t0 = time.time()
    essays = load_essays(cfg.data)
    lo, hi = essays.score_range
    frame = essays.frame
    records = frame.to_dict("records")
    pos = {ix: k for k, ix in enumerate(frame.index)}
    cache_path = Path(cfg.cache)
    cache = Cache(cache_path if cache_path.is_absolute() else cfg.base_dir / cache_path)
    rows = []
    for m in cfg.models:
        backend = None if cache_only else make_backend(m, cfg.base_dir)

        class _Miss:
            def complete(self, messages):
                raise CacheMiss(f"uncached call for model {m.name} in cache-only mode (run `scorebench run` first)")

        ask = _asker(backend or _Miss(), m, cache)
        fixed: dict[str, list[dict]] = {}
        for s in cfg.strategies:
            k_max = int(s.options.get("seeds", n_seeds))
            for n_anc in anchors:
                for seed in range(1, min(n_seeds, k_max) + 1):
                    anc, test = split_anchors(frame, n_anc, seed)
                    strat = make_strategy(s, cfg.task, (lo, hi))
                    uses_anchors = hasattr(strat, "anchor_labels") and not (
                        s.name == "fusion" and s.options.get("method_sd", "tune") != "tune")
                    if uses_anchors:
                        strat.anchor_labels = {pos[ix]: float(h) for ix, h in anc["h1"].items()}
                        if hasattr(strat, "embed_cache"):
                            strat.embed_cache, strat.base_dir = cache, cfg.base_dir
                        res = strat.score_set(records, ask, cfg.workers)
                    else:
                        if s.display not in fixed:
                            fixed[s.display] = strat.score_set(records, ask, cfg.workers)
                        res = fixed[s.display]
                    pred = frame.drop(columns=["text"]).copy()
                    pred["expected"] = [r["expected"] for r in res]
                    a, t = pred.loc[anc.index], pred.loc[test.index].copy()
                    if s.calibrate:
                        cal = CALIBRATORS[s.calibrate]().fit(a["expected"], a["h1"])
                        t["score_cont"] = cal.transform(t["expected"])
                    else:
                        t["score_cont"] = t["expected"]
                    t["score_int"] = np.clip(np.rint(t["score_cont"]), lo, hi)
                    met = evaluate(t, "score_cont", "score_int", lo, hi, essays.groups)
                    met.update({"model": m.display, "strategy": s.display, "anchors": n_anc, "seed": seed})
                    rows.append(met)
                if progress:
                    print(f"[{m.display} | {s.display}] anchors {n_anc}: done", flush=True)
    runs = pd.DataFrame(rows)
    mcols = [c for c in runs.columns if c in METRICS or c.startswith("smd[")]
    summary = runs.groupby(["model", "strategy", "anchors"])[mcols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    diff = None
    if baseline:
        base = runs[runs.strategy == baseline].set_index(["model", "anchors", "seed"])
        out = []
        for (mod, strat), g in runs[runs.strategy != baseline].groupby(["model", "strategy"]):
            g = g.set_index(["model", "anchors", "seed"])
            j = g.join(base, rsuffix="_base", how="inner")
            for n_anc, h in j.groupby(level="anchors"):
                row = {"model": mod, "strategy": strat, "baseline": baseline, "anchors": n_anc, "draws": len(h)}
                for c in ("qwk", "prmse"):
                    if c in h and f"{c}_base" in h:
                        d = h[c] - h[f"{c}_base"]
                        row[f"{c}_diff_mean"] = float(d.mean())
                        row[f"{c}_diff_sd"] = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
                        if c == "qwk":
                            row["wins"] = int((d > 0).sum())
                out.append(row)
        diff = pd.DataFrame(out)
    out = Path(out_dir or Path(cfg.output_dir) / f"{cfg.name}_seeds")
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "seeds_runs.csv", index=False)
    summary.to_csv(out / "seeds_summary.csv", index=False)
    if diff is not None:
        diff.to_csv(out / "seeds_diff.csv", index=False)
    if progress:
        print(f"Done in {time.time() - t0:.1f} s. Results in {out}")
    return {"runs": runs, "summary": summary, "diff": diff}
