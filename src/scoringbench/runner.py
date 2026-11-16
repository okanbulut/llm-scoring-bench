"""Run a study: every model x strategy on the same essays, then metrics and a report."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .backends import make_backend
from .cache import Cache, request_key
from .calibrate import CALIBRATORS
from .config import StudyConfig
from .data import load_essays, split_anchors
from .metrics import reliability_check
from .metrics import evaluate, human_benchmark
from .report import write_report
from .strategies import make_strategy


def _asker(backend, model_cfg, cache: Cache):
    def ask(messages: list[dict]) -> dict:
        key = request_key({"model": model_cfg.name, "base_url": model_cfg.base_url, "messages": messages,
                           "temperature": model_cfg.temperature, "max_tokens": model_cfg.max_tokens,
                           "top_logprobs": model_cfg.top_logprobs, "extra_body": model_cfg.extra_body})
        hit = cache.get(key)
        if hit is not None:
            return hit
        reply = backend.complete(messages)
        cache.put(key, reply)
        return reply
    return ask


def run_study(cfg: StudyConfig, out_dir: str | Path | None = None, progress: bool = True) -> dict:
    t0 = time.time()
    essays = load_essays(cfg.data)
    lo, hi = essays.score_range
    out = Path(out_dir or Path(cfg.output_dir) / cfg.name)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cfg.cache)
    cache = Cache(cache_path if cache_path.is_absolute() else cfg.base_dir / cache_path)
    frame = essays.frame
    rows, metric_rows = [], []
    fusion_params: dict = {}

    for m in cfg.models:
        ask = _asker(make_backend(m, cfg.base_dir), m, cache)
        for s in cfg.strategies:
            strat = make_strategy(s, cfg.task, (lo, hi))
            if s.calibrate and hasattr(strat, "anchor_labels"):
                # same split as below (same seed, same h1), made before scoring so a strategy can tune on it
                anc, _ = split_anchors(frame, s.anchors, cfg.data.seed)
                pos = {ix: k for k, ix in enumerate(frame.index)}
                strat.anchor_labels = {pos[ix]: float(h) for ix, h in anc["h1"].items()}
            if hasattr(strat, "embed_cache"):
                strat.embed_cache, strat.base_dir = cache, cfg.base_dir
            if progress:
                print(f"[{m.display} | {s.display}] scoring {len(frame)} essays ...", flush=True)
            results = strat.score_set(frame.to_dict("records"), ask, cfg.workers)
            res = pd.DataFrame(results)
            pred = frame.drop(columns=["text"]).copy()
            pred["expected"] = res["expected"].values
            pred["mode"] = res["mode"].values
            pred["coverage"] = res["coverage"].values
            pred["source"] = res["source"].values
            pred["calls"] = res["calls"].values
            pred["tokens"] = res["tokens"].values
            pred["probs"] = [json.dumps(p) for p in res["probs"]]
            if "raw" in res:
                pred["raw"] = res["raw"].values
            if "se" in res:
                pred["se"] = res["se"].values
            if "fusion_params" in res and res["fusion_params"].notna().any():
                fusion_params[f"{m.display} | {s.display}"] = res["fusion_params"].dropna().iloc[0]
            if "traits" in res:
                pred["traits"] = [json.dumps(t) for t in res["traits"]]
            anchors, test = split_anchors(pred, s.anchors, cfg.data.seed) if s.calibrate else (pred.iloc[:0], pred)
            test = test.copy()
            if s.calibrate:
                cal = CALIBRATORS[s.calibrate]().fit(anchors["expected"], anchors["h1"])
                test["score_cont"] = cal.transform(test["expected"])
            else:
                test["score_cont"] = test["expected"]
            test["score_int"] = np.clip(np.rint(test["score_cont"]), lo, hi)
            test["model"], test["strategy"] = m.display, s.display
            rows.append(test)
            met = evaluate(test, "score_cont", "score_int", lo, hi, essays.groups)
            if "se" in test and "raw" in test:
                met.update(reliability_check(test))
            met.update({"model": m.display, "strategy": s.display, "anchors": len(anchors),
                        "logprob_share": float((test["source"] == "logprobs").mean()),
                        "unparsed": int((test["source"] == "unparsed").sum()),
                        "calls_per_essay": float(test["calls"].mean()), "tokens_per_essay": float(test["tokens"].mean())})
            metric_rows.append(met)

    preds = pd.concat(rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    lead = ["model", "strategy", "n", "anchors", "qwk", "qwk_h2", "prmse", "prmse_int", "exact", "adjacent",
            "r", "mean_diff", "smd", "sd_ratio"]
    metrics = metrics[[c for c in lead if c in metrics] + [c for c in metrics if c not in lead]]
    human = human_benchmark(frame, lo, hi)
    preds.to_csv(out / "predictions.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    meta = {"study": cfg.name, "n_essays": len(frame), "score_range": [lo, hi], "human_human": human,
            "seconds": round(time.time() - t0, 1), "data": asdict(cfg.data),
            "models": [{k: v for k, v in asdict(m).items() if "key" not in k} for m in cfg.models],
            "strategies": [asdict(s) for s in cfg.strategies], "fusion_params": fusion_params}
    (out / "run.json").write_text(json.dumps(meta, indent=2, default=str))
    write_report(out / "report.html", cfg.name, metrics, human, meta)
    if progress:
        print(f"Done in {meta['seconds']} s. Results in {out}")
    return {"metrics": metrics, "predictions": preds, "human_human": human, "out_dir": out}
