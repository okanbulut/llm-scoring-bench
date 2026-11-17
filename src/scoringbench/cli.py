"""Command line: scorebench run study.yaml"""

from __future__ import annotations

import argparse

from .config import load_config
from .runner import run_study


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="scorebench", description="Test LLMs on automated essay scoring.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run every model x strategy in a study config")
    r.add_argument("config")
    r.add_argument("--out", help="output folder (default: output_dir/name from the config)")
    r.add_argument("--sample", type=int, help="override data.sample, e.g. for a quick check")
    sd = sub.add_parser("seeds", help="repeat the anchor split over many draws and compare strategies")
    sd.add_argument("config")
    sd.add_argument("--anchors", type=int, nargs="+", default=[50, 100, 200])
    sd.add_argument("--seeds", type=int, default=20)
    sd.add_argument("--baseline", help="strategy label to compare against (paired by draw)")
    sd.add_argument("--out")
    sd.add_argument("--allow-calls", action="store_true",
                    help="allow uncached model calls (default: cache only, an uncached call is an error)")
    a = ap.parse_args(argv)
    if a.cmd == "seeds":
        from .seeds import run_seeds

        res = run_seeds(load_config(a.config), a.anchors, a.seeds, a.baseline, a.out, cache_only=not a.allow_calls)
        cols = [c for c in res["summary"].columns if c in ("model", "strategy", "anchors") or c.startswith("qwk")]
        print(res["summary"][cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        if res["diff"] is not None:
            print(res["diff"].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
        return
    if a.cmd == "run":
        cfg = load_config(a.config)
        if a.sample:
            cfg.data.sample = a.sample
        res = run_study(cfg, a.out)
        cols = [c for c in ("model", "strategy", "n", "qwk", "prmse", "exact", "smd", "sd_ratio") if c in res["metrics"]]
        print(res["metrics"][cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
