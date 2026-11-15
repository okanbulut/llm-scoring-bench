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
    a = ap.parse_args(argv)
    if a.cmd == "run":
        cfg = load_config(a.config)
        if a.sample:
            cfg.data.sample = a.sample
        res = run_study(cfg, a.out)
        cols = [c for c in ("model", "strategy", "n", "qwk", "prmse", "exact", "smd", "sd_ratio") if c in res["metrics"]]
        print(res["metrics"][cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
