"""A small self-contained HTML report."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


def write_report(path: Path, name: str, metrics: pd.DataFrame, human: dict, meta: dict) -> None:
    fmt = metrics.copy()
    for c in fmt.columns:
        if fmt[c].dtype.kind == "f":
            fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    table = fmt.to_html(index=False, border=0, classes="m")
    hh = ""
    if human:
        hh = ("<p>Human-human benchmark: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                                         for k, v in human.items()) + "</p>")
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(name)}</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#222}}table.m{{border-collapse:collapse;font-size:13px}}
table.m th,table.m td{{padding:4px 8px;border-bottom:1px solid #ddd;text-align:right}}table.m th{{background:#f4f4f4}}
p.note{{color:#666;font-size:12px}}</style></head><body>
<h1>{html.escape(name)}</h1>
<p>{meta['n_essays']} essays, score range {meta['score_range'][0]} to {meta['score_range'][1]}, run time {meta['seconds']} s.</p>
{hh}{table}
<p class="note">qwk: integer score vs rater 1. qwk_h2: vs rater 2. prmse: continuous score vs the human true score
(needs two raters); prmse_int uses the integer score. smd: standardised mean difference (system minus human);
smd[group=level] is the same within a subgroup. sd_ratio below 1 means the model compresses the scale.</p>
</body></html>"""
    Path(path).write_text(doc, encoding="utf-8")
