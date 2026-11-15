"""Data loaders. Every loader returns the same frame:

essay_id, text, h1, [h2], plus any subgroup columns. h1 and h2 are human scores on the rater scale.
h1 is the reference score for QWK; PRMSE needs h2 as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig

# Rater (domain 1) score ranges for the eight ASAP-AES prompts.
ASAP_RATER_RANGE = {1: (1, 6), 2: (1, 6), 3: (0, 3), 4: (0, 3), 5: (0, 4), 6: (0, 4), 7: (0, 12), 8: (0, 30)}


@dataclass
class EssaySet:
    frame: pd.DataFrame
    score_range: tuple[int, int]
    groups: list[str]

    @property
    def has_two_raters(self) -> bool:
        return "h2" in self.frame and self.frame["h2"].notna().any()


def _read(path: Path, **kw) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\t", encoding=kw.get("encoding", "utf-8"))
    return pd.read_csv(path, encoding=kw.get("encoding", "utf-8"))


def _apply_filter(df: pd.DataFrame, flt: dict) -> pd.DataFrame:
    for col, val in flt.items():
        df = df[df[col].isin(val if isinstance(val, list) else [val])]
    return df


def load_asap(cfg: DataConfig) -> EssaySet:
    df = _read(Path(cfg.path), encoding="latin-1")
    df = _apply_filter(df, cfg.filter)
    sets = sorted(df["essay_set"].unique())
    if len(sets) != 1:
        raise ValueError("Filter ASAP to one essay_set, e.g. filter: {essay_set: 1}; scales differ by prompt.")
    rng = cfg.score_range or ASAP_RATER_RANGE[int(sets[0])]
    out = pd.DataFrame({
        "essay_id": df["essay_id"].astype(str),
        "text": df["essay"],
        "h1": df["rater1_domain1"].astype(float),
        "h2": df["rater2_domain1"].astype(float),
    })
    return EssaySet(out.reset_index(drop=True), tuple(rng), [])


def load_persuade(cfg: DataConfig) -> EssaySet:
    df = _read(Path(cfg.path))
    df = _apply_filter(df, cfg.filter)
    out = pd.DataFrame({"essay_id": df["essay_id"].astype(str), "text": df["text"], "h1": df["score"].astype(float)})
    for g in cfg.groups:
        out[g] = df[g].values
    return EssaySet(out.reset_index(drop=True), tuple(cfg.score_range or (1, 6)), list(cfg.groups))


def load_table(cfg: DataConfig) -> EssaySet:
    """Any CSV/TSV/parquet. columns: {id: ..., text: ..., human: [rater1_col, rater2_col]}."""
    c = cfg.columns
    if not cfg.score_range:
        raise ValueError("loader 'table' needs data.score_range, e.g. [1, 6]")
    df = _apply_filter(_read(Path(cfg.path)), cfg.filter)
    human = c["human"] if isinstance(c["human"], list) else [c["human"]]
    out = pd.DataFrame({"essay_id": df[c.get("id", "essay_id")].astype(str), "text": df[c.get("text", "text")]})
    for i, col in enumerate(human[:2], start=1):
        out[f"h{i}"] = df[col].astype(float).values
    for g in cfg.groups:
        out[g] = df[g].values
    return EssaySet(out.reset_index(drop=True), tuple(cfg.score_range), list(cfg.groups))


LOADERS = {"asap": load_asap, "persuade": load_persuade, "table": load_table}


def load_essays(cfg: DataConfig) -> EssaySet:
    if cfg.loader not in LOADERS:
        raise ValueError(f"Unknown loader {cfg.loader!r}; choose from {sorted(LOADERS)}")
    es = LOADERS[cfg.loader](cfg)
    es.frame = es.frame.dropna(subset=["text", "h1"])
    if cfg.sample and cfg.sample < len(es.frame):
        es.frame = es.frame.sample(n=cfg.sample, random_state=cfg.seed).reset_index(drop=True)
    return es


def split_anchors(frame: pd.DataFrame, n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out n scored essays (stratified by h1 where possible) for calibration; return (anchors, rest)."""
    if n <= 0:
        return frame.iloc[:0], frame
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    shuffled = frame.iloc[order]
    # spread anchors over score levels: round-robin across h1 levels
    picks: list[int] = []
    by_level = {lvl: list(idx) for lvl, idx in shuffled.groupby("h1").groups.items()}
    while len(picks) < n and any(by_level.values()):
        for lvl in sorted(by_level):
            if by_level[lvl] and len(picks) < n:
                picks.append(by_level[lvl].pop(0))
    anchors = frame.loc[picks]
    return anchors, frame.drop(index=picks)
