"""Study configuration: one YAML file names the data, the models, the strategies and the metrics."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    loader: str  # "asap", "persuade" or "table"
    path: str
    filter: dict[str, Any] = field(default_factory=dict)
    sample: int | None = None  # essays drawn per run (None = all)
    seed: int = 1
    score_range: tuple[int, int] | None = None  # rater scale; loaders fill it when known
    columns: dict[str, Any] = field(default_factory=dict)  # for loader "table": id, text, human (list), group cols
    groups: list[str] = field(default_factory=list)  # subgroup columns kept for fairness checks


@dataclass
class TaskConfig:
    prompt_text: str = ""  # the writing prompt the students answered
    rubric: str = ""  # rubric text shown to the model
    context: str = ""  # who wrote the essays and under what conditions, e.g. "Grade 8 students, 45 minutes, no drafts"

    @classmethod
    def from_dict(cls, d: dict, base: Path) -> "TaskConfig":
        def read(key: str) -> str:
            if d.get(key):
                return str(d[key])
            if d.get(f"{key}_file"):
                return (base / d[f"{key}_file"]).read_text(encoding="utf-8")
            return ""

        return cls(prompt_text=read("prompt_text"), rubric=read("rubric"), context=read("context"))


@dataclass
class ModelConfig:
    name: str  # model id sent to the API
    backend: str = "openai"  # "openai" (any OpenAI-compatible server) or "fake" (tests)
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    api_key_file: str | None = None  # a file with a line KEY_NAME = 'value' (key name = api_key_env)
    extra_body: dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.0
    max_tokens: int = 8
    top_logprobs: int = 10
    label: str | None = None  # name shown in reports

    @property
    def display(self) -> str:
        return self.label or self.name


@dataclass
class StrategyConfig:
    name: str  # "vanilla" for now
    calibrate: str | None = None  # None or "quantile"
    anchors: int = 0  # scored essays held out to fit the calibration
    label: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def display(self) -> str:
        if self.label:
            return self.label
        return self.name + (f"+{self.calibrate}{self.anchors}" if self.calibrate else "")


@dataclass
class StudyConfig:
    name: str
    data: DataConfig
    task: TaskConfig
    models: list[ModelConfig]
    strategies: list[StrategyConfig]
    output_dir: str = "runs"
    cache: str = "runs/cache.sqlite"
    workers: int = 4
    base_dir: Path = Path(".")


def _strategy(s: Any) -> StrategyConfig:
    if isinstance(s, str):
        return StrategyConfig(name=s)
    s = dict(s)
    known = {k: s.pop(k) for k in list(s) if k in ("name", "calibrate", "anchors", "label")}
    return StrategyConfig(**known, options=s)


def load_config(path: str | os.PathLike) -> StudyConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent
    d = dict(raw["data"])
    if d.get("score_range"):
        d["score_range"] = tuple(d["score_range"])
    return StudyConfig(
        name=raw.get("name", path.stem),
        data=DataConfig(**d),
        task=TaskConfig.from_dict(raw.get("task", {}), base),
        models=[ModelConfig(**m) for m in raw["models"]],
        strategies=[_strategy(s) for s in raw.get("strategies", ["vanilla"])],
        output_dir=raw.get("output_dir", "runs"),
        cache=raw.get("cache", "runs/cache.sqlite"),
        workers=int(raw.get("workers", 4)),
        base_dir=base,
    )
