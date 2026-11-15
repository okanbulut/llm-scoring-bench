"""Prompt strategies. A strategy turns one essay into one or more chat requests and reads back a score.

Only `vanilla` ships in this first version. MTS (trait-by-trait), LCES (pairwise) and the fusion model
plug in through the same interface.
"""

from __future__ import annotations

from typing import Callable

from .config import StrategyConfig, TaskConfig
from .parse import score_distribution

SYSTEM = "You are an experienced, fair essay rater. You score student essays exactly as the rubric says."


class Strategy:
    name = "base"

    def __init__(self, cfg: StrategyConfig, task: TaskConfig, score_range: tuple[int, int]):
        self.cfg, self.task, (self.lo, self.hi) = cfg, task, score_range

    def score(self, essay: dict, ask: Callable[[list[dict]], dict]) -> dict:
        raise NotImplementedError


class Vanilla(Strategy):
    """Rubric + essay, answer with one integer. With log-probs, the full distribution over score points
    is kept, so the expected score is available as well as the most likely score."""

    name = "vanilla"

    def messages(self, essay: dict) -> list[dict]:
        parts = []
        if self.task.prompt_text:
            parts.append(f"Writing prompt the student answered:\n{self.task.prompt_text.strip()}")
        if self.task.rubric:
            parts.append(f"Scoring rubric:\n{self.task.rubric.strip()}")
        parts.append(f"Essay:\n{essay['text'].strip()}")
        parts.append(f"Score this essay on a scale from {self.lo} to {self.hi}. "
                     f"Answer with a single integer only, with no other text.")
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]

    def score(self, essay: dict, ask: Callable[[list[dict]], dict]) -> dict:
        reply = ask(self.messages(essay))
        out = score_distribution(reply, self.lo, self.hi)
        out["calls"] = 1
        out["tokens"] = reply.get("prompt_tokens", 0) + reply.get("completion_tokens", 0)
        return out


STRATEGIES = {"vanilla": Vanilla}


def make_strategy(cfg: StrategyConfig, task: TaskConfig, score_range: tuple[int, int]) -> Strategy:
    if cfg.name not in STRATEGIES:
        raise ValueError(f"Unknown strategy {cfg.name!r}; available: {sorted(STRATEGIES)}")
    return STRATEGIES[cfg.name](cfg, task, score_range)
