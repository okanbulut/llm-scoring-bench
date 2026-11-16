"""Model backends. A backend takes chat messages and returns a reply dict:

{"content": str, "first_token_logprobs": {token: logprob} or None, "prompt_tokens": int, "completion_tokens": int}
"""

from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path

from .config import ModelConfig

RETRY_STATUS = {429, 500, 502, 503, 504}


def resolve_api_key(m: ModelConfig, base_dir: Path = Path(".")) -> str:
    if os.environ.get(m.api_key_env):
        return os.environ[m.api_key_env]
    if m.api_key_file:
        p = Path(m.api_key_file)
        p = p if p.is_absolute() else base_dir / p
        if p.exists():
            hit = re.search(rf"{re.escape(m.api_key_env)}\s*[=:]\s*['\"]?([A-Za-z0-9_\-.]+)", p.read_text(errors="ignore"))
            if hit:
                return hit.group(1)
    raise SystemExit(f"No API key: set ${m.api_key_env} or point api_key_file to a file with a {m.api_key_env} line")


class OpenAIBackend:
    """Any OpenAI-compatible chat endpoint (Vulcan, vLLM, OpenAI, Ollama...)."""

    def __init__(self, m: ModelConfig, base_dir: Path = Path("."), max_wait: float = 420.0):
        from openai import OpenAI

        self.m = m
        self.max_wait = max_wait  # Vulcan models scaled to zero take ~5 min to spin up (HTTP 503)
        self.client = OpenAI(api_key=resolve_api_key(m, base_dir), base_url=m.base_url, timeout=300, max_retries=0)

    def complete(self, messages: list[dict]) -> dict:
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        waited, delay = 0.0, 5.0
        while True:
            try:
                resp = self.client.chat.completions.create(
                    model=self.m.name, messages=messages, temperature=self.m.temperature,
                    max_tokens=self.m.max_tokens, logprobs=self.m.top_logprobs > 0,
                    top_logprobs=self.m.top_logprobs or None, extra_body=self.m.extra_body or None)
                break
            except (APIStatusError, APIConnectionError, APITimeoutError) as e:
                status = getattr(e, "status_code", None)
                if (status is not None and status not in RETRY_STATUS) or waited >= self.max_wait:
                    raise
                time.sleep(delay)
                waited += delay
                delay = min(delay * 2, 60.0)
        ch = resp.choices[0]
        lp = None
        if ch.logprobs and ch.logprobs.content:
            lp = {t.token: t.logprob for t in (ch.logprobs.content[0].top_logprobs or [])}
        usage = resp.usage
        return {"content": ch.message.content or "", "first_token_logprobs": lp,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0}


class FakeBackend:
    """Deterministic stand-in for tests and dry runs.

    It reads a hidden true score from the essay text ("[true=k]") and answers like a harsh, compressed rater:
    most mass one point below the true score, never above `cap`. No network.
    """

    def __init__(self, m: ModelConfig, base_dir: Path = Path("."), severity: int = 1, cap: int = 5):
        self.m, self.severity, self.cap = m, severity, cap

    def complete(self, messages: list[dict]) -> dict:
        text = messages[-1]["content"]
        if "Essay A:" in text and "Essay B:" in text:  # pairwise: noisy comparison with a pull towards "A"
            ta, tb = (int(x) for x in re.findall(r"\[true=(\d+)\]", text)[:2])
            pa = 1 / (1 + math.exp(-(1.2 * (ta - tb) + 0.6)))
            lp = {"A": math.log(max(pa, 1e-9)), "B": math.log(max(1 - pa, 1e-9))}
            return {"content": "A" if pa >= 0.5 else "B", "first_token_logprobs": lp,
                    "prompt_tokens": len(text) // 4, "completion_tokens": 1}
        hit = re.search(r"\[true=(\d+)\]", text)
        true = int(hit.group(1)) if hit else 3
        lo_hi = re.search(r"from (\d+) to (\d+)", text)
        lo, hi = (int(lo_hi.group(1)), int(lo_hi.group(2))) if lo_hi else (1, 6)
        centre = max(lo, min(self.cap, hi, true - self.severity))
        weights = {k: math.exp(-1.5 * abs(k - (centre + 0.2 * (true - centre)))) for k in range(lo, hi + 1)}
        z = sum(weights.values())
        lp = {str(k): math.log(w / z) for k, w in weights.items()}
        top = max(lp, key=lp.get)
        return {"content": top, "first_token_logprobs": lp, "prompt_tokens": len(text) // 4, "completion_tokens": 1}


BACKENDS = {"openai": OpenAIBackend, "fake": FakeBackend}


def make_backend(m: ModelConfig, base_dir: Path = Path(".")):
    if m.backend not in BACKENDS:
        raise ValueError(f"Unknown backend {m.backend!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[m.backend](m, base_dir)
