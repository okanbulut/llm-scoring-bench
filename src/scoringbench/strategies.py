"""Prompt strategies. A strategy turns one essay into one or more chat requests and reads back a score.

Strategies:
- vanilla: rubric + essay, one integer; the score distribution is read from log-probabilities.
- mts: Multi Trait Specialization (Lee et al., 2024): one call per trait, trait scores averaged, then
  min-max scaled over the essay set to the rubric range.
- pairwise: comparative judgment in the spirit of LCES (Shibata et al., 2025): each essay is compared with
  k random partners in both orders, P(A better) from log-probabilities, a Bradley-Terry model with soft
  outcomes gives a latent quality, min-max scaled to the rubric range.

A strategy scores a whole set (`score_set`), so set-level methods (scaling, pairwise models) fit the same
interface as per-essay prompting.
"""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from .config import StrategyConfig, TaskConfig
from .parse import score_distribution

SYSTEM = "You are an experienced, fair essay rater. You score student essays exactly as the rubric says."


class Strategy:
    name = "base"

    def __init__(self, cfg: StrategyConfig, task: TaskConfig, score_range: tuple[int, int]):
        self.cfg, self.task, (self.lo, self.hi) = cfg, task, score_range

    def score(self, essay: dict, ask: Callable[[list[dict]], dict]) -> dict:
        raise NotImplementedError

    def score_set(self, essays: list[dict], ask: Callable[[list[dict]], dict], workers: int = 4) -> list[dict]:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(lambda e: self.score(e, ask), essays))

    def header_parts(self) -> list[str]:
        parts = []
        if self.task.context:
            parts.append(f"About the writers:\n{self.task.context.strip()}\nJudge the essay against what the rubric "
                         f"expects of these writers, not against adult or professional writing.")
        if self.task.prompt_text:
            parts.append(f"Writing prompt the student answered:\n{self.task.prompt_text.strip()}")
        if self.task.rubric:
            parts.append(f"Scoring rubric:\n{self.task.rubric.strip()}")
        return parts

    def minmax(self, x) -> np.ndarray:
        x = np.asarray(x, float)
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(lo) or hi <= lo:
            return np.full_like(x, (self.lo + self.hi) / 2)
        return self.lo + (x - lo) / (hi - lo) * (self.hi - self.lo)


class Vanilla(Strategy):
    """Rubric + essay, answer with one integer. With log-probs, the full distribution over score points
    is kept, so the expected score is available as well as the most likely score."""

    name = "vanilla"

    def messages(self, essay: dict) -> list[dict]:
        parts = self.header_parts()
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


DEFAULT_TRAITS = [
    {"name": "Content", "description": "position, ideas, development and use of reasons and evidence"},
    {"name": "Organization", "description": "structure, introduction and conclusion, paragraphing, transitions, coherence"},
    {"name": "Language", "description": "word choice, sentence variety, grammar, usage, spelling and mechanics"},
]


class MTS(Strategy):
    """Multi Trait Specialization (Lee et al., 2024, Findings of EMNLP). Each trait is scored in its own call
    on the rubric range; the overall score is the mean of trait expected scores, min-max scaled over the set.
    Options: traits (list of {name, description}); scale: "minmax" (default, as in the paper) or "none"."""

    name = "mts"

    def __init__(self, cfg, task, score_range):
        super().__init__(cfg, task, score_range)
        self.traits = cfg.options.get("traits") or DEFAULT_TRAITS

    def messages(self, essay: dict, trait: dict) -> list[dict]:
        parts = self.header_parts()
        parts.append(f"Essay:\n{essay['text'].strip()}")
        parts.append(f"Focus only on this trait: {trait['name']} ({trait['description']}). Ignore every other "
                     f"aspect of the essay. Score the essay on {trait['name']} on a scale from {self.lo} to {self.hi}. "
                     f"Answer with a single integer only, with no other text.")
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]

    def score(self, essay: dict, ask) -> dict:
        trait_scores, tokens = {}, 0
        for t in self.traits:
            reply = ask(self.messages(essay, t))
            tokens += reply.get("prompt_tokens", 0) + reply.get("completion_tokens", 0)
            trait_scores[t["name"]] = score_distribution(reply, self.lo, self.hi)
        exp = [d["expected"] for d in trait_scores.values()]
        mean = float(np.nanmean(exp)) if not all(math.isnan(e) for e in exp) else float("nan")
        srcs = {d["source"] for d in trait_scores.values()}
        return {"expected": mean, "mode": float("nan"), "probs": [], "coverage": float(np.mean([d["coverage"] for d in trait_scores.values()])),
                "source": "logprobs" if srcs == {"logprobs"} else ("unparsed" if "unparsed" in srcs else "text"),
                "calls": len(self.traits), "tokens": tokens,
                "traits": {k: round(v["expected"], 4) for k, v in trait_scores.items()},
                "trait_probs": {k: v["probs"] for k, v in trait_scores.items()}}

    def score_set(self, essays, ask, workers=4):
        out = super().score_set(essays, ask, workers)
        if self.cfg.options.get("scale", "minmax") == "minmax":
            scaled = self.minmax([o["expected"] for o in out])
            for o, v in zip(out, scaled):
                o["raw"], o["expected"] = o["expected"], float(v)
        return out


def ab_probability(reply: dict) -> float:
    """P(first essay better) from first-token log-probs over "A"/"B"; falls back to the reply text."""
    lp = reply.get("first_token_logprobs") or {}
    pa = sum(math.exp(v) for k, v in lp.items() if k.strip().upper() == "A")
    pb = sum(math.exp(v) for k, v in lp.items() if k.strip().upper() == "B")
    if pa + pb > 0.5:
        return pa / (pa + pb)
    m = re.search(r"\b([AB])\b", (reply.get("content") or "").upper())
    return {"A": 1.0, "B": 0.0}.get(m.group(1), 0.5) if m else 0.5


def bradley_terry(n: int, pairs: list[tuple[int, int, float]], ridge: float = 0.01, iters: int = 200) -> np.ndarray:
    """Latent strengths from soft pairwise outcomes p = P(i beats j), by Newton steps on the penalised
    log-likelihood sum p*log s(ti-tj) + (1-p)*log s(tj-ti) - ridge/2 * sum t^2."""
    theta = np.zeros(n)
    if not pairs:
        return theta
    I = np.array([p[0] for p in pairs]); J = np.array([p[1] for p in pairs]); P = np.array([p[2] for p in pairs])
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(theta[I] - theta[J])))
        g = np.zeros(n); np.add.at(g, I, P - q); np.add.at(g, J, q - P); g -= ridge * theta
        w = q * (1 - q)
        H = np.zeros((n, n))
        np.add.at(H, (I, I), -w); np.add.at(H, (J, J), -w); np.add.at(H, (I, J), w); np.add.at(H, (J, I), w)
        H -= ridge * np.eye(n)
        step = np.linalg.solve(H, g)
        theta -= step
        if np.abs(step).max() < 1e-6:
            break
    return theta - theta.mean()


class Pairwise(Strategy):
    """Comparative judgment. Options: k (partners per essay, default 6), seed, both_orders (default true:
    each pair is asked as A/B and B/A and the two probabilities averaged, which removes position bias)."""

    name = "pairwise"

    def messages(self, a: dict, b: dict) -> list[dict]:
        parts = self.header_parts()
        parts.append(f"Essay A:\n{a['text'].strip()}")
        parts.append(f"Essay B:\n{b['text'].strip()}")
        parts.append("Which essay is better according to the rubric? Answer with a single letter, A or B, "
                     "with no other text.")
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]

    def plan_pairs(self, n: int) -> list[tuple[int, int]]:
        k = int(self.cfg.options.get("k", 6))
        rng = np.random.default_rng(int(self.cfg.options.get("seed", 1)))
        seen, pairs = set(), []
        for _ in range(max(1, k // 2) + 1):
            order = rng.permutation(n)
            for a, b in zip(order, np.roll(order, 1)):
                key = (min(a, b), max(a, b))
                if a != b and key not in seen:
                    seen.add(key); pairs.append(key)
        # cap so that each essay has about k partners on average
        return pairs[: max(1, n * k // 2)]

    def collect(self, essays, ask, workers=4):
        """Ask every planned comparison. Returns (pairs [(i, j, P(i better))], calls per essay, tokens per essay)."""
        n = len(essays)
        both = bool(self.cfg.options.get("both_orders", True))
        jobs = []
        for i, j in self.plan_pairs(n):
            jobs.append((i, j, False))
            if both:
                jobs.append((j, i, True))

        def run(job):
            a, b, _ = job
            r = ask(self.messages(essays[a], essays[b]))
            return ab_probability(r), r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            res = list(pool.map(run, jobs))
        calls, tokens = np.zeros(n), np.zeros(n)
        prob = {}
        for (a, b, flipped), (p, tok) in zip(jobs, res):
            key = (b, a) if flipped else (a, b)
            prob.setdefault(key, []).append(1 - p if flipped else p)
            for e in (a, b):
                calls[e] += 0.5; tokens[e] += tok / 2
        pairs = [(i, j, float(np.mean(ps))) for (i, j), ps in prob.items()]
        return pairs, calls, tokens

    def score_set(self, essays, ask, workers=4):
        n = len(essays)
        pairs, calls, tokens = self.collect(essays, ask, workers)
        theta = bradley_terry(n, pairs)
        scaled = self.minmax(theta)
        return [{"expected": float(scaled[e]), "raw": float(theta[e]), "mode": float("nan"), "probs": [],
                 "coverage": 1.0, "source": "logprobs", "calls": float(calls[e]), "tokens": float(tokens[e])}
                for e in range(n)]


class Fusion(Strategy):
    """Psychometric fusion of LLM judgments (this project's method).

    Collects, for every essay, the holistic score distribution (vanilla prompt), one score distribution per
    trait (MTS prompts) and soft pairwise outcomes (pairwise prompts), all with the same prompts as those
    strategies so cached replies are reused. A joint model treats them as indicators of one latent quality:
    a graded response model per score indicator (own discrimination and thresholds, which absorb severity and
    scale compression) and a Bradley-Terry model for comparisons. Output: the latent score (MAP) and its SE,
    min-max scaled to the rubric range unless a calibration is set.
    v2: a method factor shared by the score indicators (method_sd). With anchors (calibrate + anchors set),
    method_sd is chosen on the anchors' human scores from METHOD_SD_GRID unless fixed in the options.
    Options: components (subset of [vanilla, mts, pairwise]; default all), traits, k, seed, both_orders,
    scale ("minmax" or "none"), method_sd (a number, or "tune"; default "tune", which falls back to 0 = v1
    when there are no anchors), method_sd_grid."""

    name = "fusion"

    def __init__(self, cfg, task, score_range):
        super().__init__(cfg, task, score_range)
        self.components = cfg.options.get("components") or ["vanilla", "mts", "pairwise"]
        self.vanilla = Vanilla(cfg, task, score_range)
        self.mts = MTS(cfg, task, score_range)
        self.pairwise = Pairwise(cfg, task, score_range)
        self.anchor_labels: dict[int, float] = {}  # set by the runner: essay position -> human score

    def score_set(self, essays, ask, workers=4):
        from .fusion import fit_fusion

        n = len(essays)
        indicators: dict[str, np.ndarray] = {}
        calls, tokens = np.zeros(n), np.zeros(n)
        if "vanilla" in self.components:
            out = Strategy.score_set(self.vanilla, essays, ask, workers)
            indicators["holistic"] = np.array([o["probs"] for o in out], float)
            calls += [o["calls"] for o in out]; tokens += [o["tokens"] for o in out]
        if "mts" in self.components:
            out = Strategy.score_set(self.mts, essays, ask, workers)
            for t in self.mts.traits:
                indicators[t["name"]] = np.array([o["trait_probs"][t["name"]] for o in out], float)
            calls += [o["calls"] for o in out]; tokens += [o["tokens"] for o in out]
        pairs = []
        if "pairwise" in self.components:
            pairs, c, tok = self.pairwise.collect(essays, ask, workers)
            calls += c; tokens += tok
        msd = self.cfg.options.get("method_sd", "tune")
        if msd == "tune" and self.anchor_labels and pairs and indicators:
            from .fusion import METHOD_SD_GRID, tune_method_sd
            idx = sorted(self.anchor_labels)
            grid = tuple(self.cfg.options.get("method_sd_grid", METHOD_SD_GRID))
            fit = tune_method_sd(indicators, pairs, n, idx, [self.anchor_labels[i] for i in idx], grid)
        else:
            fit = fit_fusion(indicators, pairs, n, method_sd=0.0 if msd == "tune" else float(msd))
        theta, se = fit["theta"], fit["se"]
        scaled = self.minmax(theta) if self.cfg.options.get("scale", "minmax") == "minmax" else theta
        return [{"expected": float(scaled[e]), "raw": float(theta[e]), "se": float(se[e]), "mode": float("nan"),
                 "probs": [], "coverage": 1.0, "source": "logprobs", "calls": float(calls[e]),
                 "tokens": float(tokens[e]), "fusion_params": fit["params"] if e == 0 else None}
                for e in range(n)]


class Anchored(Strategy):
    """Adaptive comparisons against human-scored anchor essays (an "anchor ladder").

    Needs anchors (calibrate + anchors in the study config); the runner passes their human scores. Each other
    essay is placed on the ladder of anchor score levels by binary search: it is compared (both orders, same
    prompt as `pairwise`) with an anchor at the middle of its remaining range and moves up or down by the
    answer, so about log2(levels) comparisons reach its level; extra rounds compare it with other anchors at
    the levels it sits between. Anchors are also chained to the next level up. All comparisons go into one
    Bradley-Terry fit over essays and anchors, and the calibration then maps the anchors' strengths to their
    human scores. Unlike essay-vs-essay pairs, every comparison here carries known score information.
    Options: rounds (default 3, about 6 calls per essay with both orders), peer_k (extra essay-vs-essay
    partners, default 0), seed, both_orders (default true)."""

    name = "anchored"

    def __init__(self, cfg, task, score_range):
        super().__init__(cfg, task, score_range)
        self.pairwise = Pairwise(cfg, task, score_range)
        self.anchor_labels: dict[int, float] = {}

    def score_set(self, essays, ask, workers=4):
        if not self.anchor_labels:
            raise ValueError("strategy 'anchored' needs anchors: set calibrate and anchors for it in the config")
        n = len(essays)
        opts = self.cfg.options
        rounds = int(opts.get("rounds", 3))
        both = bool(opts.get("both_orders", True))
        rng = np.random.default_rng(int(opts.get("seed", 1)))
        levels = sorted(set(self.anchor_labels.values()))
        by_level = {lv: [i for i, h in self.anchor_labels.items() if h == lv] for lv in levels}
        targets = [i for i in range(n) if i not in self.anchor_labels]
        calls, tokens = np.zeros(n), np.zeros(n)
        results: dict[tuple[int, int], list[float]] = {}

        def ask_pairs(pairs):
            jobs = [(a, b, False) for a, b in pairs] + ([(b, a, True) for a, b in pairs] if both else [])

            def run(job):
                a, b, _ = job
                r = ask(self.pairwise.messages(essays[a], essays[b]))
                return ab_probability(r), r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)

            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                out = list(pool.map(run, jobs))
            got: dict[tuple[int, int], list[float]] = {}
            for (a, b, flipped), (p, tok) in zip(jobs, out):
                key = (b, a) if flipped else (a, b)
                got.setdefault(key, []).append(1 - p if flipped else p)
                # cost accounting: an essay-vs-anchor call is charged to the essay; anchor-vs-anchor calls
                # are overhead shared by all essays (anchors are not in the test set)
                side = [e for e in (a, b) if e not in self.anchor_labels]
                if side:
                    for e in side:
                        calls[e] += 1 / len(side); tokens[e] += tok / len(side)
                else:
                    calls[targets] += 1 / len(targets); tokens[targets] += tok / len(targets)
            for key, ps in got.items():
                results.setdefault(key, []).extend(ps)
            return {key: float(np.mean(ps)) for key, ps in got.items()}

        used = {i: set() for i in targets}

        def pick(i, lv):
            pool = [a for a in by_level[lv] if a not in used[i]] or by_level[lv]
            a = int(rng.choice(pool)); used[i].add(a)
            return a

        # binary search over the ladder; lo/hi index the levels still possible
        lo = {i: 0 for i in targets}; hi = {i: len(levels) - 1 for i in targets}
        for _ in range(rounds):
            plan = {}
            for i in targets:
                if lo[i] <= hi[i]:
                    mid = (lo[i] + hi[i]) // 2
                else:  # placed between levels hi and lo: refine against one of those, alternating
                    cand = [x for x in (hi[i], lo[i]) if 0 <= x < len(levels)]
                    mid = cand[len(used[i]) % len(cand)]
                plan[i] = (mid, pick(i, levels[mid]))
            got = ask_pairs([(i, a) for i, (_, a) in plan.items()])
            for i, (mid, a) in plan.items():
                if lo[i] > hi[i]:
                    continue
                if got[(i, a)] > 0.5:
                    lo[i] = mid + 1
                else:
                    hi[i] = mid - 1
        # chain anchors to the next level so the anchors themselves are on one scale
        chain = []
        for lv, nxt in zip(levels, levels[1:]):
            for a in by_level[lv]:
                chain.append((a, int(rng.choice(by_level[nxt]))))
        ask_pairs(chain)
        pairs = [(i, j, float(np.mean(ps))) for (i, j), ps in results.items()]
        peer_k = int(opts.get("peer_k", 0))
        if peer_k > 0:
            peer = Pairwise(StrategyConfig(name="pairwise", options={**opts, "k": peer_k}), self.task, (self.lo, self.hi))
            extra, c, tok = peer.collect(essays, ask, workers)
            pairs += extra; calls += c; tokens += tok
        theta = bradley_terry(n, pairs)
        scaled = self.minmax(theta)
        return [{"expected": float(scaled[e]), "raw": float(theta[e]), "mode": float("nan"), "probs": [],
                 "coverage": 1.0, "source": "logprobs", "calls": float(calls[e]), "tokens": float(tokens[e])}
                for e in range(n)]


STRATEGIES = {"vanilla": Vanilla, "mts": MTS, "pairwise": Pairwise, "fusion": Fusion, "anchored": Anchored}


def make_strategy(cfg: StrategyConfig, task: TaskConfig, score_range: tuple[int, int]) -> Strategy:
    if cfg.name not in STRATEGIES:
        raise ValueError(f"Unknown strategy {cfg.name!r}; available: {sorted(STRATEGIES)}")
    return STRATEGIES[cfg.name](cfg, task, score_range)
