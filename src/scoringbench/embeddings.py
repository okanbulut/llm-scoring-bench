"""Essay embeddings for the hybrid strategy, cached like model replies.

Backends (strategy option `embed: {backend, model, ...}`):
- openai: any OpenAI-compatible /v1/embeddings endpoint (base_url, api_key_env / api_key_file as for models).
- sentence-transformers: a local model or any Hugging Face encoder id/path (pip install sentence-transformers);
  options max_seq_length (essays are long; default is the model's), trust_remote_code, device.
- fake: deterministic vectors for offline tests; reads "[true=k]" from the text and adds noise.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import numpy as np

from .cache import Cache, request_key
from .config import ModelConfig


def _fake(texts, dim=32):
    out = []
    for t in texts:
        seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        v = rng.normal(0, 1, dim)
        m = re.search(r"\[true=(\d+)\]", t)
        if m:
            v[:4] += 0.8 * (int(m.group(1)) - 3.5)  # a few dimensions carry the score
        out.append(v)
    return np.array(out)


def embed_texts(texts: list[str], spec: dict, cache: Cache | None = None, base_dir: Path = Path("."),
                batch: int = 32) -> np.ndarray:
    backend = spec.get("backend", "fake")
    model = spec.get("model", "fake")
    keys = [request_key({"embed": model, "backend": backend, "base_url": spec.get("base_url"), "text": t})
            for t in texts]
    vecs: list = [None] * len(texts)
    if cache is not None:
        for i, k in enumerate(keys):
            hit = cache.get(k)
            if hit is not None:
                vecs[i] = hit["embedding"]
    todo = [i for i, v in enumerate(vecs) if v is None]
    if todo:
        if backend == "fake":
            new = _fake([texts[i] for i in todo])
        elif backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            # any Hugging Face encoder works: sentence-transformers adds mean pooling when the model has none
            st = SentenceTransformer(model, trust_remote_code=bool(spec.get("trust_remote_code", False)),
                                     device=spec.get("device"))
            if spec.get("max_seq_length"):
                st.max_seq_length = int(spec["max_seq_length"])
            new = st.encode([texts[i] for i in todo], batch_size=batch, normalize_embeddings=True)
        elif backend == "openai":
            from openai import OpenAI

            from .backends import resolve_api_key

            m = ModelConfig(name=model, base_url=spec.get("base_url"),
                            api_key_env=spec.get("api_key_env", "OPENAI_API_KEY"),
                            api_key_file=spec.get("api_key_file"))
            client = OpenAI(api_key=resolve_api_key(m, base_dir), base_url=m.base_url, timeout=300)
            new = []
            for s in range(0, len(todo), batch):
                chunk = [texts[i] for i in todo[s:s + batch]]
                for attempt in range(6):
                    try:
                        resp = client.embeddings.create(model=model, input=chunk)
                        break
                    except Exception:
                        if attempt == 5:
                            raise
                        time.sleep(5 * 2 ** attempt)
                new.extend(d.embedding for d in resp.data)
            new = np.array(new)
        else:
            raise ValueError(f"Unknown embedding backend {backend!r}")
        for i, v in zip(todo, np.asarray(new, float)):
            vecs[i] = v.tolist()
            if cache is not None:
                cache.put(keys[i], {"embedding": vecs[i]})
    return np.array(vecs, float)


def ridge_loo(X: np.ndarray, y: np.ndarray, alphas=(0.1, 1, 10, 100, 1000, 1e4)) -> dict:
    """Ridge regression of y on X (centred), alpha chosen by exact leave-one-out error.
    Works in the n x n kernel form, so it is cheap for few anchors and wide embeddings.
    Returns alpha, out-of-fold predictions for the training rows, and a predict(X_new) function."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    mx, my = X.mean(0), y.mean()
    Xc, yc = X - mx, y - my
    K = Xc @ Xc.T
    n = len(y)
    best = None
    for a in alphas:
        Hinv = np.linalg.inv(K + a * np.eye(n))
        H = K @ Hinv  # hat matrix of the centred problem; the unpenalised intercept adds 1/n to the leverage
        fit = H @ yc
        loo = yc - (yc - fit) / np.maximum(1 - np.diag(H) - 1 / n, 1e-9)
        err = float(np.mean((yc - loo) ** 2))
        if best is None or err < best[0]:
            best = (err, a, Hinv, loo)
    _, a, Hinv, loo = best
    dual = Hinv @ yc
    w = Xc.T @ dual
    return {"alpha": a, "oof": loo + my, "loo_mse": best[0], "predict": lambda Z: (np.asarray(Z, float) - mx) @ w + my}
