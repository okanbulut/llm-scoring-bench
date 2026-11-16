"""Supervised baseline: fine-tune a Hugging Face encoder (e.g. UltraBERT) on the anchors' human scores.

Regression head on the standardised score (MSE), AdamW with linear warm-up/decay, then predict every essay.
Optional dependencies: torch and transformers. Device: "cuda", "mps" or "cpu" (default: best available).
"""

from __future__ import annotations

import numpy as np


def _device(pref=None):
    import torch

    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def finetune_predict(train_texts, train_y, all_texts, model_name, epochs=10, lr=3e-5, batch_size=8,
                     max_length=512, weight_decay=0.01, warmup=0.1, seed=1, device=None,
                     trust_remote_code=False, grad_accum=1) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    dev = _device(device)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, problem_type="regression", trust_remote_code=trust_remote_code).to(dev)
    y = np.asarray(train_y, float)
    mu, sd = y.mean(), y.std() + 1e-9
    yz = torch.tensor((y - mu) / sd, dtype=torch.float32)
    enc = tok(list(train_texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    n = len(y)
    steps = max(1, epochs * int(np.ceil(n / batch_size)) // grad_accum)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)
    rng = np.random.default_rng(seed)
    model.train()
    k = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            ix = torch.tensor(order[s:s + batch_size])
            batch = {kk: v[ix].to(dev) for kk, v in enc.items()}
            out = model(**batch)
            loss = torch.nn.functional.mse_loss(out.logits.squeeze(-1), yz[ix].to(dev)) / grad_accum
            loss.backward()
            k += 1
            if k % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(all_texts), 16):
            e = tok(list(all_texts[s:s + 16]), truncation=True, max_length=max_length, padding=True,
                    return_tensors="pt").to(dev)
            preds.append(model(**e).logits.squeeze(-1).float().cpu().numpy())
    del model
    if dev == "mps":
        torch.mps.empty_cache()
    return np.concatenate(preds) * sd + mu
