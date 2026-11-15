# llm-scoring-bench

Test any LLM on automated essay scoring with one config file. Write a YAML file that names the essays,
the rubric, the models and the prompt strategies; run one command; get every model and strategy scored on
the same essays, with the metrics operational scoring programs use.

```bash
pip install -e .
export TYK_KEY=...            # or OPENAI_API_KEY, or whatever api_key_env names
scorebench run examples/asap_set1.yaml --sample 20    # quick check
scorebench run examples/asap_set1.yaml                 # full run
```

Output (in `output_dir/<name>/`): `metrics.csv`, `predictions.csv` (one row per essay x model x strategy,
with the full score distribution), `report.html`, and `run.json` (settings, human-human benchmark).

## What it does

| Stage | Now | Next |
| --- | --- | --- |
| Data | ASAP-AES (per prompt, two raters), PERSUADE 2.0, any CSV/TSV/parquet | ASAP++ traits |
| Models | Any OpenAI-compatible endpoint (Alliance Vulcan, vLLM, OpenAI, Ollama); retries while a model spins up | Local Hugging Face |
| Scores | Reads the score distribution from first-token log-probabilities; keeps the expected score and the most likely score; falls back to the reply text | |
| Strategies | `vanilla` (rubric + essay) | MTS trait prompting, LCES pairwise, few-shot, the fusion model |
| Calibration | Quantile (equipercentile) linking on a few scored anchor essays | |
| Metrics | QWK (vs rater 1 and 2), PRMSE, exact/adjacent agreement, r, mean difference, SMD overall and by subgroup, SD ratio, calls and tokens per essay | Rater-effect diagnostics |
| Cache | Every reply is cached in SQLite by a hash of model, prompt and settings, so reruns are free | |

PRMSE follows Haberman (2008) and Loukina et al. (2020), as in ETS's rsmtool. It needs two human scores per
essay, so it is reported for ASAP but not for PERSUADE (one final score).

## Config

See `examples/asap_set1.yaml`. Keys: `data` (loader, path, filter, sample, seed, groups), `task`
(`prompt_text` / `rubric`, or `*_file`), `models` (name, base_url, api_key_env, extra_body...), `strategies`
(a name, or `{name, calibrate, anchors}`), `output_dir`, `cache`, `workers`.

For Qwen models on Vulcan, pass `extra_body: {reasoning_effort: none}` so the model answers without hidden
reasoning tokens.

## Tests

`pytest` runs without network: a fake backend acts as a harsh, compressed rater on synthetic essays.

## License

MIT.
