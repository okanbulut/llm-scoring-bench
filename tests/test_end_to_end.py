import numpy as np
import pandas as pd

from scoringbench.config import load_config
from scoringbench.runner import run_study


def _synthetic(tmp_path, n=150):
    rng = np.random.default_rng(3)
    true = rng.integers(1, 7, n)
    h1 = np.clip(true + rng.integers(-1, 2, n), 1, 6)
    h2 = np.clip(true + rng.integers(-1, 2, n), 1, 6)
    ell = rng.choice(["Yes", "No"], n)
    text = [f"Essay number {i}. [true={t}] Some words here." for i, t in enumerate(true)]
    pd.DataFrame({"id": range(n), "essay": text, "r1": h1, "r2": h2, "ell": ell}).to_csv(tmp_path / "d.csv", index=False)
    (tmp_path / "study.yaml").write_text(f"""
name: fake-study
data:
  loader: table
  path: {tmp_path / 'd.csv'}
  score_range: [1, 6]
  columns: {{id: id, text: essay, human: [r1, r2]}}
  groups: [ell]
task:
  rubric: "6 = excellent ... 1 = very weak"
models:
  - {{name: fake-harsh, backend: fake}}
strategies:
  - vanilla
  - {{name: vanilla, calibrate: quantile, anchors: 30}}
cache: {tmp_path / 'cache.sqlite'}
workers: 2
""")
    return tmp_path / "study.yaml"


def test_run_study_with_fake_backend(tmp_path):
    cfg = load_config(_synthetic(tmp_path))
    res = run_study(cfg, tmp_path / "out", progress=False)
    m = res["metrics"].set_index("strategy")
    assert set(m.index) == {"vanilla", "vanilla+quantile30"}
    raw, cal = m.loc["vanilla"], m.loc["vanilla+quantile30"]
    assert raw["mean_diff"] < -0.5  # the fake rater is harsh
    assert abs(cal["mean_diff"]) < abs(raw["mean_diff"])  # calibration removes most of it
    assert cal["n"] == 120 and raw["n"] == 150
    assert 0.75 < cal["sd_ratio"] < 1.3  # representative anchors keep the human spread
    assert 0 < raw["prmse"] <= 1 and "smd[ell=Yes]" in m.columns
    assert raw["logprob_share"] == 1.0
    for f in ("predictions.csv", "metrics.csv", "report.html", "run.json"):
        assert (tmp_path / "out" / f).exists()
    # second run is served from the cache and gives identical numbers
    again = run_study(cfg, tmp_path / "out2", progress=False)["metrics"].set_index("strategy")
    assert again.loc["vanilla", "qwk"] == raw["qwk"]
