from scoringbench.cli import main

from test_end_to_end import _synthetic


def test_cli_run(tmp_path, capsys):
    main(["run", str(_synthetic(tmp_path)), "--out", str(tmp_path / "cli"), "--sample", "60"])
    out = capsys.readouterr().out
    assert "vanilla+quantile30" in out and (tmp_path / "cli" / "report.html").exists()
