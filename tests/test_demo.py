"""The demo must run end to end, verify clean, show lag buckets, and be fast."""
import time

from click.testing import CliRunner

from martingale.cli import cli
from martingale.demo import run_demo


def test_demo_runs_verifies_and_reports(tmp_path):
    t0 = time.time()
    out = run_demo(tmp_path / "ws", echo=lambda *_: None)
    assert out["verify"]["ok"]
    assert set(out["report"]["staleness_histogram"]) >= {"0", "2", "3", "4"}
    assert out["report"]["lag0_floor"]["float_informational"]["max_abs_log_ratio"] < 1e-4
    assert out["bench_seq_is"]["all_unbiased"] and not out["bench_token_ppo_clip"]["all_unbiased"]
    assert time.time() - t0 < 60


def test_demo_cli_keeps_workspace_and_verify_passes(tmp_path):
    r = CliRunner().invoke(cli, ["demo", "--dir", str(tmp_path / "ws")])
    assert r.exit_code == 0, r.output
    assert "independent checker: ok=True" in r.output
    v = CliRunner().invoke(cli, ["verify", "--dir", str(tmp_path / "ws")])
    assert v.exit_code == 0, v.output and "all sequences verified clean" in v.output


def test_demo_refuses_non_empty_workspace(tmp_path):
    (tmp_path / "ws").mkdir(); (tmp_path / "ws" / "x").write_text("x")
    import pytest
    with pytest.raises(FileExistsError):
        run_demo(tmp_path / "ws", echo=lambda *_: None)


def test_demo_head_is_deterministic_across_runs(tmp_path):
    a = run_demo(tmp_path / "a", echo=lambda *_: None)["head"]
    b = run_demo(tmp_path / "b", echo=lambda *_: None)["head"]
    assert a == b
