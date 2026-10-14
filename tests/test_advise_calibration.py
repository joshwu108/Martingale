"""The advice each committed real-run record would have produced at the default thresholds (docs/advise.md).

Pins the calibration: if a threshold default or the record changes, this table changes with it, on purpose."""
import json
from fractions import Fraction
from pathlib import Path

import pytest
from click.testing import CliRunner

from martingale.cli import cli
from martingale.diagnostics import decompose
from martingale.diagnostics.advise import AdviseConfig, advise_decomposition
from martingale.record import TokenLedger

RESULTS = Path(__file__).resolve().parent.parent / "benchmarks" / "modal" / "results"
TRL_LR = Fraction("5e-6")        # benchmarks/modal/trl_grpo_vllm.py, learning_rate=5e-6
VERL_LR = Fraction("1e-6")       # verl config.json "lr"

# name: (lr, updates per pass, status, passing interval, first failing lag)
EXPECTED = {
    "trl_grpo_vllm_a10g_16steps_seed0_t1.0": (TRL_LR, 2, "engine_floor", None, 0),
    "trl_grpo_vllm_a10g_16steps_seed0_t1.0_fp32_probe": (TRL_LR, 2, "engine_floor", None, 0),
    "trl_grpo_vllm_a10g_16steps_seed0_t1.0_probe": (TRL_LR, 2, "within_observed", 4, None),
    "trl_grpo_vllm_a10g_16steps_seed0_t1.0_replay1of4_replay": (TRL_LR, 2, "reduce", 2, 2),
    "verl_grpo_vllm_t4_8steps_seed0": (VERL_LR, 2, "within_observed", 4, None),
}


def _advice(name, cfg=AdviseConfig(), **kw):
    lr, per_pass, *_ = EXPECTED[name]
    ledger = TokenLedger(RESULTS / name / "tokens.db", readonly=True)
    try:
        return advise_decomposition(decompose(ledger), cfg, lr=lr, updates_per_pass=per_pass, **kw)
    finally:
        ledger.close()


def test_every_committed_record_is_calibrated():
    assert sorted(p.parent.name for p in RESULTS.glob("*/tokens.db")) == sorted(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_default_advice_per_record(name):
    lr, _, status, passing, failing = EXPECTED[name]
    a = _advice(name)
    assert (a.status, a.passing_interval, a.first_failing_lag) == (status, passing, failing)
    assert a.observed_interval == 4 and a.rows_source == "fresh"
    if passing is not None:
        assert a.lr_ceiling == lr * passing / 4


def test_replay_record_numbers_behind_the_advice():
    a = _advice("trl_grpo_vllm_a10g_16steps_seed0_t1.0_replay1of4_replay", target_interval=8)
    assert a.lr_ceiling == Fraction(1, 400000) and a.lr_failing == TRL_LR * 3 / 4
    assert a.lr_ceiling_at_target == TRL_LR * 2 / 8
    assert a.knobs["max_iterations"] == 1 and a.knobs["max_async_level"] == 0
    lag2 = next(r for r in a.rows if r.lag == 2)
    assert lag2.n_tokens == 104 and lag2.ess_fraction == pytest.approx(0.351, abs=1e-3)


def test_threshold_separates_healthy_from_pathological_buckets():
    """Every qualifying bucket of the healthy records clears 0.9; the two pathological ones sit below 0.4."""
    healthy = [r.ess_fraction for n in ("trl_grpo_vllm_a10g_16steps_seed0_t1.0_probe", "verl_grpo_vllm_t4_8steps_seed0")
               for r in _advice(n).rows if r.qualifies]
    assert min(healthy) > 0.9
    bad = [next(r for r in _advice(n).rows if r.lag == lag).ess_fraction for n, lag in (
        ("trl_grpo_vllm_a10g_16steps_seed0_t1.0", 0), ("trl_grpo_vllm_a10g_16steps_seed0_t1.0_replay1of4_replay", 2))]
    assert max(bad) < 0.4 < AdviseConfig().ess_min < 0.9


def test_cli_advise_on_a_committed_record(tmp_path):
    ws = RESULTS / "trl_grpo_vllm_a10g_16steps_seed0_t1.0_replay1of4_replay"
    before = (ws / "tokens.db").read_bytes()
    out = tmp_path / "d.json"
    r = CliRunner().invoke(cli, ["doctor", "--dir", str(ws), "--advise", "--lr", "5e-6", "--updates-per-pass", "2",
                                 "--target-interval", "8", "--json", str(out)])
    assert r.exit_code == 0, r.output
    assert "Suggestion, not a setting" in r.output and "arXiv:2607.01083" in r.output and "S = 2" in r.output
    advice = json.loads(out.read_text())["advice"]
    assert advice["status"] == "reduce" and advice["lr_ceiling"] == "1/400000"
    assert (ws / "tokens.db").read_bytes() == before


def test_cli_without_advise_prints_no_advice():
    r = CliRunner().invoke(cli, ["doctor", "--dir", str(RESULTS / "verl_grpo_vllm_t4_8steps_seed0")])
    assert r.exit_code == 0 and "Suggestion, not a setting" not in r.output


@pytest.mark.parametrize("lr", ["fast", "1/0", "0", "-1e-6"])
def test_cli_rejects_a_bad_lr(lr):
    r = CliRunner().invoke(cli, ["doctor", "--dir", str(RESULTS / "verl_grpo_vllm_t4_8steps_seed0"), "--advise",
                                 "--lr", lr])
    assert r.exit_code == 2 and "--lr" in r.output, r.output


@pytest.mark.parametrize("args", [["--lr", "5e-6"], ["--advise", "--target-interval", "8"]])
def test_cli_rejects_advice_options_that_would_be_ignored(args):
    r = CliRunner().invoke(cli, ["doctor", "--dir", str(RESULTS / "verl_grpo_vllm_t4_8steps_seed0"), *args])
    assert r.exit_code == 2 and "need" in r.output
