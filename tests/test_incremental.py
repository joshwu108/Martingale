"""Incremental diagnosis must equal the full decomposition exactly, step by step."""
from fractions import Fraction

from martingale.demo import run_demo
from martingale.diagnostics import IncrementalDiagnosis, RunningBucket, decompose
from martingale.record import Recorder, SamplerConfig

S = SamplerConfig(temperature=1.0)
TOK = "f" * 64


def test_incremental_cumulative_equals_full_decompose_on_demo_record(tmp_path):
    run_demo(tmp_path / "ws", echo=lambda *_: None)
    rec = Recorder(tmp_path / "ws")
    inc = IncrementalDiagnosis(rec.ledger)
    d = inc.update(step=0)
    full = decompose(rec.ledger)
    assert [b["lag"] for b in d.cumulative_by_lag] == full["lags"]
    for a, b in zip(d.cumulative_by_lag, full["by_lag"]):
        for k in ("n_tokens", "n_sequences", "n_mixed_sequences", "mean_log_ratio", "mean_abs_log_ratio", "max_abs_log_ratio"):
            assert a[k] == b[k], k
        assert abs(a["float_informational"]["ess"] - b["float_informational"]["ess"]) < 1e-9
    assert d.n_new_scores == full["n_scored_tokens"]
    # a second update with nothing new is empty but keeps the cumulative view
    d2 = inc.update(step=1)
    assert d2.n_new_scores == 0 and d2.by_lag == [] and d2.cumulative_by_lag == d.cumulative_by_lag


def test_step_buckets_cover_only_new_scores(tmp_path):
    rec = Recorder(tmp_path / "ws")
    r0 = rec.publish_revision("1" * 64, TOK, S, step=0)
    r2 = rec.publish_revision("2" * 64, TOK, S, step=2)
    inc = IncrementalDiagnosis(rec.ledger)
    with rec.sequence(0, "a", [1]) as a:
        a.token(r0.digest, 1, -1.0); a.token(r0.digest, 2, -1.0)
    rec.score(a.record.digest, r2.digest, [-1.5, -0.5])
    d1 = inc.update(step=2)
    assert d1.n_new_sequences == 1 and d1.n_new_scores == 2 and d1.new_weights_digests == ["1" * 64]
    assert d1.by_lag[0]["lag"] == 2 and d1.by_lag[0]["mean_abs_log_ratio"] == "1/2"
    with rec.sequence(0, "b", [1]) as b:
        b.token(r2.digest, 1, -1.0)
    rec.score(b.record.digest, r2.digest, [-1.0])
    d2 = inc.update(step=3)
    assert [x["lag"] for x in d2.by_lag] == [0] and d2.floor_mean_abs == Fraction(0)
    assert [x["lag"] for x in d2.cumulative_by_lag] == [0, 2]
    assert d2.unscored_new_sequences == 0


def test_reservoir_is_bounded_and_deterministic():
    a, b = RunningBucket(1, size=8), RunningBucket(1, size=8)
    for i in range(100):
        for bk in (a, b):
            bk.add("d" * 64, i, Fraction(i, 100), False, (0.2,))
    assert len(a.reservoir) == 8 and a.reservoir == b.reservoir
    assert a.summary()["float_informational"]["percentiles_from_reservoir_of"] == 8
    assert a.n_tokens == 100 and a.sum_abs_log_ratio == sum(Fraction(i, 100) for i in range(100))
