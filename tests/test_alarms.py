"""Each alarm fires exactly once on a record built to trigger it, and not otherwise."""
from fractions import Fraction

import pytest

from martingale.diagnostics import AlarmConfig, AlarmHistory, IncrementalDiagnosis, evaluate
from martingale.diagnostics.incremental import StepDiagnosis
from martingale.record import Recorder, SamplerConfig

S = SamplerConfig(temperature=1.0)
TOK = "f" * 64
CFG = AlarmConfig(min_floor_tokens=2)


def _diag(step, by_lag=(), floor=None, weights=(), span=0.0, neg=0, n_seq=1):
    return StepDiagnosis(step=step, n_new_scores=sum(b["n_tokens"] for b in by_lag), n_new_sequences=n_seq,
                         by_lag=list(by_lag), cumulative_by_lag=[], attribution={"float_informational": {}},
                         floor_mean_abs=None if floor is None else Fraction(floor), new_weights_digests=list(weights),
                         span_fraction=span, n_negative_lag=neg, unscored_new_sequences=0)


def _bucket(lag, n, ess_fraction, p50=1.0):
    return {"lag": lag, "n_tokens": n, "float_informational": {"ess_fraction": ess_fraction, "ratio_p50": p50}}


class TestAlarms:
    def test_quiet_step_has_no_alarms(self):
        h = AlarmHistory()
        assert evaluate(_diag(1, [_bucket(0, 50, 1.0), _bucket(1, 50, 0.9)], floor=1e-6, weights=["a" * 64]), h, CFG) == []

    def test_negative_lag(self):
        a = evaluate(_diag(1, neg=3), AlarmHistory(), CFG)
        assert [x.kind for x in a] == ["negative_lag"] and a[0].level == "error"

    def test_weights_unchanged_only_across_an_optimizer_step(self):
        h = AlarmHistory()
        assert evaluate(_diag(1, weights=["a" * 64]), h, CFG) == []
        assert evaluate(_diag(1, weights=["a" * 64]), h, CFG) == []          # same step: no optimizer step between
        a = evaluate(_diag(2, weights=["a" * 64]), h, CFG)
        assert [x.kind for x in a] == ["weights_unchanged"]
        assert evaluate(_diag(3, weights=["b" * 64]), h, CFG) == []

    def test_floor_jump_and_stale_server_proxy(self):
        h = AlarmHistory()
        for step in range(1, 4):
            assert evaluate(_diag(step, [_bucket(0, 10, 1.0)], floor=1e-6), h, CFG) == []
        a = evaluate(_diag(4, [_bucket(0, 10, 1.0)], floor=5e-6), h, CFG)            # 5x, no new generation
        assert [x.kind for x in a] == ["floor_jump"] and a[0].level == "warn"
        a = evaluate(_diag(5, [_bucket(0, 10, 1.0, p50=1.0007)], floor=1e-4, weights=["c" * 64]), h, CFG)   # 50x, median at 1
        assert [x.kind for x in a] == ["floor_tail"] and a[0].level == "warn" and "Not a stale server" in a[0].message
        h2 = AlarmHistory()
        for step in range(1, 4):
            evaluate(_diag(step, [_bucket(0, 10, 1.0)], floor=1e-6), h2, CFG)
        a = evaluate(_diag(5, [_bucket(0, 10, 1.0, p50=0.8)], floor=1e-4, weights=["c" * 64]), h2, CFG)     # median moved
        assert [x.kind for x in a] == ["stale_server"] and a[0].level == "error" and "PROXY" in a[0].message

    def test_floor_needs_enough_tokens(self):
        h = AlarmHistory()
        for step in range(1, 4):
            evaluate(_diag(step, [_bucket(0, 10, 1.0)], floor=1e-6), h, CFG)
        assert evaluate(_diag(4, [_bucket(0, 1, 1.0)], floor=1e-2), h, CFG) == []

    def test_ess_collapse_levels_and_token_share(self):
        a = evaluate(_diag(1, [_bucket(2, 100, 0.2)]), AlarmHistory(), CFG)
        assert [(x.kind, x.level) for x in a] == [("ess_collapse", "warn")]
        a = evaluate(_diag(1, [_bucket(2, 100, 0.05)]), AlarmHistory(), CFG)
        assert [(x.kind, x.level) for x in a] == [("ess_collapse", "error")]
        assert evaluate(_diag(1, [_bucket(1, 95, 1.0), _bucket(4, 5, 0.01)]), AlarmHistory(), CFG) == []   # < 10% share

    def test_span_and_unmatched(self):
        a = evaluate(_diag(1, span=0.2, n_seq=10), AlarmHistory(), CFG)
        assert [x.kind for x in a] == ["span"]
        a = evaluate(_diag(1), AlarmHistory(), CFG, unmatched_rows=5, matched_rows=95)
        assert [x.kind for x in a] == ["unmatched"] and a[0].level == "error"
        assert evaluate(_diag(1), AlarmHistory(), CFG, unmatched_rows=0, matched_rows=95) == []


def test_alarms_from_a_real_record(tmp_path):
    """negative lag detected from the ledger itself through IncrementalDiagnosis."""
    rec = Recorder(tmp_path / "ws")
    r1 = rec.publish_revision("1" * 64, TOK, S, step=1)
    r5 = rec.publish_revision("2" * 64, TOK, S, step=5)
    with rec.sequence(0, "b", [1]) as b:
        b.token(r5.digest, 5, -0.5)
    rec.score(b.record.digest, r1.digest, [-0.5])
    d = IncrementalDiagnosis(rec.ledger).update(step=5)
    assert [a.kind for a in evaluate(d, AlarmHistory(), CFG)] == ["negative_lag"]
