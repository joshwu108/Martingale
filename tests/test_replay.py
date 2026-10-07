"""Replaying the live monitor over a finished record gives the alarms the trainer would have raised."""
from fractions import Fraction

from martingale.diagnostics import AlarmConfig, decompose
from martingale.diagnostics.replay import replay
from martingale.record import TokenLedger
from tests.conftest import REAL_RECORD, needs_real_record


def test_replay_demo_is_one_step_whose_cumulative_equals_decompose(demo_ws):
    ledger = TokenLedger(demo_ws / "tokens.db")
    steps = replay(ledger)
    assert len(steps) == 1
    st = steps[0]
    assert st.diagnosis.step == 5 and st.diagnosis.n_new_scores == 384 and st.diagnosis.n_new_sequences == 48
    d = decompose(ledger)
    assert [(b["lag"], b["n_tokens"], b["mean_abs_log_ratio"]) for b in st.diagnosis.cumulative_by_lag] == \
        [(b["lag"], b["n_tokens"], b["mean_abs_log_ratio"]) for b in d["by_lag"]]
    assert isinstance(st.alarms, list)


@needs_real_record
def test_replay_real_record_raises_stale_server_at_the_generations_the_engine_served_old_weights():
    ledger = TokenLedger(REAL_RECORD / "tokens.db")
    steps = replay(ledger)
    assert [s.diagnosis.step for s in steps] == list(range(1, 17))
    assert [s.diagnosis.n_new_scores for s in steps] == [34, 30, 34, 30, 27, 29, 27, 29, 27, 29, 27, 29, 32, 32, 32, 32]
    assert [s.diagnosis.n_new_sequences for s in steps] == [16, 0, 0, 0] * 4
    stale = {s.diagnosis.step: [a for a in s.alarms if a.kind == "stale_server"] for s in steps}
    assert {k for k, v in stale.items() if v} == {5, 9}
    assert stale[5][0].evidence["n_confident_disagreements"] == 3
    assert len(steps[0].diagnosis.new_weights_digests) == 1
    last = steps[-1].diagnosis.cumulative_by_lag
    d = decompose(ledger)
    assert [b["n_tokens"] for b in last] == [120] * 4
    assert [Fraction(b["mean_abs_log_ratio"]) for b in last] == [Fraction(b["mean_abs_log_ratio"]) for b in d["by_lag"]]


@needs_real_record
def test_replay_honours_alarm_config():
    ledger = TokenLedger(REAL_RECORD / "tokens.db")
    steps = replay(ledger, AlarmConfig(confident_min_tokens=10))
    assert not any(a.kind == "stale_server" for s in steps for a in s.alarms)
