"""The acceptance runner's synthetic replay: earlier completions re-injected with full provenance (CPU, fake trainer)."""
import importlib.util
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from checker.verify_tokens import verify_export
from martingale.diagnostics import decompose
from martingale.integrations.trl import MartingaleRecorder
from tests.test_trl_integration import FakeTrainer, make_output

_SPEC = importlib.util.spec_from_file_location(
    "replay_inject", Path(__file__).resolve().parent.parent / "benchmarks" / "modal" / "replay_inject.py")
replay_inject = importlib.util.module_from_spec(_SPEC)
sys.modules["replay_inject"] = replay_inject          # dataclasses resolve annotations through sys.modules
_SPEC.loader.exec_module(replay_inject)
SyntheticReplay = replay_inject.SyntheticReplay


def _fake_logps(ids: list[int]) -> list[float]:
    """FakeTrainer._get_per_token_logps_and_entropies, in float32 as the trainer computes it."""
    return (-0.5 - 0.01 * torch.tensor(ids).float()).tolist()


@pytest.fixture
def flight(tmp_path):
    return MartingaleRecorder(tmp_path / "ws", tokenizer=b"t")


def test_first_generation_only_fills_the_pool(flight):
    rp = SyntheticReplay(flight, Fraction(1, 2), seed=0)
    tr = FakeTrainer()
    out = flight.on_generation(make_output(), tr)
    new = rp.after_generation(out, tr)
    assert new is out and rp.pool_size == 2 and rp.stats["replaced_rows"] == 0


def test_a_fraction_of_rows_is_replaced_with_earlier_completions_and_registered(flight, tmp_path):
    rp = SyntheticReplay(flight, Fraction(1, 2), seed=0)
    tr = FakeTrainer()
    out0 = flight.on_generation(make_output(), tr)
    rp.after_generation(out0, tr)
    tr.state.global_step = 2
    with torch.no_grad():
        tr.model.weight.add_(1.0)
    out2 = flight.on_generation(make_output(), tr)
    new = rp.after_generation(out2, tr)
    assert rp.stats["replaced_rows"] == 1 and flight.stats["replayed_sequences"] == 1
    (r,) = rp.last_replaced
    rid = int(new["martingale_row_id"][r])
    seq = flight.recorder.ledger.get_sequence(flight.sequence_digest(rid))
    assert seq.provenance == "replayed" and seq.replay.draw_id == f"2/{r}"
    origin = flight.recorder.ledger.get_sequence(seq.replay.origin_digest)
    assert origin.sequence_id.startswith("step0/row")
    # the batch row now carries the origin's tokens, behaviour log-probs and advantage, re-padded
    comp = [t.token_id for t in origin.tokens]
    assert new["completion_ids"][r, :len(comp)].tolist() == comp
    assert new["completion_mask"][r].tolist() == [1] * len(comp) + [0] * (new["completion_ids"].size(1) - len(comp))
    assert new["prompt_ids"][r, -origin.prompt_len:].tolist() == list(origin.prompt_ids)
    assert new["prompt_mask"][r, -origin.prompt_len:].tolist() == [1] * origin.prompt_len
    old = new["old_per_token_logps"][r, :len(comp)].tolist()
    from martingale.record import bits_to_fraction
    assert old == [float(bits_to_fraction(t.logprob_bits)) for t in origin.tokens]
    assert float(new["advantages"][r]) == float(bits_to_fraction(seq.reward_bits))
    # the declared IS weight is exp(sum(current - behaviour)) from a no-grad forward, stored exactly
    expected = math.exp(sum(c - b for c, b in zip(_fake_logps(comp), old)))
    assert seq.replay.is_weight == Fraction(expected) and seq.replay.rescale == 1
    # untouched rows are byte-identical to what TRL produced
    for k in ("prompt_ids", "completion_ids", "advantages", "martingale_row_id"):
        keep = [i for i in range(2) if i != r]
        assert torch.equal(new[k][keep][:, -out2[k].size(1):] if new[k].dim() == 2 else new[k][keep], out2[k][keep])
    # the loss path scores the replayed row at lag 2 and the checker accepts the record
    ids = torch.cat([new["prompt_ids"], new["completion_ids"]], dim=1)
    am = torch.cat([new["prompt_mask"], new["completion_mask"]], dim=1)
    lp = torch.zeros_like(new["completion_ids"], dtype=torch.float32)
    assert flight.on_scores(ids, am, new["completion_ids"].size(1), lp, tr, row_ids=new["martingale_row_id"]) == 2
    d = decompose(flight.recorder.ledger)
    assert [b["lag"] for b in d["by_provenance"]["replayed"]["by_lag"]] == [2]
    assert d["replayed"]["sequence_weights"]["n"] == 1 and d["replayed"]["n_without_origin"] == 0
    report = verify_export(flight.recorder.export_for_checker(tmp_path / "exp"), expected_head=flight.head())
    assert report["ok"] and report["n_replayed_sequences"] == 1


def test_replacement_is_deterministic_in_seed_and_step(tmp_path):
    picks = []
    for _ in range(2):
        flight = MartingaleRecorder(tmp_path / f"ws{len(picks)}", tokenizer=b"t")
        rp = SyntheticReplay(flight, Fraction(1, 2), seed=7)
        tr = FakeTrainer()
        rp.after_generation(flight.on_generation(make_output(), tr), tr)
        tr.state.global_step = 1
        rp.after_generation(flight.on_generation(make_output(), tr), tr)
        picks.append((tuple(rp.last_replaced), tuple(rp.stats["draw_ids"])))
    assert picks[0] == picks[1]


def test_fraction_zero_leaves_the_batch_alone(flight):
    rp = SyntheticReplay(flight, Fraction(0), seed=0)
    tr = FakeTrainer()
    rp.after_generation(flight.on_generation(make_output(), tr), tr)
    tr.state.global_step = 1
    out = flight.on_generation(make_output(), tr)
    assert rp.after_generation(out, tr) is out and rp.stats["replaced_rows"] == 0


def test_unregistrable_origin_marks_the_row_minus_one(flight, monkeypatch):
    rp = SyntheticReplay(flight, Fraction(1), seed=0)
    tr = FakeTrainer()
    rp.after_generation(flight.on_generation(make_output(), tr), tr)
    tr.state.global_step = 1
    from martingale.integrations.trl import OriginNotFound

    def boom(*a, **k):
        raise OriginNotFound("gone")
    monkeypatch.setattr(flight, "register_replayed", boom)
    new = rp.after_generation(flight.on_generation(make_output(), tr), tr)
    assert new["martingale_row_id"].tolist() == [-1, -1] and rp.stats["unregistered_rows"] == 2
