"""A replay buffer registers replayed rows with the TRL flight recorder; the loss path scores them by row id."""
from fractions import Fraction

import pytest

torch = pytest.importorskip("torch")

from checker.verify_tokens import verify_export
from martingale.diagnostics import decompose
from martingale.integrations.trl import AmbiguousOrigin, MartingaleRecorder, OriginNotFound
from tests.test_trl_integration import FakeTrainer, make_output

CD = "c" * 64


@pytest.fixture
def flight(tmp_path):
    return MartingaleRecorder(tmp_path / "ws", tokenizer=b"{tokenizer}")


def _loss_batch(out):
    ids = torch.cat([out["prompt_ids"], out["completion_ids"]], dim=1)
    am = torch.cat([out["prompt_mask"], out["completion_mask"]], dim=1)
    return ids, am, out["completion_ids"].size(1)


def test_register_replayed_finds_the_fresh_origin_and_scores_by_row_id(flight, tmp_path):
    tr = FakeTrainer()
    flight.on_generation(make_output(), tr)                            # step 0: rows 0 and 1 are fresh
    origin = flight.sequence_digest(1)
    assert origin is not None and flight.sequence_digest(7) is None
    tr.state.global_step = 3
    with torch.no_grad():
        tr.model.weight.add_(1.0)
    flight.on_generation(make_output(), tr)                            # step 3: a new fresh batch (rows 2, 3)
    rid = flight.register_replayed(tr, prompt_ids=[7, 8, 9], completion_ids=[13, 14, 15], behavior_step=0,
                                   draw_id="3/0", content_digest=CD, is_weight=Fraction(1, 2), rescale=Fraction(1, 1),
                                   advantage=0.75)
    assert rid == 4
    seq = flight.recorder.ledger.get_sequence(flight.sequence_digest(rid))
    assert seq.provenance == "replayed" and seq.replay.origin_digest == origin and seq.replay.draw_id == "3/0"
    assert [t.token_id for t in seq.tokens] == [13, 14, 15] and seq.sequence_id == "step3/replay4"
    assert flight.stats["replayed_sequences"] == 1 and flight.stats["replayed_tokens"] == 3
    # the loss batch: a fresh row (id 2), the replayed row (id 4), and a row the buffer could not register (-1)
    ids, am, keep = _loss_batch(make_output())
    ids = torch.cat([ids, ids[1:2]]); am = torch.cat([am, am[1:2]])
    lp = torch.tensor([[-1.5, -2.5, 0.0], [-3.5, -4.5, -5.5], [-3.0, -4.0, -5.0]])
    assert flight.on_scores(ids, am, keep, lp, tr, row_ids=torch.tensor([2, 4, -1])) == 2
    assert flight.stats["unmatched_rows"] == 0 and flight.stats["skipped_rows"] == 1
    d = decompose(flight.recorder.ledger)
    rb = d["by_provenance"]["replayed"]["by_lag"]
    assert [(b["lag"], b["n_tokens"]) for b in rb] == [(3, 3)]
    assert [(b["lag"], b["n_tokens"]) for b in d["by_provenance"]["fresh"]["by_lag"]] == [(0, 2)]
    assert d["replayed"]["sequence_weights"]["n"] == 1
    report = verify_export(flight.recorder.export_for_checker(tmp_path / "exp"), expected_head=flight.head())
    assert report["ok"], report["errors"]
    assert report["n_replayed_sequences"] == 1


def test_register_replayed_refuses_an_unknown_origin(flight):
    tr = FakeTrainer()
    flight.on_generation(make_output(), tr)
    with pytest.raises(OriginNotFound):
        flight.register_replayed(tr, prompt_ids=[7, 8, 9], completion_ids=[13, 14, 99], behavior_step=0,
                                 draw_id="x", content_digest=CD, is_weight=1)
    with pytest.raises(OriginNotFound):                               # right content, wrong step
        flight.register_replayed(tr, prompt_ids=[7, 8, 9], completion_ids=[13, 14, 15], behavior_step=2,
                                 draw_id="x", content_digest=CD, is_weight=1)
    with pytest.raises(OriginNotFound):                               # an explicit digest that is not in the record
        flight.register_replayed(tr, origin_digest="9" * 64, draw_id="x", content_digest=CD, is_weight=1)
    assert flight.stats["replayed_sequences"] == 0


def test_identical_rows_need_the_behaviour_logprobs_to_pick_the_origin(flight):
    tr = FakeTrainer()
    out = make_output()
    for k in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask", "advantages"):
        out[k] = torch.cat([out[k][1:2], out[k][1:2]], dim=0)                       # two identical rows ...
    out["old_per_token_logps"] = torch.tensor([[-3.0, -4.0, -5.0], [-3.5, -4.5, -5.5]])   # ... with different bits
    flight.on_generation(out, tr)
    kw = dict(prompt_ids=[7, 8, 9], completion_ids=[13, 14, 15], behavior_step=0, draw_id="d", content_digest=CD, is_weight=1)
    with pytest.raises(AmbiguousOrigin):
        flight.register_replayed(tr, **kw)
    assert isinstance(AmbiguousOrigin("x"), OriginNotFound)                          # the -1 sentinel path catches it
    rid = flight.register_replayed(tr, behavior_logprobs=[-3.5, -4.5, -5.5], **kw)
    assert flight.recorder.ledger.get_sequence(flight.sequence_digest(rid)).replay.origin_digest == flight.sequence_digest(1)
    with pytest.raises(AmbiguousOrigin):                                             # bits that match neither row
        flight.register_replayed(tr, behavior_logprobs=[-9.0, -9.0, -9.0], **kw)


def test_a_replayed_origin_digest_is_origin_not_found(flight):
    tr = FakeTrainer()
    flight.on_generation(make_output(), tr)
    rid = flight.register_replayed(tr, origin_digest=flight.sequence_digest(0), draw_id="d", content_digest=CD, is_weight=1)
    with pytest.raises(OriginNotFound):
        flight.register_replayed(tr, origin_digest=flight.sequence_digest(rid), draw_id="e", content_digest=CD, is_weight=1)


def test_register_replayed_with_an_explicit_origin_digest_and_before_any_generation(tmp_path):
    flight = MartingaleRecorder(tmp_path / "ws", tokenizer=b"t")
    tr = FakeTrainer()
    flight.on_generation(make_output(), tr)
    origin = flight.sequence_digest(0)
    other = MartingaleRecorder(tmp_path / "ws", tokenizer=b"t")       # a fresh recorder over the same workspace
    rid = other.register_replayed(tr, origin_digest=origin, draw_id="d", content_digest=CD, is_weight=1)
    assert rid == 0 and other.sequence_digest(0) is not None
    assert other.recorder.ledger.get_sequence(other.sequence_digest(0)).replay.origin_digest == origin
