"""Row ids survive a multi-process scatter: rank-packed ids, resolved through a shared ledger.

Reservoir's `mix_distributed` gathers every rank's generation slice to rank 0, replaces dead rows
there (registering replayed rows with rank 0's recorder) and scatters the slices back, so a rank
scores rows whose ids another rank allocated. Two fake trainers with process_index 0 and 1 and two
recorders over ONE workspace play that out here without torch.distributed."""
from fractions import Fraction

import pytest

torch = pytest.importorskip("torch")

from checker.verify_tokens import verify_export
from martingale.diagnostics import decompose
from martingale.integrations.trl import (
    ROW_ID_RANK_BITS,
    MartingaleRecorder,
    pack_row_id,
    unpack_row_id,
)
from tests.test_trl_integration import FakeTrainer, make_output

CD = "d" * 64


def _rank(k: int) -> FakeTrainer:
    tr = FakeTrainer()
    tr.accelerator.process_index = k
    return tr


def _loss_batch(out):
    ids = torch.cat([out["prompt_ids"], out["completion_ids"]], dim=1)
    am = torch.cat([out["prompt_mask"], out["completion_mask"]], dim=1)
    return ids, am, out["completion_ids"].size(1)


def test_pack_unpack_round_trip_and_bounds():
    assert pack_row_id(0, 5) == 5                                  # rank 0 ids are the plain local counter
    assert unpack_row_id(pack_row_id(3, 7)) == (3, 7)
    assert pack_row_id(1, 0) == 1 << ROW_ID_RANK_BITS
    assert unpack_row_id(-1) == (-1, -1)                           # the skip sentinel stays a sentinel
    with pytest.raises(ValueError):
        pack_row_id(-1, 0)
    with pytest.raises(ValueError):
        pack_row_id(0, 1 << ROW_ID_RANK_BITS)
    with pytest.raises(ValueError):
        pack_row_id(1 << 23, 0)


def test_two_ranks_allocate_disjoint_ids_and_the_ledger_resolves_them(tmp_path):
    ws = tmp_path / "ws"
    f0, f1 = MartingaleRecorder(ws, tokenizer=b"t"), MartingaleRecorder(ws, tokenizer=b"t")
    out0 = f0.on_generation(make_output(), _rank(0))
    out1 = f1.on_generation(make_output(), _rank(1))
    assert out0["martingale_row_id"].tolist() == [0, 1]
    assert out1["martingale_row_id"].tolist() == [pack_row_id(1, 0), pack_row_id(1, 1)]
    assert set(out0["martingale_row_id"].tolist()).isdisjoint(out1["martingale_row_id"].tolist())
    seqs = list(f0.recorder.ledger.all_sequences())
    assert sorted(s.actor_id for s in seqs) == [0, 0, 1, 1]        # one chain per rank in one file
    # a third recorder over the same workspace resolves both ranks' ids without any in-memory map
    f2 = MartingaleRecorder(ws, tokenizer=b"t")
    assert f2.sequence_digest(pack_row_id(1, 1)) == f1.sequence_digest(pack_row_id(1, 1))
    assert f2.sequence_digest(0) == f0.sequence_digest(0)
    assert f2.sequence_digest(pack_row_id(2, 0)) is None


def test_owner_registers_a_replayed_row_for_another_rank_and_that_rank_scores_it(tmp_path):
    ws = tmp_path / "ws"
    owner, other = MartingaleRecorder(ws, tokenizer=b"t"), MartingaleRecorder(ws, tokenizer=b"t")
    t0, t1 = _rank(0), _rank(1)
    owner.on_generation(make_output(), t0)
    o1 = make_output()
    o1["old_per_token_logps"] = torch.tensor([[-1.0, -2.0, 0.0], [-3.5, -4.5, -5.5]])   # rank 1's own bits
    out1 = other.on_generation(o1, t1)                                 # step 0: rank 1's fresh rows
    for t in (t0, t1):
        t.state.global_step = 2
    with torch.no_grad():
        t1.model.weight.add_(1.0)
    owner.on_generation(make_output(), t0)
    out1_step2 = other.on_generation(make_output(), t1)
    # rank 0 owns the buffer: it replaces a dead row in rank 1's slice with rank 1's own step-0 completion
    rid = owner.register_replayed(t0, prompt_ids=[7, 8, 9], completion_ids=[13, 14, 15], behavior_step=0,
                                  behavior_logprobs=[-3.5, -4.5, -5.5], draw_id="2/0", content_digest=CD,
                                  is_weight=Fraction(2, 3), advantage=0.25)
    assert unpack_row_id(rid) == (0, 4)                              # rank 0's own counter continues
    seq = owner.recorder.ledger.get_sequence(owner.sequence_digest(rid))
    assert seq.replay.origin_digest == other.sequence_digest(out1["martingale_row_id"][1].item())
    assert seq.actor_id == 0                                         # written on the owner's chain
    # the scatter: rank 1's loss batch holds its own fresh row and the row the owner registered
    ids, am, keep = _loss_batch(out1_step2)
    lp = torch.tensor([[-1.5, -2.5, 0.0], [-3.5, -4.5, -5.5]])
    row_ids = torch.tensor([out1_step2["martingale_row_id"][0].item(), rid])
    assert other.on_scores(ids, am, keep, lp, t1, row_ids=row_ids) == 2
    assert other.stats["unmatched_rows"] == 0 and other.stats["foreign_rows"] == 1
    d = decompose(owner.recorder.ledger)
    assert [(b["lag"], b["n_tokens"]) for b in d["by_provenance"]["replayed"]["by_lag"]] == [(2, 3)]
    assert d["replayed"]["sequence_weights"]["n"] == 1
    report = verify_export(owner.recorder.export_for_checker(tmp_path / "exp"), expected_head=owner.head())
    assert report["ok"], report["errors"]
    assert sorted(report["heads"]) == [0, 1] and report["n_replayed_sequences"] == 1


def test_without_a_shared_workspace_a_foreign_id_is_unmatched_not_misattributed(tmp_path):
    owner = MartingaleRecorder(tmp_path / "ws0", tokenizer=b"t")
    other = MartingaleRecorder(tmp_path / "ws1", tokenizer=b"t")
    t0, t1 = _rank(0), _rank(1)
    owner.on_generation(make_output(), t0)
    out1 = other.on_generation(make_output(), t1)
    rid = owner.register_replayed(t0, origin_digest=owner.sequence_digest(1), draw_id="d", content_digest=CD, is_weight=1)
    ids, am, keep = _loss_batch(out1)
    row_ids = torch.tensor([out1["martingale_row_id"][0].item(), rid])
    assert other.on_scores(ids, am, keep, torch.zeros(2, 3), t1, row_ids=row_ids) == 1
    assert other.stats["unmatched_rows"] == 1 and other.stats["foreign_rows"] == 1
    assert other.recorder.ledger.count_scores() == 2                 # only rank 1's own row was scored


def test_ids_resolve_after_more_than_keep_generations_batches(tmp_path):
    flight = MartingaleRecorder(tmp_path / "ws", tokenizer=b"t")
    tr = _rank(0)
    first = flight.on_generation(make_output(), tr)["martingale_row_id"][0].item()
    for step in (1, 2, 3):
        tr.state.global_step = step
        flight.on_generation(make_output(), tr)
    assert flight.sequence_digest(first) is not None                 # the ledger remembers what the maps forgot
