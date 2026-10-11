"""The verl adapter keeps the TRL row-id contract: rank-packed ids, resolved through a shared ledger.

verl's driver records rollouts and an actor worker, in another Ray process with its own recorder
over the same workspace, scores the loss rows; with more than one rank a loss batch can hold rows
another rank's driver allocated. Two fake ranks over ONE workspace play that out here, as
tests/test_distributed_rows.py does for TRL."""
import pytest

torch = pytest.importorskip("torch")

from checker.verify_tokens import verify_export
from martingale.integrations._common import KEEP_GENERATIONS, pack_row_id
from martingale.integrations.verl import LENGTH_KEY, ROW_ID_KEY, MartingaleVerlRecorder
from tests.test_verl_integration import FakeDataProto


def make_batch(shift: int = 0) -> FakeDataProto:
    """Two rows (one padded); ``shift`` changes the completion tokens so each rank's rows differ."""
    return FakeDataProto({
        "prompts": torch.tensor([[0, 5, 6], [7, 8, 9]]),
        "responses": torch.tensor([[11 + shift, 12 + shift, 0], [13 + shift, 14 + shift, 15 + shift]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]),
        "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        "rollout_log_probs": torch.tensor([[-1.0, -2.0, 0.0], [-3.0, -4.0, -5.0]]),
        "advantages": torch.tensor([[0.5, 0.5, 0.0], [-0.5, -0.5, -0.5]]),
    })


def take(*rows: tuple[FakeDataProto, int]) -> FakeDataProto:
    """A loss batch built from rows of recorded batches, ids and saved lengths included: the scatter."""
    keys = rows[0][0].batch.keys()
    return FakeDataProto({k: torch.stack([b.batch[k][i] for b, i in rows]) for k in keys})


@pytest.fixture
def weights():
    return {"weight": torch.ones(2, 3)}


def test_two_ranks_allocate_disjoint_ids_and_the_ledger_resolves_them(tmp_path, weights):
    ws = tmp_path / "ws"
    d0, d1 = MartingaleVerlRecorder(ws, tokenizer=b"t"), MartingaleVerlRecorder(ws, tokenizer=b"t")
    b0, b1 = make_batch(), make_batch(shift=100)
    d0.on_rollout(b0, step=0, weights=weights, actor_id=0)
    d1.on_rollout(b1, step=0, weights=weights, actor_id=1)
    assert b0.batch[ROW_ID_KEY].tolist() == [0, 1]                  # rank 0 ids are the plain counter
    assert b1.batch[ROW_ID_KEY].tolist() == [pack_row_id(1, 0), pack_row_id(1, 1)]
    seqs = list(d0.recorder.ledger.all_sequences())
    assert sorted(s.actor_id for s in seqs) == [0, 0, 1, 1]         # one chain per rank in one file
    assert sorted(s.sequence_id for s in seqs) == ["step0/row0", "step0/row0", "step0/row1", "step0/row1"]
    # a worker over the same workspace resolves both ranks' ids without any in-memory map
    worker = MartingaleVerlRecorder(ws, tokenizer=b"t")
    assert worker.sequence_digest(pack_row_id(1, 1)) == d1.sequence_digest(pack_row_id(1, 1))
    assert worker.sequence_digest(0) == d0.sequence_digest(0)
    assert worker.sequence_digest(pack_row_id(2, 0)) is None
    assert worker.sequence_digest(-1) is None                       # the skip sentinel never resolves


def test_a_worker_scores_rows_another_rank_allocated(tmp_path, weights):
    ws = tmp_path / "ws"
    d0, d1 = MartingaleVerlRecorder(ws, tokenizer=b"t"), MartingaleVerlRecorder(ws, tokenizer=b"t")
    b0, b1 = make_batch(), make_batch(shift=100)
    d0.on_rollout(b0, step=0, weights=weights, actor_id=0)
    d1.on_rollout(b1, step=0, weights=weights, actor_id=1)
    worker = MartingaleVerlRecorder(ws, tokenizer=b"t")             # rank 1's actor: a separate process
    loss = take((b1, 0), (b0, 1))                                    # its own row and one rank 0 allocated
    assert worker.on_train_step(loss, loss.batch["rollout_log_probs"] - 0.5, step=0, weights=weights) == 2
    assert worker.stats["unmatched_rows"] == 0 and worker.stats["scored_tokens"] == 5
    ledger = worker.recorder.ledger
    assert len(ledger.scores_for(d0.sequence_digest(1))) == 3        # scored against rank 0's sequence
    assert len(ledger.scores_for(d1.sequence_digest(pack_row_id(1, 0)))) == 2
    assert ledger.scores_for(d0.sequence_digest(0)) == []            # and nothing else
    report = verify_export(d0.recorder.export_for_checker(tmp_path / "exp"), expected_head=d0.head())
    assert report["ok"], report["errors"]
    assert sorted(report["heads"]) == [0, 1]


def test_without_a_shared_workspace_a_foreign_id_is_unmatched_not_misattributed(tmp_path, weights):
    d0 = MartingaleVerlRecorder(tmp_path / "ws0", tokenizer=b"t")
    d1 = MartingaleVerlRecorder(tmp_path / "ws1", tokenizer=b"t")
    b0, b1 = make_batch(), make_batch()                              # same content: content must not decide
    d0.on_rollout(b0, step=0, weights=weights, actor_id=0)
    d1.on_rollout(b1, step=0, weights=weights, actor_id=1)
    loss = take((b1, 0), (b0, 1))
    assert d1.on_train_step(loss, torch.zeros(2, 3), step=0, weights=weights) == 1
    assert d1.stats["unmatched_rows"] == 1
    assert d1.recorder.ledger.count_scores() == 2                    # only rank 1's own row was scored


def test_ids_resolve_after_more_than_keep_generations_batches(tmp_path, weights):
    flight = MartingaleVerlRecorder(tmp_path / "ws", tokenizer=b"t")
    first = make_batch()
    flight.on_rollout(first, step=0, weights=weights)
    for step in range(1, KEEP_GENERATIONS + 2):
        flight.on_rollout(make_batch(), step=step, weights=weights)
    assert flight.sequence_digest(0) is not None                    # the ledger remembers what the maps forgot
    assert flight.on_train_step(first, first.batch["rollout_log_probs"], step=0, weights=weights) == 2


def test_a_reopened_workspace_continues_each_ranks_counter(tmp_path, weights):
    ws = tmp_path / "ws"
    d0, d1 = MartingaleVerlRecorder(ws, tokenizer=b"t"), MartingaleVerlRecorder(ws, tokenizer=b"t")
    d0.on_rollout(make_batch(), step=0, weights=weights, actor_id=0)
    d1.on_rollout(make_batch(), step=0, weights=weights, actor_id=1)
    d1.on_rollout(make_batch(), step=1, weights=weights, actor_id=1)
    before = d0.sequence_digest(pack_row_id(1, 0))
    resumed = MartingaleVerlRecorder(ws, tokenizer=b"t")
    b0, b1, b2 = make_batch(), make_batch(), make_batch()
    resumed.on_rollout(b0, step=2, weights=weights, actor_id=0)
    resumed.on_rollout(b1, step=2, weights=weights, actor_id=1)
    resumed.on_rollout(b2, step=2, weights=weights, actor_id=2)
    assert b0.batch[ROW_ID_KEY].tolist() == [2, 3]
    assert b1.batch[ROW_ID_KEY].tolist() == [pack_row_id(1, 4), pack_row_id(1, 5)]
    assert b2.batch[ROW_ID_KEY].tolist() == [pack_row_id(2, 0), pack_row_id(2, 1)]
    assert resumed.sequence_digest(pack_row_id(1, 0)) == before      # earlier ids were not rebound
    assert b0.batch[LENGTH_KEY].tolist() == [2, 3]
