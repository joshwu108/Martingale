"""verl 0.9.1's own data path into the loss wrapper, on CPU (skipped without verl installed).

RayPPOTrainer: DataProto union -> reorder (_balance_batch) -> to_tensordict ->
left_right_2_no_padding; actor worker: tu.make_iterator mini-batches (shuffled) ->
tu.chunk_tensordict micro-batches -> loss_function(model_output, data) with nested
per-token log-probs over the unpadded sequence. Only the model forward is synthetic.
"""
from functools import partial

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("verl")

import numpy as np
from tensordict import TensorDict
from verl.protocol import DataProto
from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import left_right_2_no_padding

from martingale.integrations.verl import ROW_ID_KEY, MartingaleVerlRecorder, wrap_ppo_loss
from martingale.record import bits_to_fraction

PROMPT_W, RESP_W = 5, 4
PROMPT_LENS = [2, 5, 3, 4, 1, 5, 2, 3]
RESP_LENS = [4, 1, 3, 2, 4, 3, 1, 2]
SHIFT = 0.5          # live log-prob = engine log-prob - SHIFT, so a misaligned score is visible


def rollout_output() -> DataProto:
    n = len(PROMPT_LENS)
    prompts = torch.zeros(n, PROMPT_W, dtype=torch.long)
    responses = torch.zeros(n, RESP_W, dtype=torch.long)
    pmask = torch.zeros(n, PROMPT_W, dtype=torch.long)
    rmask = torch.zeros(n, RESP_W, dtype=torch.long)
    for i, (p, r) in enumerate(zip(PROMPT_LENS, RESP_LENS)):
        prompts[i, PROMPT_W - p:] = torch.arange(p) + 100 * (i + 1)          # left padded
        pmask[i, PROMPT_W - p:] = 1
        responses[i, :r] = torch.arange(r) + 100 * (i + 1) + 50              # right padded
        rmask[i, :r] = 1
    attention = torch.cat([pmask, rmask], dim=1)
    lp = -(torch.arange(n * RESP_W, dtype=torch.float32).reshape(n, RESP_W) + 1) / 64 * rmask
    return DataProto(batch=TensorDict({
        "prompts": prompts, "responses": responses, "input_ids": torch.cat([prompts, responses], dim=1),
        "attention_mask": attention, "position_ids": (attention.cumsum(1) - 1).clamp(min=0),
        "response_mask": rmask, "rollout_log_probs": lp,
    }, batch_size=[n]))


def live_log_probs(micro: TensorDict) -> torch.Tensor:
    """What the FSDP forward returns: one value per unpadded token, the response's at positions
    prompt_len-1 .. seq_len-2 (no_padding_2_padding left-shifts by one)."""
    rows = []
    for i in range(micro.batch_size[0]):
        p = int(micro["attention_mask"][i, :PROMPT_W].sum())
        r = int(micro["attention_mask"][i, PROMPT_W:].sum())
        values = torch.full((p + r,), 7.0)
        values[p - 1:p - 1 + r] = micro["rollout_log_probs"][i, :r] - SHIFT
        rows.append(values)
    nested = torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    assert torch.equal(nested.offsets(), micro["input_ids"].offsets())
    return nested.requires_grad_(True)


def test_row_ids_and_live_logps_survive_verls_data_path(tmp_path):
    flight = MartingaleVerlRecorder(tmp_path / "ws", tokenizer=b"t")
    gen = rollout_output()
    flight.on_rollout(gen, step=0, weights={"w": torch.ones(2)})
    assert ROW_ID_KEY in gen.batch.keys()

    prompt_side = DataProto(batch=TensorDict({"dummy": torch.zeros(len(gen), 1)}, batch_size=[len(gen)]),
                            non_tensor_batch={"uid": np.array([f"u{i // 2}" for i in range(len(gen))], dtype=object)})
    batch = prompt_side.union(gen)
    batch.reorder(torch.tensor([3, 7, 0, 5, 1, 6, 2, 4]))
    td = left_right_2_no_padding(batch.to_tensordict())
    assert td["input_ids"].is_nested and not td["prompts"].is_nested

    calls, seen_ids = [], []

    def ppo_loss(config, model_output, data, dp_group=None):
        calls.append(config)
        return torch.tensor(0.0), {}

    loss_fn = partial(wrap_ppo_loss(ppo_loss, flight, step=lambda: 0, weights=lambda: {"w": torch.ones(2)}),
                      config="actor-config")
    for mini in tu.make_iterator(td, mini_batch_size=4, epochs=1, seed=0, dataloader_kwargs={"shuffle": True}):
        for micro in tu.chunk_tensordict(mini, 2):
            seen_ids += micro[ROW_ID_KEY].tolist()
            loss_fn(model_output={"log_probs": live_log_probs(micro)}, data=micro, dp_group=None)

    assert calls == ["actor-config"] * 4
    assert sorted(seen_ids) == list(range(len(gen)))
    assert flight.stats["unmatched_rows"] == 0 and flight.stats["scored_sequences"] == len(gen)
    assert flight.stats["scored_tokens"] == sum(RESP_LENS)
    ledger = flight.recorder.ledger
    for seq in ledger.all_sequences():
        scores = ledger.scores_for(seq.digest)
        assert [s.position for s in scores] == list(range(len(seq.tokens)))
        for s in scores:
            engine = bits_to_fraction(seq.tokens[s.position].logprob_bits)
            assert float(bits_to_fraction(s.logprob_bits)) == pytest.approx(float(engine) - SHIFT)
