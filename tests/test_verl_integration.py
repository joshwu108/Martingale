"""CPU contract tests for the verl DataProto hook pair."""
import pytest

torch = pytest.importorskip("torch")

from checker.verify_tokens import verify_export
from martingale.diagnostics import decompose
from martingale.integrations.verl import MartingaleVerlRecorder, wrap_ppo_loss
from martingale.record import bits_to_fraction


class FakeDataProto:
    def __init__(self, batch):
        self.batch = batch


def make_batch():
    return FakeDataProto({
        "prompts": torch.tensor([[0, 5, 6], [7, 8, 9]]),
        "responses": torch.tensor([[11, 12, 0], [13, 14, 15]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]),
        "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        "rollout_log_probs": torch.tensor([[-1.0, -2.0, 0.0], [-3.0, -4.0, -5.0]]),
        "advantages": torch.tensor([[0.5, 0.5, 0.0], [-0.5, -0.5, -0.5]]),
    })


@pytest.fixture
def flight(tmp_path):
    return MartingaleVerlRecorder(tmp_path / "ws", tokenizer=b"t")


@pytest.fixture
def weights():
    return {"weight": torch.ones(2, 3)}


def test_rollout_and_repeated_train_steps(flight, weights, tmp_path):
    batch = make_batch()
    flight.on_rollout(batch, step=0, weights=weights)
    assert batch.batch["martingale_row_id"].tolist() == [0, 1]
    assert flight.stats["sequences"] == 2
    seqs = list(flight.recorder.ledger.all_sequences())
    assert [s.prompt_len for s in seqs] == [2, 3]
    assert [len(s.tokens) for s in seqs] == [2, 3]
    assert bits_to_fraction(seqs[0].tokens[0].logprob_bits) == -1
    lp = batch.batch["rollout_log_probs"] - 0.5
    assert flight.on_train_step(batch, lp, step=0, weights=weights) == 2
    assert flight.on_train_step(batch, lp, step=0, weights=weights) == 0
    weights["weight"] += 1
    assert flight.on_train_step(batch, lp, step=1, weights=weights) == 2
    assert decompose(flight.recorder.ledger)["staleness_histogram"] == {"0": 5, "1": 5}
    report = verify_export(flight.recorder.export_for_checker(tmp_path / "exp"), expected_head=flight.head())
    assert report["ok"], report["errors"]


def test_row_ids_survive_shuffle_and_nan_is_skipped(flight, weights):
    batch = make_batch()
    batch.batch["rollout_log_probs"][0, 1] = float("nan")
    flight.on_rollout(batch, step=0, weights=weights)
    assert flight.stats["nan_rows"] == 1 and flight.stats["sequences"] == 1
    for key, value in batch.batch.items():
        batch.batch[key] = value[[1, 0]]
    assert flight.on_train_step(batch, torch.zeros(2, 3), step=0, weights=weights) == 1


def test_missing_engine_logps_and_impure_mask_fail(flight, weights):
    batch = make_batch()
    batch.batch.pop("rollout_log_probs")
    with pytest.raises(ValueError, match="rollout_log_probs"):
        flight.on_rollout(batch, step=0, weights=weights)
    batch = make_batch()
    batch.batch["response_mask"][0] = torch.tensor([1, 0, 1])
    with pytest.raises(ValueError, match="pure"):
        flight.on_rollout(batch, step=0, weights=weights)


def test_sharded_weights_fail(flight, weights):
    weights["weight"] = torch.empty(0, 3)
    with pytest.raises(RuntimeError, match="sharded"):
        flight.on_rollout(make_batch(), step=0, weights=weights)


def test_content_fallback_and_loss_wrapper(flight, weights, monkeypatch):
    batch = make_batch()
    flight.on_rollout(batch, step=0, weights=weights)
    batch.batch.pop("martingale_row_id")
    calls = []

    class Padded(dict):
        def to_padded_tensor(self):
            return self

    data = Padded(batch.batch)
    monkeypatch.setattr("martingale.integrations._verl_compat.require_verl",
                        lambda: type("Support", (), {"no_padding_2_padding": staticmethod(lambda x, _: x)})())

    def base(config, model_output, data, dp_group=None):
        calls.append((config, dp_group))
        return "loss", {}

    wrapped = wrap_ppo_loss(base, flight, step=lambda: 0, weights=lambda: weights)
    assert wrapped("cfg", {"log_probs": batch.batch["rollout_log_probs"]}, data) == ("loss", {})
    assert calls == [("cfg", None)] and flight.stats["scored_tokens"] == 5


def test_worker_process_can_score_driver_record(tmp_path, weights):
    workspace = tmp_path / "ws"
    driver = MartingaleVerlRecorder(workspace, tokenizer=b"t")
    batch = make_batch()
    driver.on_rollout(batch, step=0, weights=weights)
    worker = MartingaleVerlRecorder(workspace, tokenizer=b"t")
    assert worker.on_train_step(batch, batch.batch["rollout_log_probs"], step=0, weights=weights) == 2
    newer = MartingaleVerlRecorder(workspace, tokenizer=b"t")
    next_batch = make_batch()
    newer.on_rollout(next_batch, step=1, weights=weights)
    assert next_batch.batch["martingale_row_id"].tolist() == [2, 3]


def test_training_rejection_mask_keeps_original_token_alignment(flight, weights):
    batch = make_batch()
    flight.on_rollout(batch, step=0, weights=weights)
    batch.batch["response_mask"][1] = torch.tensor([1, 0, 1])
    assert flight.on_train_step(batch, batch.batch["rollout_log_probs"], step=0, weights=weights) == 2
    assert flight.stats["scored_tokens"] == 5


def test_real_class_smoke():
    pytest.importorskip("verl")
    from martingale.integrations._verl_compat import require_verl

    support = require_verl()
    assert hasattr(support.data_proto, "reorder")
    import tensordict

    batch = make_batch().batch
    proto = support.data_proto(batch=tensordict.TensorDict(batch, batch_size=[2]))
    proto.reorder(torch.tensor([1, 0]))
    assert proto.batch["responses"][0, 0].item() == 13


def test_real_trl_class_smoke():
    pytest.importorskip("trl")
    from martingale.integrations.trl import build_trainer_class

    cls = build_trainer_class()
    assert cls.__name__ == "MartingaleGRPOTrainer"


class TestLogprobsMode:
    """2026-10-10: the revision binds rollout.logprobs_mode (verl 0.9.1 default processed_logprobs,
    config/rollout.py:197) and that the actor scales by temperature (transformer_impl.py:98-103)."""

    ROLLOUT = {"name": "vllm", "temperature": 0.7, "top_p": 1.0, "top_k": -1, "response_length": 16,
               "dtype": "bfloat16", "tensor_model_parallel_size": 1, "seed": 42,
               "logprobs_mode": "processed_logprobs", "calculate_log_probs": True}

    def _flight(self, tmp_path, **cfg):
        from types import SimpleNamespace
        rollout = SimpleNamespace(**{**self.ROLLOUT, **cfg})          # attribute access, like a DictConfig
        return MartingaleVerlRecorder(tmp_path / "ws", tokenizer=b"t", rollout_config=rollout)

    def test_rollout_config_is_bound_into_both_revisions(self, tmp_path, weights):
        flight = self._flight(tmp_path)
        batch = make_batch()
        flight.on_rollout(batch, step=0, weights=weights)
        flight.on_train_step(batch, batch.batch["rollout_log_probs"], step=0, weights=weights)
        revs = {r.sampler.logprobs_source: r.sampler for r in flight.recorder.ledger.all_revisions()}
        eng, act = revs["engine_sampling"], revs["trainer_scores"]
        assert eng.logprobs_mode == "processed_logprobs" and eng.temperature_applied is True
        assert eng.temperature == 0.7 and eng.engine == "verl:vllm" and eng.max_tokens == 16 and eng.top_k == -1
        assert act.logprobs_mode is None and act.temperature_applied is True and act.temperature == 0.7
        assert decompose(flight.recorder.ledger)["lag0_floor"] is not None        # like for like: floor reported

    def test_plain_dict_config_works_too(self, tmp_path, weights):
        flight = MartingaleVerlRecorder(tmp_path / "ws", tokenizer=b"t", rollout_config=dict(self.ROLLOUT))
        flight.on_rollout(make_batch(), step=0, weights=weights)
        assert next(flight.recorder.ledger.all_revisions()).sampler.logprobs_mode == "processed_logprobs"

    def test_raw_logprobs_rollout_withholds_the_floor(self, tmp_path, weights):
        flight = self._flight(tmp_path, logprobs_mode="raw_logprobs")
        batch = make_batch()
        flight.on_rollout(batch, step=0, weights=weights)
        flight.on_train_step(batch, batch.batch["rollout_log_probs"], step=0, weights=weights)
        s = next(r.sampler for r in flight.recorder.ledger.all_revisions() if r.sampler.logprobs_source == "engine_sampling")
        assert s.logprobs_mode == "raw_logprobs" and s.temperature_applied is False
        d = decompose(flight.recorder.ledger)
        assert d["lag0_floor"] is None and "raw_logprobs" in d["lag0_floor_withheld"]["reason"]

    def test_without_rollout_config_the_mode_is_unknown(self, flight, weights):
        flight.on_rollout(make_batch(), step=0, weights=weights)
        s = next(flight.recorder.ledger.all_revisions()).sampler
        assert s.logprobs_source == "engine_sampling" and s.logprobs_mode is None and s.temperature_applied is None
        assert s.engine == "verl"


def test_require_verl_missing_package():
    import importlib.util

    if importlib.util.find_spec("verl") is not None:
        pytest.skip("verl installed")
    from martingale.integrations._verl_compat import require_verl

    with pytest.raises(ImportError, match="verl==0.9.1"):
        require_verl()
