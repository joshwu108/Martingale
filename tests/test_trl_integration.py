"""TRL integration against a fake GRPOTrainer: generation rows -> sequences, loss path -> scores,
mixin wiring, lag accounting, and the whole record passing the independent checker."""
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from checker.verify_tokens import verify_export
from martingale.diagnostics import decompose
from martingale.integrations.trl import MartingaleGRPOMixin, MartingaleRecorder, sampler_from_trainer
from martingale.record import bits_to_fraction

PAD = 0


class FakeArgs(SimpleNamespace):
    pass


class FakeTrainer:
    def __init__(self, use_vllm=False):
        self.model = nn.Linear(3, 2)
        self.model.train()
        self.args = FakeArgs(temperature=0.9, top_p=1.0, top_k=None, min_p=None, repetition_penalty=1.0,
                             max_completion_length=8, use_vllm=use_vllm, vllm_mode="server", seed=1,
                             per_device_train_batch_size=2)
        self.state = SimpleNamespace(global_step=0)
        self.accelerator = SimpleNamespace(process_index=0, unwrap_model=lambda m: m)
        self.recompute_calls = 0

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None):
        self.recompute_calls += 1
        comp = input_ids[:, -logits_to_keep:]
        return -0.5 - 0.01 * comp.float(), None, None   # deterministic fake log-probs


def make_output(with_old=True, with_sampling=False):
    prompt_ids = torch.tensor([[PAD, 5, 6], [7, 8, 9]])          # left-padded
    prompt_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    completion_ids = torch.tensor([[11, 12, PAD], [13, 14, 15]])   # right-padded
    completion_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    out = {"prompt_ids": prompt_ids, "prompt_mask": prompt_mask, "completion_ids": completion_ids,
           "completion_mask": completion_mask, "advantages": torch.tensor([0.5, -0.5])}
    if with_old:
        out["old_per_token_logps"] = torch.tensor([[-1.0, -2.0, 0.0], [-3.0, -4.0, -5.0]])
    if with_sampling:
        out["sampling_per_token_logps"] = torch.tensor([[-1.1, -2.1, 0.0], [-3.1, -4.1, -5.1]])
    return out


@pytest.fixture
def flight(tmp_path):
    return MartingaleRecorder(tmp_path / "ws", tokenizer=b"{tokenizer}")


class TestGeneration:
    def test_rows_become_sequences_with_unpadded_prompt_and_exact_bits(self, flight):
        tr = FakeTrainer()
        flight.on_generation(make_output(), tr)
        seqs = list(flight.recorder.ledger.all_sequences())
        assert [len(s.tokens) for s in seqs] == [2, 3]
        assert seqs[0].prompt_len == 2 and seqs[1].prompt_len == 3
        assert [t.token_id for t in seqs[1].tokens] == [13, 14, 15]
        assert bits_to_fraction(seqs[1].tokens[2].logprob_bits) == -5.0
        assert bits_to_fraction(seqs[0].reward_bits) == 0.5
        assert flight.stats["sequences"] == 2 and flight.stats["tokens"] == 5

    def test_logprob_source_preference_is_bound_into_sampler(self, flight):
        tr = FakeTrainer(use_vllm=True)
        flight.on_generation(make_output(with_old=True, with_sampling=True), tr)
        rev = next(flight.recorder.ledger.all_revisions())
        assert rev.sampler.logprobs_mode == "engine_sampling" and rev.sampler.engine == "vllm:server"
        seq = next(flight.recorder.ledger.all_sequences())
        assert bits_to_fraction(seq.tokens[0].logprob_bits) == bits_to_fraction(
            __import__("martingale.record", fromlist=["float_bits"]).float_bits(-1.1))

    def test_recompute_when_no_logprobs_given(self, flight):
        tr = FakeTrainer()
        flight.on_generation(make_output(with_old=False), tr)
        assert tr.recompute_calls == 1
        assert next(flight.recorder.ledger.all_revisions()).sampler.logprobs_mode == "trainer_recompute"

    def test_one_weights_digest_per_step(self, flight):
        tr = FakeTrainer()
        flight.on_generation(make_output(), tr)
        flight.on_generation(make_output(), tr)
        assert flight.stats["revisions"] == 1
        tr.state.global_step = 1
        flight.on_generation(make_output(), tr)
        assert flight.stats["revisions"] == 2

    def test_sampler_from_trainer(self):
        s = sampler_from_trainer(FakeTrainer(), "x")
        assert s.temperature == 0.9 and s.max_tokens == 8 and s.engine == "transformers" and s.dtype == "float32"


class TestScores:
    def _loss_batch(self, out):
        ids = torch.cat([out["prompt_ids"], out["completion_ids"]], dim=1)
        am = torch.cat([out["prompt_mask"], out["completion_mask"]], dim=1)
        return ids, am, out["completion_ids"].size(1)

    def test_scores_match_rows_and_record_lag(self, flight, tmp_path):
        tr = FakeTrainer()
        out = flight.on_generation(make_output(), tr)
        assert "martingale_row_id" in out and out["martingale_row_id"].tolist() == [0, 1]
        ids, am, keep = self._loss_batch(out)
        tr.state.global_step = 2                                   # two optimizer steps later
        with torch.no_grad():
            tr.model.weight.add_(1.0)                              # weights changed -> new revision
        lp = torch.tensor([[-1.5, -2.5, 0.0], [-3.5, -4.5, -5.5]])
        assert flight.on_scores(ids, am, keep, lp, tr) == 2
        d = decompose(flight.recorder.ledger)
        assert d["staleness_histogram"] == {"2": 5}
        b = d["by_lag"][0]
        assert b["mean_log_ratio"] == "-1/2"                       # exact: every token scored 0.5 lower
        report = verify_export(flight.recorder.export_for_checker(tmp_path / "exp"), expected_head=flight.head())
        assert report["ok"], report["errors"]

    def test_unmatched_rows_are_counted_not_recorded(self, flight):
        tr = FakeTrainer()
        flight.on_generation(make_output(), tr)
        ids = torch.tensor([[1, 2, 3, 99, 98, 97]]); am = torch.ones_like(ids)
        assert flight.on_scores(ids, am, 3, torch.zeros(1, 3), tr) == 0
        assert flight.stats["unmatched_rows"] == 1

    def test_record_scores_off(self, tmp_path):
        flight = MartingaleRecorder(tmp_path / "ws", tokenizer=b"t", record_scores=False)
        tr = FakeTrainer(); out = make_output(); flight.on_generation(out, tr)
        assert flight.on_scores(*self._loss_batch(out), torch.zeros(2, 3), tr) == 0


class TestMixinWiring:
    def test_mixin_routes_generation_and_grad_path_only(self, flight):
        calls = {"gen": 0, "lp": 0}

        class FakeBase(FakeTrainer):
            def _generate_and_score_completions(self, inputs):
                calls["gen"] += 1
                return make_output()

            def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None):
                calls["lp"] += 1
                return super()._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep)

        class Trainer(MartingaleGRPOMixin, FakeBase):
            def __init__(self):
                super().__init__()
                self.flight_recorder = flight

        tr = Trainer()
        out = tr._generate_and_score_completions({})
        assert calls["gen"] == 1 and flight.stats["sequences"] == 2
        ids = torch.cat([out["prompt_ids"], out["completion_ids"]], dim=1)
        am = torch.cat([out["prompt_mask"], out["completion_mask"]], dim=1)
        with torch.no_grad():                                           # old-logps path: no scores
            tr._get_per_token_logps_and_entropies(tr.model, ids, am, 3)
        assert flight.stats["score_calls"] == 0
        tr._get_per_token_logps_and_entropies(tr.model, ids, am, 3)    # loss path: scores
        assert flight.stats["score_calls"] == 1 and flight.stats["scored_tokens"] == 5
        tr.model.eval()
        assert tr._generate_and_score_completions({}) is not None and flight.stats["sequences"] == 2

    def test_trainer_class_requires_trl(self):
        pytest.importorskip("trl", reason="trl not installed: build_trainer_class must raise ImportError")

    def test_build_trainer_class_import_error_without_trl(self):
        import importlib
        try:
            import trl  # noqa: F401
            pytest.skip("trl installed")
        except ImportError:
            pass
        from martingale.integrations import trl as mod
        importlib.reload(mod)
        with pytest.raises(ImportError, match="requires trl"):
            mod.build_trainer_class()


class TestReviewFindings:
    """2026-10-06 review: duplicate rows, row ids through a shuffle, NaN rows, ref model, mask purity."""

    def _loss_batch(self, out, perm=None):
        ids = torch.cat([out["prompt_ids"], out["completion_ids"]], dim=1)
        am = torch.cat([out["prompt_mask"], out["completion_mask"]], dim=1)
        if perm is not None:
            ids, am = ids[perm], am[perm]
        return ids, am, out["completion_ids"].size(1)

    def test_duplicate_rows_in_one_batch_each_get_scored_by_row_id(self, flight):
        tr = FakeTrainer()
        out = make_output()
        for k in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask", "old_per_token_logps", "advantages"):
            out[k] = torch.cat([out[k][1:2], out[k][1:2]], dim=0)          # two identical rows
        out = flight.on_generation(out, tr)
        assert flight.stats["sequences"] == 2
        ids, am, keep = self._loss_batch(out)
        assert flight.on_scores(ids, am, keep, torch.zeros(2, 3), tr, row_ids=out["martingale_row_id"]) == 2
        assert flight.stats["unmatched_rows"] == 0 and flight.stats["scored_tokens"] == 6

    def test_duplicate_rows_without_row_ids_use_content_fifo(self, flight):
        tr = FakeTrainer()
        out = make_output()
        for k in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask", "old_per_token_logps", "advantages"):
            out[k] = torch.cat([out[k][1:2], out[k][1:2]], dim=0)
        out = flight.on_generation(out, tr)
        ids, am, keep = self._loss_batch(out)
        assert flight.on_scores(ids, am, keep, torch.zeros(2, 3), tr) == 2       # two distinct sequences, no IntegrityError
        assert flight.on_scores(ids, am, keep, torch.zeros(2, 3), tr) == 0       # same weights again: duplicates, not errors
        assert flight.stats["unmatched_rows"] == 0 and flight.stats["duplicate_scores"] == 2

    def test_row_ids_survive_a_shuffle(self, flight):
        tr = FakeTrainer()
        out = flight.on_generation(make_output(), tr)
        perm = torch.tensor([1, 0])
        ids, am, keep = self._loss_batch(out, perm)
        lp = torch.tensor([[-3.5, -4.5, -5.5], [-1.5, -2.5, 0.0]])
        assert flight.on_scores(ids, am, keep, lp, tr, row_ids=out["martingale_row_id"][perm]) == 2
        seqs = list(flight.recorder.ledger.all_sequences())
        s0 = flight.recorder.ledger.scores_for(seqs[0].digest)                  # row 0 (2 tokens) scored -1.5, -2.5
        assert [float(bits_to_fraction(x.logprob_bits)) for x in s0] == [-1.5, -2.5]

    def test_repeated_content_in_a_later_generation_is_not_scored_against_the_old_sequence(self, flight):
        tr = FakeTrainer()
        out1 = flight.on_generation(make_output(), tr)
        tr.state.global_step = 3
        with torch.no_grad():
            tr.model.weight.add_(1.0)
        out2 = flight.on_generation(make_output(), tr)                      # same content, new step
        ids, am, keep = self._loss_batch(out2)
        flight.on_scores(ids, am, keep, torch.zeros(2, 3), tr, row_ids=out2["martingale_row_id"])
        d = decompose(flight.recorder.ledger)
        assert d["staleness_histogram"] == {"0": 5}                           # lag 0, not lag 3

    def test_nan_logprob_rows_are_skipped_and_counted(self, flight):
        tr = FakeTrainer()
        out = make_output()
        out["old_per_token_logps"][0, 1] = float("nan")
        flight.on_generation(out, tr)
        assert flight.stats["sequences"] == 1 and flight.stats["nan_rows"] == 1
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            flight.head()
        assert any("skipped" in str(x.message) for x in w)

    def test_impure_mask_is_refused(self, flight):
        tr = FakeTrainer()
        out = make_output()
        out["completion_mask"] = torch.tensor([[1, 0, 1], [1, 1, 1]])
        with pytest.raises(ValueError, match="not pure"):
            flight.on_generation(out, tr)

    def test_ref_model_pass_is_not_scored(self, flight):
        class FakeBase(FakeTrainer):
            def _generate_and_score_completions(self, inputs):
                return make_output()

        class Trainer(MartingaleGRPOMixin, FakeBase):
            def __init__(self):
                super().__init__()
                self.flight_recorder = flight
                self.ref_model = nn.Linear(3, 2)

        tr = Trainer()
        out = tr._generate_and_score_completions({})
        ids = torch.cat([out["prompt_ids"], out["completion_ids"]], dim=1)
        am = torch.cat([out["prompt_mask"], out["completion_mask"]], dim=1)
        tr._get_per_token_logps_and_entropies(tr.ref_model, ids, am, 3)      # grad enabled but ref model
        assert flight.stats["score_calls"] == 0
        tr._get_per_token_logps_and_entropies(tr.model, ids, am, 3)
        assert flight.stats["score_calls"] == 1

    def test_compute_loss_threads_row_ids(self, flight):
        seen = {}

        class FakeBase(FakeTrainer):
            def _generate_and_score_completions(self, inputs):
                return make_output()

            def _compute_loss(self, model, inputs):
                ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
                am = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
                return self._get_per_token_logps_and_entropies(model, ids, am, inputs["completion_ids"].size(1))[0].sum()

        class Trainer(MartingaleGRPOMixin, FakeBase):
            def __init__(self):
                super().__init__()
                self.flight_recorder = flight

        tr = Trainer()
        out = tr._generate_and_score_completions({})
        perm = torch.tensor([1, 0])
        inputs = {k: (v[perm] if hasattr(v, "shape") else v) for k, v in out.items()}
        tr._compute_loss(tr.model, inputs)
        assert flight.stats["scored_tokens"] == 5 and flight.stats["unmatched_rows"] == 0
        assert tr._martingale_row_ids is None

    def test_sharded_state_dict_is_refused(self, flight):
        tr = FakeTrainer()
        tr.model.weight = nn.Parameter(torch.empty(0, 3))
        with pytest.raises(RuntimeError, match="sharded"):
            flight.on_generation(make_output(), tr)


class TestRealRunFindings:
    """2026-10-06 first vLLM run: num_iterations=2 re-scores every row; the lookup must not consume ids."""

    def _loss_batch(self, out):
        ids = torch.cat([out["prompt_ids"], out["completion_ids"]], dim=1)
        am = torch.cat([out["prompt_mask"], out["completion_mask"]], dim=1)
        return ids, am, out["completion_ids"].size(1)

    def test_same_rows_scored_again_at_a_later_step_are_matched(self, flight):
        tr = FakeTrainer()
        out = flight.on_generation(make_output(), tr)
        ids, am, keep = self._loss_batch(out)
        assert flight.on_scores(ids, am, keep, torch.zeros(2, 3), tr, row_ids=out["martingale_row_id"]) == 2
        tr.state.global_step = 1
        with torch.no_grad():
            tr.model.weight.add_(1.0)
        assert flight.on_scores(ids, am, keep, torch.zeros(2, 3), tr, row_ids=out["martingale_row_id"]) == 2
        assert flight.stats["unmatched_rows"] == 0
        d = decompose(flight.recorder.ledger)
        assert d["staleness_histogram"] == {"0": 5, "1": 5}

    def test_same_rows_same_revision_scored_once(self, flight):
        tr = FakeTrainer()
        out = flight.on_generation(make_output(), tr)
        ids, am, keep = self._loss_batch(out)
        flight.on_scores(ids, am, keep, torch.zeros(2, 3), tr, row_ids=out["martingale_row_id"])
        flight.on_scores(ids, am, keep, torch.zeros(2, 3), tr, row_ids=out["martingale_row_id"])   # grad accumulation
        seqs = list(flight.recorder.ledger.all_sequences())
        assert len(flight.recorder.ledger.scores_for(seqs[0].digest)) == 2                   # not 4
