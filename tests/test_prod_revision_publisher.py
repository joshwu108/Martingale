"""Tests for prod/revision_publisher.py — checkpoint digest and LLMRevision publishing."""
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from martingale.prod.revision_publisher import NO_TOKENIZER, RevisionPublisher, checkpoint_digest
from martingale.record import Recorder, SamplerConfig


class TestCheckpointDigest:
    def test_same_model_same_digest(self):
        model = nn.Linear(4, 2)
        assert checkpoint_digest(model) == checkpoint_digest(model)

    def test_different_weights_different_digest(self):
        assert checkpoint_digest(nn.Linear(4, 2)) != checkpoint_digest(nn.Linear(4, 2))

    def test_digest_is_hex_string(self):
        d = checkpoint_digest(nn.Linear(2, 2))
        assert isinstance(d, str) and len(d) == 64

    def test_state_dict_input(self):
        model = nn.Linear(4, 2)
        assert checkpoint_digest(model) == checkpoint_digest(model.state_dict())

    def test_digest_does_not_depend_on_torch_save_framing(self):
        """Raw-bytes digest equals the digest of an explicit (dtype, shape, bytes) triple."""
        from martingale.record import weights_digest
        model = nn.Linear(3, 1)
        triples = {k: (str(v.dtype).removeprefix("torch."), tuple(v.shape), v.contiguous().view(-1).view(torch.uint8).numpy().tobytes())
                   for k, v in model.state_dict().items()}
        assert checkpoint_digest(model) == weights_digest(triples)


class TestRevisionPublisher:
    @pytest.fixture
    def pub(self, tmp_path):
        return RevisionPublisher(Recorder(tmp_path / "ws"), tokenizer=NO_TOKENIZER,
                                 sampler=SamplerConfig(temperature=1.0))

    def test_publish_registers_revision(self, pub):
        model = nn.Linear(4, 2)
        rev = pub.publish(model)
        assert pub.recorder.ledger.has_revision(rev.digest)
        assert rev.weights_digest == checkpoint_digest(model)

    def test_publish_idempotent_per_checkpoint(self, pub):
        model = nn.Linear(4, 2)
        assert pub.publish(model).digest == pub.publish(model).digest
        assert sum(1 for _ in pub.recorder.ledger.all_revisions()) == 1

    def test_different_models_different_revisions_with_parent_chain(self, pub):
        r1 = pub.publish(nn.Linear(4, 2))
        r2 = pub.publish(nn.Linear(4, 2))
        assert r1.digest != r2.digest
        assert r2.parent_digest == r1.digest and r2.step == r1.step + 1
        assert pub.latest_digest == r2.digest

    def test_explicit_step_is_bound_into_digest(self, pub):
        model = nn.Linear(4, 2)
        a = pub.publish(model, step=10)
        b = pub.publish(model, step=11)
        assert a.digest != b.digest and a.weights_digest == b.weights_digest

    def test_sampler_and_tokenizer_are_bound(self, tmp_path):
        model = nn.Linear(2, 2)
        a = RevisionPublisher(Recorder(tmp_path / "a"), sampler=SamplerConfig(temperature=1.0)).publish(model)
        b = RevisionPublisher(Recorder(tmp_path / "b"), sampler=SamplerConfig(temperature=0.7)).publish(model)
        c = RevisionPublisher(Recorder(tmp_path / "c"), tokenizer=b"tokenizer.json bytes").publish(model)
        assert len({a.digest, b.digest, c.digest}) == 3
