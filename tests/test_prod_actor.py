"""Tests for prod/actor.py — AsyncActor over the token record."""
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from martingale.prod.actor import AsyncActor, PinnedContext, ProtocolViolation
from martingale.prod.revision_publisher import RevisionPublisher
from martingale.record import Recorder, bits_to_fraction


@pytest.fixture
def setup(tmp_path):
    rec = Recorder(tmp_path / "ws")
    pub = RevisionPublisher(rec)
    return rec, pub, AsyncActor(actor_id=0, recorder=rec)


class TestPinnedContext:
    def test_context_manager_provides_revision_digest(self, setup):
        rec, pub, actor = setup
        rev = pub.publish(nn.Linear(4, 2))
        with actor.pin_revision(rev.digest) as ctx:
            assert isinstance(ctx, PinnedContext) and ctx.revision_digest == rev.digest

    def test_cannot_draw_outside_context(self, setup):
        _, _, actor = setup
        with pytest.raises(ProtocolViolation):
            actor.sample_and_record(torch.zeros(2), episode_id=0, step=0)

    def test_cannot_pin_unpublished_revision(self, setup):
        _, _, actor = setup
        with pytest.raises(ProtocolViolation):
            actor.pin_revision("9" * 64)

    def test_record_uses_pinned_revision_and_exact_logprob_bits(self, setup):
        rec, pub, actor = setup
        rev = pub.publish(nn.Linear(4, 2))
        logits = torch.tensor([0.3, -1.2])
        with actor.pin_revision(rev.digest) as ctx:
            out = actor.sample_and_record(logits, episode_id=0, step=0)
            ctx.commit_episode(0)
        assert out["revision_digest"] == rev.digest and out["action"] in (0, 1)
        seq = next(rec.ledger.all_sequences())
        tok = seq.tokens[0]
        assert tok.token_id == out["action"]
        want = float(torch.log_softmax(logits, -1)[out["action"]].to(torch.float32))
        assert float(bits_to_fraction(tok.logprob_bits)) == pytest.approx(want, abs=0.0)  # bits, not rounding

    def test_pin_released_after_context(self, setup):
        _, pub, actor = setup
        rev = pub.publish(nn.Linear(4, 2))
        with actor.pin_revision(rev.digest):
            pass
        with pytest.raises(ProtocolViolation):
            actor.sample_and_record(torch.zeros(2))

    def test_episode_written_with_reward(self, setup):
        rec, pub, actor = setup
        rev = pub.publish(nn.Linear(4, 2))
        with actor.pin_revision(rev.digest) as ctx:
            for step in range(3):
                actor.sample_and_record(torch.randn(2), episode_id=0, step=step)
            ctx.commit_episode(0, reward=2.5)
        seq = next(rec.ledger.all_sequences())
        assert len(seq.tokens) == 3 and all(t.revision_digest == rev.digest for t in seq.tokens)
        assert bits_to_fraction(seq.reward_bits) == 2.5

    def test_mixed_revision_episode(self, setup):
        rec, pub, actor = setup
        rev1 = pub.publish(nn.Linear(4, 2))
        rev2 = pub.publish(nn.Linear(4, 2))
        with actor.pin_revision(rev1.digest):
            actor.sample_and_record(torch.randn(2), episode_id=0, step=0)
        with actor.pin_revision(rev2.digest) as ctx:
            actor.sample_and_record(torch.randn(2), episode_id=0, step=1)
            ctx.commit_episode(0)
        seq = next(rec.ledger.all_sequences())
        assert seq.is_mixed_revision and seq.revision_digests == (rev1.digest, rev2.digest)

    def test_commit_without_steps_and_abort(self, setup):
        rec, pub, actor = setup
        rev = pub.publish(nn.Linear(4, 2))
        with actor.pin_revision(rev.digest) as ctx:
            with pytest.raises(ProtocolViolation):
                ctx.commit_episode(7)
            actor.sample_and_record(torch.randn(2), episode_id=1)
            actor.abort_episode(1)
        assert rec.ledger.count_sequences() == 0
