"""Tests for prod/actor.py — AsyncActor with pin_revision context manager."""
import pytest
import torch
import torch.nn as nn
from fractions import Fraction
from martingale.prod.actor import AsyncActor, PinnedContext, ProtocolViolation
from martingale.prod.revision_publisher import RevisionPublisher
from martingale.store.sqlite import SQLiteRevisionStore, SQLiteLedger


def _make_policy():
    """Simple 1-layer softmax policy."""
    model = nn.Linear(4, 2)
    return model


class TestPinnedContext:
    def test_context_manager_provides_revision_digest(self, tmp_path):
        store = SQLiteRevisionStore(tmp_path / "rev.db")
        publisher = RevisionPublisher(store)
        model = _make_policy()
        rev = publisher.publish(model)
        ledger = SQLiteLedger(tmp_path / "ledger.db", store)
        actor = AsyncActor(actor_id=0, ledger=ledger, seed=b"test")

        with actor.pin_revision(rev.digest) as ctx:
            assert isinstance(ctx, PinnedContext)
            assert ctx.revision_digest == rev.digest

    def test_cannot_draw_outside_context(self, tmp_path):
        store = SQLiteRevisionStore(tmp_path / "rev.db")
        ledger = SQLiteLedger(tmp_path / "ledger.db", store)
        actor = AsyncActor(actor_id=0, ledger=ledger, seed=b"test")
        obs = torch.zeros(4)
        with pytest.raises(ProtocolViolation, match="pin"):
            actor.sample_and_record(obs, log_probs=torch.tensor([-0.5, -0.5]))

    def test_record_uses_pinned_revision(self, tmp_path):
        store = SQLiteRevisionStore(tmp_path / "rev.db")
        publisher = RevisionPublisher(store)
        model = _make_policy()
        rev = publisher.publish(model)
        ledger = SQLiteLedger(tmp_path / "ledger.db", store)
        actor = AsyncActor(actor_id=0, ledger=ledger, seed=b"test")

        log_probs = torch.tensor([-0.5, -0.8])
        with actor.pin_revision(rev.digest) as ctx:
            result = actor.sample_and_record(
                obs=torch.zeros(4),
                log_probs=log_probs,
                episode_id=0,
                step=0,
            )
        assert result["revision_digest"] == rev.digest
        assert result["action"] in [0, 1]

    def test_pin_released_after_context(self, tmp_path):
        """After the context exits, sampling raises ProtocolViolation again."""
        store = SQLiteRevisionStore(tmp_path / "rev.db")
        publisher = RevisionPublisher(store)
        rev = publisher.publish(_make_policy())
        ledger = SQLiteLedger(tmp_path / "ledger.db", store)
        actor = AsyncActor(actor_id=0, ledger=ledger, seed=b"test")

        with actor.pin_revision(rev.digest):
            pass  # context exits here

        with pytest.raises(ProtocolViolation, match="pin"):
            actor.sample_and_record(torch.zeros(4), torch.tensor([-0.5, -0.5]))

    def test_episode_ledger_written(self, tmp_path):
        """After a full episode, trajectory is written to the ledger."""
        store = SQLiteRevisionStore(tmp_path / "rev.db")
        publisher = RevisionPublisher(store)
        rev = publisher.publish(_make_policy())
        ledger = SQLiteLedger(tmp_path / "ledger.db", store)
        actor = AsyncActor(actor_id=0, ledger=ledger, seed=b"test")

        with actor.pin_revision(rev.digest) as ctx:
            for step in range(3):
                actor.sample_and_record(
                    obs=torch.zeros(4),
                    log_probs=torch.tensor([-0.5, -0.8]),
                    episode_id=0,
                    step=step,
                )
            ctx.commit_episode(episode_id=0)

        traj = ledger.load_trajectory(actor_id=0, episode_id=0)
        assert len(traj.action_records) == 3
        for rec in traj.action_records:
            assert rec.revision_digest == rev.digest

    def test_mixed_revision_episode(self, tmp_path):
        """Mid-episode revision switch is first-class: each step pins its own revision."""
        store = SQLiteRevisionStore(tmp_path / "rev.db")
        publisher = RevisionPublisher(store)
        rev1 = publisher.publish(_make_policy())
        rev2 = publisher.publish(_make_policy())
        ledger = SQLiteLedger(tmp_path / "ledger.db", store)
        actor = AsyncActor(actor_id=0, ledger=ledger, seed=b"test")

        with actor.pin_revision(rev1.digest) as ctx:
            actor.sample_and_record(torch.zeros(4), torch.tensor([-0.5, -0.5]),
                                    episode_id=0, step=0)

        with actor.pin_revision(rev2.digest) as ctx:
            actor.sample_and_record(torch.zeros(4), torch.tensor([-0.5, -0.5]),
                                    episode_id=0, step=1)
            ctx.commit_episode(episode_id=0)

        traj = ledger.load_trajectory(actor_id=0, episode_id=0)
        assert traj.action_records[0].revision_digest == rev1.digest
        assert traj.action_records[1].revision_digest == rev2.digest
