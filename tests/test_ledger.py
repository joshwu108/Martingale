"""Tests for ledger.py — hash-chained trajectory attestations."""
import pytest
from fractions import Fraction
from martingale.revision import RevisionStore, Revision
from martingale.ledger import Ledger, ActionRecord, TrajectoryRecord
from martingale.draw import DrawResult


def _make_rev(store, probs):
    return store.publish({0: probs})


class TestActionRecord:
    def test_construction(self, tmp_path):
        store = RevisionStore(tmp_path)
        rev = _make_rev(store, {0: Fraction(1, 2), 1: Fraction(1, 2)})
        draw = DrawResult(action=0, draw_integer=12345, rejection_count=0)
        rec = ActionRecord(
            revision_digest=rev.digest,
            state=0,
            action=0,
            behavior_prob=Fraction(1, 2),
            draw=draw,
            env_next_state=0,
            env_reward=Fraction(1),
            prev_digest="0" * 64,
        )
        assert rec.revision_digest == rev.digest
        assert rec.behavior_prob == Fraction(1, 2)

    def test_digest_is_deterministic(self, tmp_path):
        store = RevisionStore(tmp_path)
        rev = _make_rev(store, {0: Fraction(1, 2), 1: Fraction(1, 2)})
        draw = DrawResult(action=0, draw_integer=99, rejection_count=0)
        rec = ActionRecord(
            revision_digest=rev.digest,
            state=0, action=0,
            behavior_prob=Fraction(1, 2),
            draw=draw,
            env_next_state=0, env_reward=Fraction(1),
            prev_digest="0" * 64,
        )
        assert isinstance(rec.digest, str)
        assert len(rec.digest) == 64
        # Same inputs → same digest
        rec2 = ActionRecord(
            revision_digest=rev.digest,
            state=0, action=0,
            behavior_prob=Fraction(1, 2),
            draw=draw,
            env_next_state=0, env_reward=Fraction(1),
            prev_digest="0" * 64,
        )
        assert rec.digest == rec2.digest


class TestLedger:
    def _setup(self, tmp_path):
        store = RevisionStore(tmp_path / "store")
        ledger = Ledger(tmp_path / "ledger", revision_store=store)
        rev = store.publish({0: {0: Fraction(1, 2), 1: Fraction(1, 2)}})
        return store, ledger, rev

    def test_append_trajectory(self, tmp_path):
        store, ledger, rev = self._setup(tmp_path)
        draw = DrawResult(action=0, draw_integer=42, rejection_count=0)
        rec = ActionRecord(
            revision_digest=rev.digest,
            state=0, action=0,
            behavior_prob=Fraction(1, 2),
            draw=draw,
            env_next_state=0, env_reward=Fraction(1),
            prev_digest=ledger.genesis_digest,
        )
        traj = ledger.append_trajectory(actor_id=0, episode_id=0, records=[rec])
        assert isinstance(traj, TrajectoryRecord)
        assert len(traj.action_records) == 1

    def test_chain_integrity(self, tmp_path):
        """Each record's prev_digest points to the previous record's digest."""
        store, ledger, rev = self._setup(tmp_path)
        recs = []
        prev = ledger.genesis_digest
        for i in range(3):
            draw = DrawResult(action=i % 2, draw_integer=i * 100, rejection_count=0)
            rec = ActionRecord(
                revision_digest=rev.digest,
                state=i % 2, action=i % 2,
                behavior_prob=Fraction(1, 2),
                draw=draw,
                env_next_state=(i + 1) % 2, env_reward=Fraction(i),
                prev_digest=prev,
            )
            recs.append(rec)
            prev = rec.digest
        traj = ledger.append_trajectory(actor_id=0, episode_id=0, records=recs)
        # Verify chain
        chain_prev = ledger.genesis_digest
        for rec in traj.action_records:
            assert rec.prev_digest == chain_prev
            chain_prev = rec.digest

    def test_mixed_revision_trajectory(self, tmp_path):
        """A trajectory can use different revisions at different steps."""
        store, ledger, rev = self._setup(tmp_path)
        rev2 = store.publish({0: {0: Fraction(3, 4), 1: Fraction(1, 4)}})
        draw0 = DrawResult(action=0, draw_integer=10, rejection_count=0)
        draw1 = DrawResult(action=1, draw_integer=20, rejection_count=0)
        rec0 = ActionRecord(
            revision_digest=rev.digest,
            state=0, action=0, behavior_prob=Fraction(1, 2),
            draw=draw0, env_next_state=0, env_reward=Fraction(1),
            prev_digest=ledger.genesis_digest,
        )
        rec1 = ActionRecord(
            revision_digest=rev2.digest,  # different revision!
            state=0, action=1, behavior_prob=Fraction(1, 4),
            draw=draw1, env_next_state=0, env_reward=Fraction(2),
            prev_digest=rec0.digest,
        )
        traj = ledger.append_trajectory(actor_id=0, episode_id=0, records=[rec0, rec1])
        assert traj.action_records[0].revision_digest == rev.digest
        assert traj.action_records[1].revision_digest == rev2.digest
