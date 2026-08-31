"""
Tests for checker/verify.py — independent ledger verifier.
Tests that valid ledgers pass, and various forgeries are rejected.
"""
import json
import pytest
from fractions import Fraction
from pathlib import Path

from martingale.draw import DrawResult, draw_action
from martingale.ledger import ActionRecord, Ledger, GENESIS_DIGEST
from martingale.revision import RevisionStore

# checker imports nothing from martingale — import it directly
from checker.verify import verify_trajectory, verify_ledger, VerificationError


SEED = b"test_checker_seed_12345"


def _build_valid_trajectory(tmp_path, n_steps=2):
    """Build a valid trajectory and return (traj_dict, store_dir, ledger_dir)."""
    store_dir = tmp_path / "store"
    ledger_dir = tmp_path / "ledger"
    store = RevisionStore(store_dir)
    ledger = Ledger(ledger_dir, store)

    probs = {0: Fraction(1, 2), 1: Fraction(1, 2)}
    rev = store.publish({0: probs})

    records = []
    prev = GENESIS_DIGEST
    for step in range(n_steps):
        draw = draw_action(probs, seed=SEED, actor_id=0, episode=0, step=step)
        rec = ActionRecord(
            revision_digest=rev.digest,
            state=0,
            action=draw.action,
            behavior_prob=probs[draw.action],
            draw=draw,
            env_next_state=0,
            env_reward=Fraction(1),
            prev_digest=prev,
        )
        records.append(rec)
        prev = rec.digest

    traj = ledger.append_trajectory(actor_id=0, episode_id=0, records=records)
    return traj.to_dict(), store_dir, ledger_dir


class TestValidLedger:
    def test_valid_trajectory_passes(self, tmp_path):
        traj_dict, store_dir, _ = _build_valid_trajectory(tmp_path)
        errors = verify_trajectory(traj_dict, store_dir, seed=SEED)
        assert errors == [], f"Valid trajectory failed: {errors}"

    def test_valid_ledger_passes(self, tmp_path):
        traj_dict, store_dir, ledger_dir = _build_valid_trajectory(tmp_path)
        report = verify_ledger(ledger_dir, store_dir, seed=SEED)
        assert report["total"] == 1
        assert report["passed"] == 1
        assert report["failed"] == 0
        assert report["errors"] == []


class TestForgeries:
    def test_tampered_action_rejected(self, tmp_path):
        """Flipping the action in a record fails draw verification."""
        traj_dict, store_dir, _ = _build_valid_trajectory(tmp_path)
        rec = traj_dict["action_records"][0]
        original_action = rec["action"]
        rec["action"] = 1 - original_action  # flip action
        # Must recompute digest too (or it fails digest check first)
        errors = verify_trajectory(traj_dict, store_dir, seed=SEED)
        assert errors, f"Tampered action should be rejected"

    def test_tampered_behavior_prob_rejected(self, tmp_path):
        """Changing behavior_prob fails probability re-derivation."""
        traj_dict, store_dir, _ = _build_valid_trajectory(tmp_path)
        rec = traj_dict["action_records"][0]
        rec["behavior_prob"] = "3/4"  # wrong prob
        errors = verify_trajectory(traj_dict, store_dir, seed=SEED)
        assert errors

    def test_tampered_digest_rejected(self, tmp_path):
        """Changing record digest fails integrity check."""
        traj_dict, store_dir, _ = _build_valid_trajectory(tmp_path)
        rec = traj_dict["action_records"][0]
        rec["digest"] = "a" * 64  # corrupt digest
        errors = verify_trajectory(traj_dict, store_dir, seed=SEED)
        assert errors

    def test_tampered_chain_rejected(self, tmp_path):
        """Breaking the prev_digest chain is detected."""
        traj_dict, store_dir, _ = _build_valid_trajectory(tmp_path, n_steps=2)
        rec = traj_dict["action_records"][1]
        rec["prev_digest"] = "b" * 64  # wrong prev
        errors = verify_trajectory(traj_dict, store_dir, seed=SEED)
        assert errors

    def test_cross_revision_forgery_rejected(self, tmp_path):
        """
        Subtle forgery: record claims revision r but draw was made under r+1,
        and both revisions assign the SAME probability to the chosen action.

        Construction: chosen action = 1 (p=1/2 in both revisions).
        Revision r:   {0: 0,   1: 1/2, 2: 1/2}  → action 1 boundary: [0, 1/2)
        Revision r+1: {0: 1/2, 1: 1/2, 2: 0}   → action 1 boundary: [1/2, 1)

        Under r+1, action=1 is drawn only when u_rat ∈ [1/2, 1).
        Under r,   action=1 maps u_rat ∈ [0, 1/2) — so any draw of action=1
        under r+1 will ALWAYS fail re-verification under r's simplex.
        100% detection.
        """
        store_dir = tmp_path / "store"
        ledger_dir = tmp_path / "ledger"
        store = RevisionStore(store_dir)
        ledger = Ledger(ledger_dir, store)

        # action 1 has same prob=1/2, but different boundaries
        probs_r  = {0: Fraction(0),   1: Fraction(1, 2), 2: Fraction(1, 2)}
        probs_r1 = {0: Fraction(1, 2), 1: Fraction(1, 2), 2: Fraction(0)}

        rev_r  = store.publish({0: probs_r})
        rev_r1 = store.publish({0: probs_r1})

        # Draw action=1 under revision r+1
        episode_to_use = None
        draw_r1 = None
        for ep in range(200):
            d = draw_action(probs_r1, seed=SEED, actor_id=0, episode=ep, step=0)
            if d.action == 1:
                draw_r1 = d
                episode_to_use = ep
                break
        if draw_r1 is None:
            pytest.skip("Could not find action=1 draw in 200 trials")

        # Forge: claim revision r but use draw from r+1
        # Both assign p(action=1) = 1/2 → probability check passes
        rec = ActionRecord(
            revision_digest=rev_r.digest,   # FORGED: claims r, but draw is under r+1
            state=0,
            action=1,
            behavior_prob=Fraction(1, 2),   # correct for both r and r+1
            draw=draw_r1,                   # drawn under r+1
            env_next_state=0,
            env_reward=Fraction(1),
            prev_digest=GENESIS_DIGEST,
        )
        traj = ledger.append_trajectory(actor_id=0, episode_id=episode_to_use, records=[rec])
        traj_dict = traj.to_dict()

        errors = verify_trajectory(traj_dict, store_dir, seed=SEED)
        # The draw_integer maps to action=1 under r+1's simplex (u_rat ∈ [1/2,1))
        # but under r's simplex, u_rat ∈ [1/2,1) maps to action=2, not action=1.
        assert errors, (
            "Cross-revision forgery with same probability should be detected "
            f"via draw_integer verification against full simplex. Errors: {errors}"
        )


class TestMultipleActors:
    def test_multiple_actors_all_pass(self, tmp_path):
        """Multiple actors' trajectories all verify correctly."""
        store_dir = tmp_path / "store"
        ledger_dir = tmp_path / "ledger"
        store = RevisionStore(store_dir)
        ledger = Ledger(ledger_dir, store)
        probs = {0: Fraction(1, 2), 1: Fraction(1, 2)}
        rev = store.publish({0: probs})

        for actor_id in range(3):
            draw = draw_action(probs, seed=SEED, actor_id=actor_id, episode=0, step=0)
            rec = ActionRecord(
                revision_digest=rev.digest,
                state=0, action=draw.action,
                behavior_prob=probs[draw.action],
                draw=draw,
                env_next_state=0, env_reward=Fraction(1),
                prev_digest=GENESIS_DIGEST,
            )
            ledger.append_trajectory(actor_id=actor_id, episode_id=0, records=[rec])

        report = verify_ledger(ledger_dir, store_dir, seed=SEED)
        assert report["passed"] == 3
        assert report["failed"] == 0
