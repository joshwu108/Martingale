"""
Tests for pipeline.py — multi-process actor/learner with pin-before-draw protocol.

Tests focus on:
- Protocol correctness: every ledgered action's revision == actor's pinned revision
- Scripted interleavings: deterministic-schedule harness
- SIGKILL cut points: recovery leaves ledger consistent
"""
import pytest
import time
import tempfile
from fractions import Fraction
from pathlib import Path

from martingale.pipeline import (
    PipelineConfig,
    run_pipeline,
    LedgerConsistencyError,
)
from martingale.revision import RevisionStore
from martingale.ledger import Ledger, GENESIS_DIGEST
from checker.verify import verify_ledger


SEED = b"pipeline-test-seed-01"


class TestPipelineBasic:
    def test_pipeline_runs_and_produces_ledger(self, tmp_path):
        """A small pipeline run produces a non-empty ledger that verifies clean."""
        cfg = PipelineConfig(
            n_actors=2,
            n_revisions=3,
            n_episodes_per_actor=2,
            n_states=2,
            n_actions=2,
            horizon=2,
            seed=SEED,
            store_dir=tmp_path / "store",
            ledger_dir=tmp_path / "ledger",
        )
        run_pipeline(cfg)

        report = verify_ledger(cfg.ledger_dir, cfg.store_dir, seed=SEED)
        assert report["total"] > 0
        assert report["failed"] == 0, f"Verification failures: {report['errors']}"

    def test_mixed_revision_trajectories_produced(self, tmp_path):
        """With rapid revision publishing, some trajectories use mixed revisions."""
        cfg = PipelineConfig(
            n_actors=2,
            n_revisions=4,
            n_episodes_per_actor=4,
            n_states=2,
            n_actions=2,
            horizon=3,
            seed=SEED,
            store_dir=tmp_path / "store",
            ledger_dir=tmp_path / "ledger",
            force_mixed_revisions=True,  # forces revision change mid-trajectory
        )
        run_pipeline(cfg)

        # Load trajectories and look for mixed-revision ones
        ledger = Ledger(cfg.ledger_dir, RevisionStore(cfg.store_dir))
        mixed = []
        for traj in ledger.all_trajectories():
            revisions_used = {r.revision_digest for r in traj.action_records}
            if len(revisions_used) > 1:
                mixed.append(traj)
        assert len(mixed) > 0, "Expected at least one mixed-revision trajectory"

        # Still verifies clean
        report = verify_ledger(cfg.ledger_dir, cfg.store_dir, seed=SEED)
        assert report["failed"] == 0


class TestProtocolInvariant:
    def test_pin_before_draw_invariant(self, tmp_path):
        """
        Protocol invariant: every ledgered action's revision == actor's pinned revision.
        Verified by the checker (which re-derives probabilities from the revision store).
        """
        cfg = PipelineConfig(
            n_actors=2,
            n_revisions=5,
            n_episodes_per_actor=3,
            n_states=2,
            n_actions=3,
            horizon=2,
            seed=SEED,
            store_dir=tmp_path / "store",
            ledger_dir=tmp_path / "ledger",
        )
        run_pipeline(cfg)
        report = verify_ledger(cfg.ledger_dir, cfg.store_dir, seed=SEED)
        assert report["failed"] == 0


class TestSIGKILLRecovery:
    def test_sigkill_leaves_consistent_ledger(self, tmp_path):
        """
        After a SIGKILL at a cut point and restart, the ledger is consistent.
        The checker must verify all trajectories that were written before the kill.
        """
        cfg = PipelineConfig(
            n_actors=1,
            n_revisions=3,
            n_episodes_per_actor=5,
            n_states=2,
            n_actions=2,
            horizon=2,
            seed=SEED,
            store_dir=tmp_path / "store",
            ledger_dir=tmp_path / "ledger",
            sigkill_after_episodes=2,  # kill after 2 complete episodes
        )
        run_pipeline(cfg)  # handles kill + recovery internally

        # Verify whatever was written before the kill
        report = verify_ledger(cfg.ledger_dir, cfg.store_dir, seed=SEED)
        assert report["failed"] == 0, f"Post-kill ledger corrupt: {report['errors']}"
