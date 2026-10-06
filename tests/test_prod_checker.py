"""
Production ledger must be verifiable by the independent checker.

Phase 0 truth pass (planning/2026-10-05-phase1-researcher-tool.md, section 0.1):
at HEAD 5333258 every trajectory written by prod.AsyncActor is rejected by
checker.verify because RevisionPublisher stores a fake 1x1 table and the actor
writes the state hash as the draw integer.

The test is marked strict xfail so the suite stays green while documenting the
defect. When Phase 1 fixes the record schema this test will XPASS, which strict
mode reports as a failure: remove the marker then. Never delete the test.
"""
import json

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Phase 0 defect 0.1: prod ledger is not verifiable by checker/verify.py",
)
def test_prod_actor_ledger_passes_independent_checker(tmp_path):
    from checker.verify import verify_trajectory
    from martingale.prod.actor import AsyncActor
    from martingale.prod.revision_publisher import RevisionPublisher
    from martingale.store.sqlite import SQLiteLedger, SQLiteRevisionStore

    # Arrange
    store = SQLiteRevisionStore(tmp_path / "revisions.db")
    ledger = SQLiteLedger(tmp_path / "ledger.db", store)
    rev = RevisionPublisher(store).publish(nn.Linear(4, 3))
    actor = AsyncActor(actor_id=0, ledger=ledger, seed=b"phase0")

    # Act
    with actor.pin_revision(rev.digest) as ctx:
        for step in range(3):
            actor.sample_and_record(
                torch.randn(4), torch.log_softmax(torch.randn(3), -1),
                episode_id=0, step=step,
            )
        ctx.commit_episode(0)
    ledger_dir, store_dir = tmp_path / "ledger_export", tmp_path / "store_export"
    ledger.export_for_checker(ledger_dir, store_dir)
    traj_file = next(ledger_dir.rglob("*.json"))
    errors = verify_trajectory(json.loads(traj_file.read_text()), store_dir, b"phase0")

    # Assert
    assert errors == [], errors
