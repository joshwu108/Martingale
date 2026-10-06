"""
Production ledger must be verifiable by the independent checker.

History: at HEAD 5333258 (2026-10-05) every trajectory written by the prototype
prod.AsyncActor was rejected by the checker, because RevisionPublisher stored a
fake 1x1 table and the actor wrote the state hash as the draw integer. This test
was a strict xfail pinning that defect. On 2026-10-06 the prototype was replaced
by the token record (docs/design.md section 7) and the test now asserts the
production path passes checker/verify_tokens.py. Never delete this test.
"""
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


def test_prod_actor_ledger_passes_independent_checker(tmp_path):
    from checker.verify_tokens import verify_export
    from martingale.prod.actor import AsyncActor
    from martingale.prod.revision_publisher import RevisionPublisher
    from martingale.record import Recorder, SamplerConfig

    # Arrange
    rec = Recorder(tmp_path / "ws")
    pub = RevisionPublisher(rec, tokenizer=b"tok", sampler=SamplerConfig(temperature=1.0, engine="ref"))
    actor = AsyncActor(actor_id=0, recorder=rec)
    r1, r2 = pub.publish(nn.Linear(4, 3)), pub.publish(nn.Linear(4, 3))

    # Act: two episodes, the second mixed-revision, then trainer-side scores
    with actor.pin_revision(r1.digest) as ctx:
        for step in range(3):
            actor.sample_and_record(torch.log_softmax(torch.randn(3), -1), episode_id=0, step=step)
        ctx.commit_episode(0, reward=1.0)
    with actor.pin_revision(r1.digest):
        actor.sample_and_record(torch.randn(3), episode_id=1, step=0)
    with actor.pin_revision(r2.digest) as ctx:
        actor.sample_and_record(torch.randn(3), episode_id=1, step=1)
        ctx.commit_episode(1, reward=0.0)
    seqs = list(rec.ledger.all_sequences())
    rec.score(seqs[0].digest, r2.digest, [-0.9, -1.1, -0.2])
    report = verify_export(rec.export_for_checker(tmp_path / "exp"), expected_head=rec.ledger.head())

    # Assert
    assert report["ok"], report["errors"]
    assert (report["n_revisions"], report["n_sequences"], report["n_tokens"]) == (2, 2, 5)
