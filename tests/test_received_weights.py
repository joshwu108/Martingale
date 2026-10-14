"""2026-10-13: the rollout replica's digest of the weights it received, bound into the revision.

Record: `received_weights_digest` is an optional manifest key inside the digest, absent when the
replica did not report one, so manifests written before it existed digest unchanged. Absent means
"no sync was observed", never "the sync matched". `ReceivedWeights` computes `weights_digest`'s
formula over (name, tensor) pairs arriving in buckets in any order. Checker: accepts both forms,
rejects an explicit null, a malformed value and a tampered one. Doctor: a revision whose received
digest differs from its weights digest raises an exact `stale_server` error, with no floor needed.
"""
import pytest

from checker.verify_tokens import verify_revision_manifest
from martingale.diagnostics import AlarmConfig, AlarmHistory, IncrementalDiagnosis, evaluate
from martingale.record import LLMRevision, ReceivedWeights, Recorder, SamplerConfig, weights_digest
from martingale.record.bits import digest_json

torch = pytest.importorskip("torch")

W, T, R = "a" * 64, "b" * 64, "c" * 64
SAMPLER = SamplerConfig(temperature=1.0, logprobs_source="engine_sampling")


def state():
    return {"layer.w": torch.arange(6, dtype=torch.float32).reshape(2, 3), "layer.b": torch.ones(3),
            "emb": torch.tensor([[1, 2]], dtype=torch.int64)}


class TestRevision:
    def test_absent_when_unreported_so_old_manifests_digest_unchanged(self):
        rev = LLMRevision(W, T, SAMPLER, step=2)
        assert rev.received_weights_digest is None
        assert "received_weights_digest" not in rev.manifest()
        old = {"kind": "llm", "weights_digest": W, "tokenizer_digest": T, "sampler": SAMPLER.to_dict(),
               "parent_digest": "0" * 64, "step": 2}
        assert rev.digest == digest_json(old)

    def test_inside_the_digest_and_round_trips(self):
        plain = LLMRevision(W, T, SAMPLER, step=2)
        rev = LLMRevision(W, T, SAMPLER, step=2, received_weights_digest=R)
        assert rev.digest != plain.digest
        assert rev.manifest()["received_weights_digest"] == R
        assert LLMRevision.from_dict(rev.to_dict()) == rev
        assert LLMRevision(W, T, SAMPLER, step=2, received_weights_digest=W).digest != rev.digest

    @pytest.mark.parametrize("bad", ["", "C" * 64, "c" * 63, 7])
    def test_malformed_is_refused(self, bad):
        with pytest.raises(ValueError, match="received_weights_digest"):
            LLMRevision(W, T, SAMPLER, step=0, received_weights_digest=bad)

    def test_sync_verdict(self):
        assert LLMRevision(W, T, SAMPLER, step=0).sync_verified is None
        assert LLMRevision(W, T, SAMPLER, step=0, received_weights_digest=W).sync_verified is True
        assert LLMRevision(W, T, SAMPLER, step=0, received_weights_digest=R).sync_verified is False


class TestReceivedWeights:
    def test_equals_weights_digest_in_any_bucket_order(self):
        s = state()
        received = ReceivedWeights()
        received.add([("layer.b", s["layer.b"]), ("emb", s["emb"])])
        received.add([("layer.w", s["layer.w"])])
        assert received.digest() == weights_digest(s)
        assert received.tensors == 3

    def test_copies_bytes_on_arrival(self):
        """The IPC bucket buffer is reused after the callback; a later overwrite must not change the digest."""
        s = state()
        buffer = s["layer.w"].clone()
        received = ReceivedWeights()
        received.add([("layer.w", buffer)])
        buffer.zero_()
        received.add([("layer.b", s["layer.b"]), ("emb", s["emb"])])
        assert received.digest() == weights_digest(s)

    def test_nothing_received_reports_nothing(self):
        assert ReceivedWeights().digest() is None

    def test_a_name_received_twice_is_refused(self):
        received = ReceivedWeights()
        received.add([("emb", torch.ones(1))])
        with pytest.raises(ValueError, match="twice"):
            received.add([("emb", torch.ones(1))])

    def test_a_missing_or_changed_tensor_changes_the_digest(self):
        s = state()
        partial = ReceivedWeights()
        partial.add([(k, v) for k, v in s.items() if k != "emb"])
        changed = ReceivedWeights()
        changed.add([(k, v + 1 if k == "layer.b" else v) for k, v in s.items()])
        assert weights_digest(s) not in (partial.digest(), changed.digest())


class TestChecker:
    def test_both_forms_verify(self):
        for rev in (LLMRevision(W, T, SAMPLER, step=1), LLMRevision(W, T, SAMPLER, step=1, received_weights_digest=R)):
            assert verify_revision_manifest(rev.to_dict()) == []

    def test_tampered_value_breaks_the_digest(self):
        d = LLMRevision(W, T, SAMPLER, step=1, received_weights_digest=R).to_dict()
        d["received_weights_digest"] = W                 # a forger claiming the sync matched
        assert any("digest mismatch" in e for e in verify_revision_manifest(d))

    def test_dropping_it_breaks_the_digest(self):
        d = LLMRevision(W, T, SAMPLER, step=1, received_weights_digest=R).to_dict()
        del d["received_weights_digest"]
        assert any("digest mismatch" in e for e in verify_revision_manifest(d))

    @pytest.mark.parametrize("bad", [None, "C" * 64, 1])
    def test_null_or_malformed_is_rejected(self, bad):
        m = {"kind": "llm", "weights_digest": W, "tokenizer_digest": T, "sampler": SAMPLER.to_dict(),
             "parent_digest": "0" * 64, "step": 1, "received_weights_digest": bad}
        errs = verify_revision_manifest({**m, "digest": digest_json(m)})
        assert any("received_weights_digest" in e for e in errs)


def _generate(rec: Recorder, rev: LLMRevision, actor: int = 0) -> None:
    with rec.sequence(actor, f"s{rev.step}", [1, 2]) as seq:
        seq.token(rev.digest, 5, -0.5)


class TestAlarm:
    def test_mismatch_is_an_exact_stale_server_error(self, tmp_path):
        rec = Recorder(tmp_path / "ws")
        rev = rec.publish_revision(W, T, SAMPLER, step=3, received_weights_digest=R)
        _generate(rec, rev)
        diag = IncrementalDiagnosis(rec.ledger).update(3)
        assert diag.received_mismatches == [{"revision_step": 3, "weights_digest": W, "received_weights_digest": R}]
        alarms = evaluate(diag, AlarmHistory(), AlarmConfig())
        stale = [a for a in alarms if a.kind == "stale_server"]
        assert len(stale) == 1 and stale[0].level == "error"
        assert stale[0].evidence["observed"] is True
        assert stale[0].evidence["received_mismatches"] == diag.received_mismatches
        assert diag.metrics()["martingale/received_weights_mismatches"] == 1.0

    def test_match_and_unreported_raise_nothing(self, tmp_path):
        rec = Recorder(tmp_path / "ws")
        _generate(rec, rec.publish_revision(W, T, SAMPLER, step=0, received_weights_digest=W), actor=0)
        _generate(rec, rec.publish_revision(W, T, SAMPLER, step=1), actor=1)
        diag = IncrementalDiagnosis(rec.ledger).update(1)
        assert diag.received_mismatches == [] and diag.unverified_generation_revisions == 1
        assert diag.metrics()["martingale/unverified_generation_revisions"] == 1.0   # absence is visible
        assert not [a for a in evaluate(diag, AlarmHistory(), AlarmConfig()) if a.kind == "stale_server"]
