"""Replayed rows in the token record: ReplayProvenance, SequenceRecord provenance, the ledger's
fail-closed origin checks, Recorder.replay_from / Recorder.replayed, head and checker round trips."""
from fractions import Fraction

import pytest

from checker.verify_tokens import verify_export
from martingale.record import ProtocolViolation, Recorder, SamplerConfig, SequenceRecord, bits_to_fraction, float_bits
from martingale.record.replay import ReplayProvenance

SAMPLER = SamplerConfig(temperature=1.0, top_p=1.0, engine="fake", engine_version="0", logprobs_mode="raw")
W1, W2, W3, TOK = "1" * 64, "2" * 64, "3" * 64, "f" * 64
CD = "c" * 64


@pytest.fixture
def rec(tmp_path):
    return Recorder(tmp_path / "ws")


def _fresh(rec):
    """Three revisions; two fresh sequences under r0 on actor 0 (the second has top-k), one under r1 on actor 1."""
    r0 = rec.publish_revision(W1, TOK, SAMPLER, step=0)
    r1 = rec.publish_revision(W2, TOK, SAMPLER, step=1)
    r3 = rec.publish_revision(W3, TOK, SAMPLER, step=3)
    with rec.sequence(0, "step0/row1", [1, 2, 4]) as b:
        b.token(r0.digest, 10, -0.25); b.token(r0.digest, 12, -0.75)
    with rec.sequence(0, "step0/row0", [1, 2, 3]) as a:
        a.token(r0.digest, 10, -0.5); a.token(r0.digest, 11, -1.5, topk=[(11, -1.5), (3, -2.0)])
    with rec.sequence(1, "step1/row2", [9]) as c:
        c.token(r1.digest, 7, -0.1)
    return r0, r1, r3, a.record, b.record, c.record


class TestReplayProvenance:
    def test_exact_weights_and_canonical(self):
        p = ReplayProvenance.of("17/3", CD, is_weight=Fraction(2, 6), rescale=Fraction(3, 4), origin_digest="a" * 64)
        assert (p.is_weight_num, p.is_weight_den) == (1, 3)
        assert p.is_weight == Fraction(1, 3) and p.rescale == Fraction(3, 4) and p.applied_weight == Fraction(1, 4)
        assert p.canonical() == {"draw_id": "17/3", "content_digest": CD, "is_weight_num": 1, "is_weight_den": 3,
                                 "rescale_num": 3, "rescale_den": 4, "origin_digest": "a" * 64}
        assert ReplayProvenance.from_dict(p.to_dict()) == p

    def test_float_weight_is_stored_exactly(self):
        p = ReplayProvenance.of("d", CD, is_weight=0.1)
        assert p.is_weight == Fraction(0.1)            # the float's exact dyadic value, not 1/10
        assert ReplayProvenance.of("d", CD, is_weight=1).rescale == 1

    @pytest.mark.parametrize("kw", [
        dict(draw_id=""), dict(draw_id=3), dict(content_digest="zz"), dict(content_digest=CD.upper()),
        dict(is_weight=-1), dict(is_weight=float("nan")), dict(is_weight=float("inf")), dict(is_weight=True),
        dict(rescale=-1), dict(rescale=True), dict(rescale=0), dict(origin_digest="short"),
    ])
    def test_malformed_rejected(self, kw):
        base = dict(draw_id="d", content_digest=CD, is_weight=1)
        base.update(kw)
        with pytest.raises((ValueError, TypeError)):
            ReplayProvenance.of(**base)

    @pytest.mark.parametrize("patch", [
        {"is_weight_num": 2, "is_weight_den": 4}, {"is_weight_den": 0}, {"is_weight_num": True}, {"rescale_den": -1},
        {"rescale_num": 0}, {"draw_id": None}, {"origin_digest": 5}, {"extra": 1},
    ])
    def test_from_dict_rejects_unreduced_bool_negative_and_unknown(self, patch):
        d = {**ReplayProvenance.of("d", CD, is_weight=1).to_dict(), **patch}
        with pytest.raises(ValueError):
            ReplayProvenance.from_dict(d)


class TestSequenceProvenance:
    def test_fresh_records_are_unchanged(self, rec):
        _, _, _, a, _, _ = _fresh(rec)
        d = a.to_dict()
        assert "provenance" not in d and "replay" not in d and a.provenance == "fresh" and a.replay is None
        assert SequenceRecord.from_dict(d) == a
        assert SequenceRecord.from_dict({**d, "provenance": "fresh", "replay": None}) == a   # explicit form accepted

    def test_replay_from_copies_bits_and_revisions_and_binds_provenance(self, rec):
        r0, r1, r3, a, b, c = _fresh(rec)
        s = rec.replay_from(a.digest, actor_id=0, sequence_id="step3/replay0", draw_id="5/0", content_digest=CD,
                            is_weight=Fraction(1, 2), rescale=Fraction(3, 4), reward=0.25)
        assert s.provenance == "replayed" and s.replay.origin_digest == a.digest
        assert [(t.token_id, t.revision_digest, t.logprob_bits, t.topk) for t in s.tokens] == \
            [(t.token_id, t.revision_digest, t.logprob_bits, t.topk) for t in a.tokens]
        assert s.prompt_digest == a.prompt_digest and s.prompt_ids == a.prompt_ids and s.prompt_len == a.prompt_len
        assert bits_to_fraction(s.reward_bits) == Fraction(1, 4) and s.digest != a.digest
        assert s.sequence_index == 2 and s.prev_sequence_digest == a.digest       # actor 0 chain continues
        d = s.to_dict()
        assert d["provenance"] == "replayed" and d["replay"]["draw_id"] == "5/0"
        assert SequenceRecord.from_dict(d) == s
        assert rec.ledger.get_sequence(s.digest).replay == s.replay

    def test_digest_covers_the_replay_block_and_the_provenance(self, rec):
        _, _, _, a, _, _ = _fresh(rec)
        s = rec.replay_from(a.digest, actor_id=1, sequence_id="x", draw_id="1", content_digest=CD, is_weight=1)
        d = s.to_dict(); d["replay"]["draw_id"] = "2"
        with pytest.raises(ValueError, match="digest mismatch"):
            SequenceRecord.from_dict(d)
        d = s.to_dict(); d["provenance"] = "fresh"
        with pytest.raises(ValueError):
            SequenceRecord.from_dict(d)
        d = s.to_dict(); d["replay"] = None
        with pytest.raises(ValueError):
            SequenceRecord.from_dict(d)

    def test_explicit_form_without_origin(self, rec):
        r0, r1, r3, a, b, c = _fresh(rec)
        s = rec.replayed(actor_id=1, sequence_id="restored", prompt_ids=[4, 4],
                         steps=[(r0.digest, 10, -0.5), (r0.digest, 11, float_bits(-1.5))],
                         draw_id="7", content_digest=CD, is_weight=Fraction(1))
        assert s.provenance == "replayed" and s.replay.origin_digest is None and s.replay.rescale == 1
        assert [bits_to_fraction(t.logprob_bits) for t in s.tokens] == [Fraction(-0.5), Fraction(-1.5)]
        assert rec.ledger.count_replayed_sequences() == 1 and rec.ledger.count_sequences() == 4

    def test_explicit_form_refuses_unpublished_revision_and_empty_steps(self, rec):
        _fresh(rec)
        with pytest.raises(ProtocolViolation):
            rec.replayed(actor_id=1, sequence_id="r", prompt_ids=[1], steps=[("9" * 64, 1, -0.1)],
                         draw_id="7", content_digest=CD, is_weight=1)
        with pytest.raises(ProtocolViolation):
            rec.replayed(actor_id=1, sequence_id="r", prompt_ids=[1], steps=[], draw_id="7", content_digest=CD, is_weight=1)
        assert rec.ledger.count_sequences() == 3


class TestLedgerFailsClosed:
    def test_origin_must_exist_and_be_fresh(self, rec):
        _, _, _, a, _, _ = _fresh(rec)
        with pytest.raises(KeyError, match="origin"):
            rec.replay_from("e" * 64, actor_id=0, sequence_id="x", draw_id="1", content_digest=CD, is_weight=1)
        s = rec.replay_from(a.digest, actor_id=0, sequence_id="x", draw_id="1", content_digest=CD, is_weight=1)
        with pytest.raises(ValueError, match="fresh"):
            rec.replay_from(s.digest, actor_id=0, sequence_id="y", draw_id="2", content_digest=CD, is_weight=1)
        assert rec.ledger.count_replayed_sequences() == 1 and not rec.ledger._con.in_transaction

    def test_explicit_rows_claiming_an_origin_must_match_it(self, rec):
        r0, _, _, a, _, _ = _fresh(rec)
        kw = dict(draw_id="1", content_digest=CD, is_weight=1, origin_digest=a.digest)
        good = [(r0.digest, 10, -0.5), (r0.digest, 11, -1.5, [(11, -1.5), (3, -2.0)])]
        ok = rec.replayed(actor_id=1, sequence_id="ok", prompt_ids=[1, 2, 3], steps=good, **kw)
        assert ok.replay.origin_digest == a.digest
        bad_bits = [(r0.digest, 10, -0.5), (r0.digest, 11, -1.5000001, [(11, -1.5), (3, -2.0)])]
        with pytest.raises(ValueError, match="differ"):
            rec.replayed(actor_id=1, sequence_id="bad", prompt_ids=[1, 2, 3], steps=bad_bits, **kw)
        with pytest.raises(ValueError, match="differ"):
            rec.replayed(actor_id=1, sequence_id="bad", prompt_ids=[1, 2, 9], steps=good, **kw)
        with pytest.raises(ValueError, match="differ"):
            rec.replayed(actor_id=1, sequence_id="bad", prompt_ids=[1, 2, 3], steps=good[:1], **kw)
        assert rec.ledger.count_replayed_sequences() == 1 and not rec.ledger._con.in_transaction

    def test_find_fresh_sequences_by_prompt_and_completion(self, rec):
        r0, r1, r3, a, b, c = _fresh(rec)
        rec.replay_from(a.digest, actor_id=0, sequence_id="x", draw_id="1", content_digest=CD, is_weight=1)
        found = rec.ledger.find_fresh_sequences(a.prompt_digest, [10, 11])
        assert [s.digest for s in found] == [a.digest]                  # the replayed copy is not an origin
        assert rec.ledger.find_fresh_sequences(a.prompt_digest, [10, 99]) == []
        assert [s.digest for s in rec.ledger.find_fresh_sequences(a.prompt_digest)] == [a.digest]
        assert [s.digest for s in rec.ledger.find_fresh_sequences(b.prompt_digest)] == [b.digest]
        assert rec.ledger.find_fresh_sequences("0" * 64) == []


class TestHeadAndChecker:
    def test_replayed_rows_move_the_head_and_verify(self, rec, tmp_path):
        r0, r1, r3, a, b, c = _fresh(rec)
        h0 = rec.ledger.head()
        s = rec.replay_from(a.digest, actor_id=1, sequence_id="step3/replay0", draw_id="1", content_digest=CD,
                            is_weight=Fraction(1, 2))
        h1 = rec.ledger.head()
        assert h1 != h0
        rec.score(s.digest, r3.digest, [-0.6, -1.4])
        rec.score(a.digest, r0.digest, [-0.5, -1.5])
        report = verify_export(rec.export_for_checker(tmp_path / "exp"), expected_head=rec.ledger.head())
        assert report["ok"], report["errors"]
        assert report["n_replayed_sequences"] == 1 and report["n_sequences"] == 4 and report["n_tokens"] == 7

    def test_reopen_round_trips_replay(self, tmp_path):
        rec = Recorder(tmp_path / "ws")
        _, _, _, a, _, _ = _fresh(rec)
        s = rec.replay_from(a.digest, actor_id=1, sequence_id="x", draw_id="1", content_digest=CD, is_weight=1)
        rec.ledger.close()
        rec2 = Recorder(tmp_path / "ws")
        got = rec2.ledger.get_sequence(s.digest)
        assert got == s and got.provenance == "replayed"
        assert rec2.ledger.count_replayed_sequences() == 1
        assert [x.digest for x in rec2.ledger.find_fresh_sequences(a.prompt_digest, [10, 11])] == [a.digest]
