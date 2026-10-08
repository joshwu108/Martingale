"""The independent checker on replayed rows: the replay block is bound into the sequence digest and the
origin binding (a replayed row's tokens equal the fresh row it claims to copy) is verified across files."""
import json
from fractions import Fraction

import pytest

from campaigns.mutation_tokens import _resign
from checker.verify_tokens import verify_export
from martingale.record import Recorder, SamplerConfig, float_bits

SAMPLER = SamplerConfig(temperature=1.0, top_p=1.0, engine="fake", engine_version="0", logprobs_mode="raw")
W1, W2, TOK, CD = "1" * 64, "2" * 64, "f" * 64, "c" * 64
ORIGIN_FILE, COPY_FILE, EXPLICIT_FILE = "actor0000_seq00000001.json", "actor0001_seq00000000.json", "actor0001_seq00000001.json"


@pytest.fixture
def exported(tmp_path):
    """actor 0: two fresh rows (the second is the origin, last in its chain); actor 1: a replayed copy of the
    origin with scores, then an explicit replayed row without an origin."""
    rec = Recorder(tmp_path / "ws")
    r0 = rec.publish_revision(W1, TOK, SAMPLER, step=0)
    r2 = rec.publish_revision(W2, TOK, SAMPLER, step=2)
    with rec.sequence(0, "step0/row1", [1, 2, 4]) as b:
        b.token(r0.digest, 10, -0.25); b.token(r0.digest, 12, -0.75)
    with rec.sequence(0, "step0/row0", [1, 2, 3]) as a:
        a.token(r0.digest, 10, -0.5); a.token(r0.digest, 11, -1.5, topk=[(11, -1.5), (3, -2.0)])
    rec.score(a.record.digest, r0.digest, [-0.5, -1.5])
    s = rec.replay_from(a.record.digest, actor_id=1, sequence_id="step2/replay0", draw_id="3/0", content_digest=CD,
                        is_weight=Fraction(1, 2), rescale=Fraction(2, 3), reward=0.5)
    rec.score(s.digest, r2.digest, [-0.7, -1.2])
    t = rec.replayed(actor_id=1, sequence_id="restored", prompt_ids=[8, 8], steps=[(r0.digest, 5, -0.3)],
                     draw_id="3/1", content_digest="d" * 64, is_weight=1)
    rec.score(t.digest, r2.digest, [-0.2])
    return rec.export_for_checker(tmp_path / "exp"), rec.ledger.head()


def _edit(path, fn):
    d = json.loads(path.read_text()); fn(d); path.write_text(json.dumps(d))


def test_clean_export_with_replayed_rows_verifies(exported):
    exp, head = exported
    r = verify_export(exp, expected_head=head)
    assert r["ok"], r["errors"]
    assert r["n_sequences"] == 4 and r["n_replayed_sequences"] == 2 and r["n_tokens"] == 7
    d = json.loads((exp / "sequences" / COPY_FILE).read_text())
    assert d["provenance"] == "replayed" and set(d["replay"]) == {
        "draw_id", "content_digest", "is_weight_num", "is_weight_den", "rescale_num", "rescale_den", "origin_digest"}
    assert "provenance" not in json.loads((exp / "sequences" / ORIGIN_FILE).read_text())


@pytest.mark.parametrize("key,val", [
    ("draw_id", "3/9"), ("draw_id", ""), ("draw_id", 3), ("content_digest", "e" * 64), ("content_digest", "C" * 64),
    ("is_weight_num", 3), ("is_weight_num", -1), ("is_weight_num", True), ("is_weight_den", 0), ("is_weight_den", 4),
    ("rescale_num", 4), ("rescale_den", 1), ("origin_digest", "a" * 64), ("origin_digest", None), ("smuggled", 1),
])
def test_single_field_forgeries_in_the_replay_block_are_rejected(exported, key, val):
    exp, head = exported
    _edit(exp / "sequences" / COPY_FILE, lambda d: d["replay"].__setitem__(key, val))
    r = verify_export(exp, expected_head=head)
    assert not r["ok"] and COPY_FILE in r["errors"], r["errors"]
    assert not verify_export(exp)["ok"]                                # the digest alone catches each of these


@pytest.mark.parametrize("mutate", [
    lambda d: d.__setitem__("provenance", "fresh"),                     # says fresh, carries a replay block
    lambda d: d.__setitem__("provenance", "cached"),
    lambda d: d.__setitem__("replay", None),                             # says replayed, no block
    lambda d: d.pop("provenance"),
    lambda d: d.__setitem__("replay", "3/0"),
])
def test_provenance_and_replay_block_must_agree(exported, mutate):
    exp, head = exported
    _edit(exp / "sequences" / COPY_FILE, mutate)
    r = verify_export(exp)
    assert not r["ok"] and COPY_FILE in r["errors"], r["errors"]


@pytest.mark.parametrize("key,val", [("provenance", "fresh"), ("replay", None)])
def test_fresh_row_carrying_the_keys_is_rejected(exported, key, val):
    """One digest, one file form: a fresh row with an explicit provenance or a null block is malleable, so refused."""
    exp, head = exported
    _edit(exp / "sequences" / ORIGIN_FILE, lambda d: d.__setitem__(key, val))
    r = verify_export(exp, expected_head=head)
    assert not r["ok"] and ORIGIN_FILE in r["errors"], r["errors"]


def test_fresh_row_given_a_replay_block_is_rejected(exported):
    exp, head = exported
    block = json.loads((exp / "sequences" / COPY_FILE).read_text())["replay"]

    def fn(d):
        d["provenance"] = "replayed"; d["replay"] = block
    _edit(exp / "sequences" / ORIGIN_FILE, fn)
    assert not verify_export(exp)["ok"]


class TestOriginBinding:
    def test_unreduced_weight_is_rejected_even_when_resigned(self, exported):
        exp, head = exported

        def fn(d):
            d["replay"]["is_weight_num"] = 2; d["replay"]["is_weight_den"] = 4; _resign(d)
        _edit(exp / "sequences" / COPY_FILE, fn)
        r = verify_export(exp)
        assert not r["ok"] and any("reduced" in e for e in r["errors"][COPY_FILE]), r["errors"]

    def test_tampered_origin_is_caught_by_its_replayed_copy_without_an_anchor(self, exported):
        exp, head = exported
        _edit(exp / "sequences" / ORIGIN_FILE,
              lambda d: (d["tokens"][1].__setitem__("logprob_bits", float_bits(-1.5000001)), _resign(d)))
        r = verify_export(exp)
        assert not r["ok"] and ORIGIN_FILE not in r["errors"]           # the origin's own chain is consistent
        assert any("origin" in e for e in r["errors"][COPY_FILE]), r["errors"]

    def test_tampered_replayed_copy_is_caught_against_its_origin(self, exported):
        exp, head = exported
        _edit(exp / "sequences" / COPY_FILE, lambda d: (d["tokens"][0].__setitem__("token_id", 99), _resign(d)))
        r = verify_export(exp)
        assert not r["ok"] and any("origin" in e for e in r["errors"][COPY_FILE])

    @pytest.mark.parametrize("field,val", [("prompt_digest", "b" * 64), ("prompt_len", 2)])
    def test_replayed_prompt_must_match_the_origin(self, exported, field, val):
        exp, head = exported

        def fn(d):
            d[field] = val
            if field == "prompt_digest":
                d["prompt_ids"] = None
            else:
                d["prompt_ids"] = d["prompt_ids"][:2]
            _resign(d)
        _edit(exp / "sequences" / COPY_FILE, fn)
        r = verify_export(exp)
        assert not r["ok"] and COPY_FILE in r["errors"]

    def test_origin_pointing_at_another_fresh_row_is_rejected(self, exported):
        exp, head = exported
        other = json.loads((exp / "sequences" / "actor0000_seq00000000.json").read_text())["digest"]
        _edit(exp / "sequences" / COPY_FILE, lambda d: (d["replay"].__setitem__("origin_digest", other), _resign(d)))
        r = verify_export(exp)
        assert not r["ok"] and any("origin" in e for e in r["errors"][COPY_FILE])

    def test_origin_must_be_in_the_export(self, exported):
        exp, head = exported
        _edit(exp / "sequences" / COPY_FILE, lambda d: (d["replay"].__setitem__("origin_digest", "9" * 64), _resign(d)))
        r = verify_export(exp)
        assert not r["ok"] and any("origin" in e and "not in" in e for e in r["errors"][COPY_FILE]), r["errors"]

    def test_origin_must_be_fresh(self, exported):
        exp, head = exported
        copy = json.loads((exp / "sequences" / COPY_FILE).read_text())["digest"]
        # the explicit row (last of its chain, so re-signing breaks nothing else) now claims the replayed copy as origin
        _edit(exp / "sequences" / EXPLICIT_FILE, lambda d: (d["replay"].__setitem__("origin_digest", copy), _resign(d)))
        r = verify_export(exp)
        assert not r["ok"] and any("fresh" in e for e in r["errors"][EXPLICIT_FILE]), r["errors"]

    def test_an_origin_that_failed_verification_fails_its_copies_too(self, exported):
        exp, head = exported
        _edit(exp / "sequences" / ORIGIN_FILE, lambda d: d["tokens"][0].__setitem__("token_id", 4242))   # not re-signed
        r = verify_export(exp)
        assert ORIGIN_FILE in r["errors"] and COPY_FILE in r["errors"]

    def test_dropping_the_replay_block_and_resigning_needs_the_anchor(self, exported):
        exp, head = exported
        _edit(exp / "sequences" / EXPLICIT_FILE, lambda d: (d.pop("provenance"), d.pop("replay"), _resign(d)))
        assert verify_export(exp)["ok"]                                  # a consistent fresh-looking chain
        assert not verify_export(exp, expected_head=head)["ok"]          # the anchor catches it


def test_cli_reports_replayed_count(exported):
    import subprocess
    import sys
    exp, head = exported
    r = subprocess.run([sys.executable, "-m", "checker.verify_tokens", str(exp), "--expected-head", head],
                       capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout)["n_replayed_sequences"] == 2
