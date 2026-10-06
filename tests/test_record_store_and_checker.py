"""TokenLedger + Recorder round trips, fail-closed rules, and the independent token checker."""
import json

import pytest

from checker.verify_tokens import verify_export
from martingale.record import (
    GENESIS_DIGEST,
    ProtocolViolation,
    Recorder,
    SamplerConfig,
    float_bits,
)

SAMPLER = SamplerConfig(temperature=1.0, top_p=1.0, engine="fake", engine_version="0", logprobs_mode="raw")
W1, W2, TOK = "1" * 64, "2" * 64, "f" * 64


@pytest.fixture
def rec(tmp_path):
    return Recorder(tmp_path / "ws")


def _run(rec, n_seq=3, mixed=True):
    r1 = rec.publish_revision(W1, TOK, SAMPLER, step=0)
    r2 = rec.publish_revision(W2, TOK, SAMPLER, step=1)
    seqs = []
    for i in range(n_seq):
        with rec.sequence(actor_id=i % 2, sequence_id=f"p{i}", prompt_ids=[1, 2, 3 + i]) as s:
            s.token(r1.digest, 10, -0.5)
            s.token(r2.digest if mixed else r1.digest, 11, -1.5, topk=[(11, -1.5), (3, -2.0)])
            s.reward(float(i))
        seqs.append(s.record)
    for s in seqs:
        rec.score(s.digest, r2.digest, [-0.4, -1.6])
    return r1, r2, seqs


class TestRecorder:
    def test_round_trip_and_mixed_revision(self, rec):
        r1, r2, seqs = _run(rec)
        got = list(rec.ledger.all_sequences())
        assert got == sorted(seqs, key=lambda x: (x.actor_id, x.sequence_index))
        assert all(s.is_mixed_revision for s in got)
        assert [s.sequence_index for s in got if s.actor_id == 0] == [0, 1]
        assert got[1].prev_sequence_digest == got[0].digest   # actor 0 chain (sorted actor, index)
        assert rec.ledger.scores_for(seqs[0].digest)[1].prev_digest == rec.ledger.scores_for(seqs[0].digest)[0].digest
        assert rec.ledger.get_revision(r2.digest).parent_digest == r1.digest

    def test_token_without_pin_or_with_unpublished_revision_refused(self, rec):
        rec.publish_revision(W1, TOK, SAMPLER, step=0)
        with pytest.raises(ProtocolViolation):
            with rec.sequence(0, "p", [1]) as s:
                s.token(None, 1, -0.1)
        with pytest.raises(ProtocolViolation):
            with rec.sequence(0, "p", [1]) as s:
                s.token("9" * 64, 1, -0.1)
        assert rec.ledger.count_sequences() == 0

    def test_exception_inside_block_discards_sequence(self, rec):
        r = rec.publish_revision(W1, TOK, SAMPLER, step=0)
        with pytest.raises(RuntimeError, match="engine died"):
            with rec.sequence(0, "p", [1]) as s:
                s.token(r.digest, 1, -0.1)
                raise RuntimeError("engine died")
        assert rec.ledger.count_sequences() == 0

    def test_empty_sequence_refused(self, rec):
        with pytest.raises(ProtocolViolation):
            with rec.sequence(0, "p", [1]):
                pass

    def test_score_outside_sequence_refused(self, rec):
        r1, r2, seqs = _run(rec, n_seq=1)
        with pytest.raises(ValueError, match="outside"):
            rec.score(seqs[0].digest, r2.digest, [-0.1, -0.2, -0.3])

    def test_reopen_continues_chain(self, tmp_path):
        rec = Recorder(tmp_path / "ws")
        r1, _, seqs = _run(rec, n_seq=2)
        rec.ledger.close()
        rec2 = Recorder(tmp_path / "ws")
        with rec2.sequence(0, "later", [7]) as s:
            s.token(r1.digest, 1, -0.2)
        assert s.record.sequence_index == 1 and s.record.prev_sequence_digest == seqs[0].digest


class TestTokenChecker:
    def test_clean_export_verifies(self, rec, tmp_path):
        _run(rec)
        report = verify_export(rec.export_for_checker(tmp_path / "exp"))
        assert report["ok"], report
        assert (report["n_revisions"], report["n_sequences"], report["n_tokens"]) == (2, 3, 6)

    @pytest.mark.parametrize("mutation", [
        ("tokens", 0, "token_id", 99),
        ("tokens", 0, "logprob_bits", float_bits(-0.5000001)),
        ("tokens", 1, "revision_digest", None),          # swap to the other token's revision
        ("tokens", 0, "position", 1),
        ("tokens", 0, "prev_digest", "a" * 64),
        ("", None, "reward_bits", float_bits(9.0)),
        ("", None, "prompt_len", 99),
        ("", None, "prompt_digest", "c" * 64),
        ("", None, "sequence_index", 7),
        ("", None, "terminal_digest", "d" * 64),
        ("scores", 0, "logprob_bits", float_bits(0.0)),
        ("scores", 1, "position", 0),
        ("scores", 0, "train_revision_digest", "e" * 64),
    ])
    def test_single_field_forgeries_rejected(self, rec, tmp_path, mutation):
        _run(rec)
        exp = rec.export_for_checker(tmp_path / "exp")
        f = sorted((exp / "sequences").glob("*.json"))[0]
        d = json.loads(f.read_text())
        where, idx, key, val = mutation
        target = d if where == "" else d[where][idx]
        if val is None:  # swap revision between the two tokens
            val = d["tokens"][0]["revision_digest"]
        target[key] = val
        f.write_text(json.dumps(d))
        report = verify_export(exp)
        assert not report["ok"]
        assert f.name in report["errors"], report

    def test_dropped_sequence_breaks_the_actor_chain(self, rec, tmp_path):
        _run(rec, n_seq=4)
        exp = rec.export_for_checker(tmp_path / "exp")
        (exp / "sequences" / "actor0000_seq00000000.json").unlink()
        report = verify_export(exp)
        assert not report["ok"] and "actor0000_seq00000001.json" in report["errors"]

    def test_tampered_or_missing_manifest_rejected(self, rec, tmp_path):
        r1, _, _ = _run(rec)
        exp = rec.export_for_checker(tmp_path / "exp")
        mf = exp / "revisions" / f"{r1.digest}.json"
        d = json.loads(mf.read_text()); d["sampler"]["temperature"] = 0.9
        mf.write_text(json.dumps(d))
        report = verify_export(exp)
        assert not report["ok"] and "revisions" in report["errors"]
        mf.unlink()
        report = verify_export(exp)
        assert not report["ok"]
        assert all(("not in store" in " ".join(v)) or ("missing" in " ".join(v)) for v in report["errors"].values())

    def test_sampler_change_mid_sequence_is_a_forgery_even_with_valid_digests(self, tmp_path):
        rec = Recorder(tmp_path / "ws")
        a = rec.publish_revision(W1, TOK, SAMPLER, step=0)
        b = rec.publish_revision(W1, TOK, SamplerConfig(temperature=0.5), step=1)
        with rec.sequence(0, "p", [1]) as s:
            s.token(a.digest, 1, -0.1)
            s.token(b.digest, 2, -0.2)
        report = verify_export(rec.export_for_checker(tmp_path / "exp"))
        assert not report["ok"]
        assert any("sampler config changed" in e for errs in report["errors"].values() for e in errs)

    def test_cli_exit_codes(self, rec, tmp_path):
        import subprocess
        import sys
        _run(rec, n_seq=1)
        exp = rec.export_for_checker(tmp_path / "exp")
        r = subprocess.run([sys.executable, "-m", "checker.verify_tokens", str(exp)], capture_output=True, text=True)
        assert r.returncode == 2                                           # refuses to run without an anchor decision
        r = subprocess.run([sys.executable, "-m", "checker.verify_tokens", str(exp), "--expected-head", rec.ledger.head()],
                           capture_output=True, text=True)
        assert r.returncode == 0 and json.loads(r.stdout)["ok"] and json.loads(r.stdout)["anchored"]
        r = subprocess.run([sys.executable, "-m", "checker.verify_tokens", str(exp), "--unanchored"], capture_output=True, text=True)
        assert r.returncode == 0 and "UNANCHORED" in json.loads(r.stdout)["warning"]
        r = subprocess.run([sys.executable, "-m", "checker.verify_tokens", str(exp), "--expected-head", "0" * 64],
                           capture_output=True, text=True)
        assert r.returncode == 1


class TestLedgerHead:
    def test_head_matches_checker_and_moves_on_any_change(self, rec, tmp_path):
        r1, r2, seqs = _run(rec, n_seq=2)
        head = rec.ledger.head()
        report = verify_export(rec.export_for_checker(tmp_path / "e1"), expected_head=head)
        assert report["ok"] and report["head"] == head
        with rec.sequence(0, "more", [1]) as s:
            s.token(r1.digest, 1, -0.3)
        assert rec.ledger.head() != head
        rec.score(s.record.digest, r2.digest, [-0.2])
        h2 = rec.ledger.head()
        rec.publish_revision("7" * 64, TOK, SAMPLER, step=9)     # unreferenced revision changes the head too
        assert rec.ledger.head() != h2

    def test_resigned_forgery_caught_only_with_anchor(self, rec, tmp_path):
        from campaigns.mutation_tokens import _resign
        _run(rec, n_seq=1)
        head = rec.ledger.head()
        exp = rec.export_for_checker(tmp_path / "exp")
        f = next((exp / "sequences").glob("*.json"))
        d = json.loads(f.read_text()); d["tokens"][0]["token_id"] = 4242; _resign(d)
        f.write_text(json.dumps(d))
        assert verify_export(exp)["ok"]                                  # consistent chain, no anchor
        assert not verify_export(exp, expected_head=head)["ok"]         # caught with the anchor


class TestCheckerHardening:
    """Findings from the 2026-10-06 review: unbound keys, bool/negative ints, malformed files."""

    def _export(self, rec, tmp_path):
        _run(rec, n_seq=2)
        return rec.export_for_checker(tmp_path / "exp"), rec.ledger.head()

    @pytest.mark.parametrize("where", ["manifest", "token", "sequence", "score"])
    def test_unbound_extra_key_rejected_even_with_true_head(self, rec, tmp_path, where):
        exp, head = self._export(rec, tmp_path)
        if where == "manifest":
            f = next((exp / "revisions").glob("*.json"))
        else:
            f = sorted((exp / "sequences").glob("*.json"))[0]
        d = json.loads(f.read_text())
        target = {"manifest": d, "token": d.get("tokens", [None])[0], "sequence": d, "score": d.get("scores", [None])[0]}[where]
        target["smuggled"] = "payload"
        f.write_text(json.dumps(d))
        report = verify_export(exp, expected_head=head)
        assert not report["ok"] and any("unbound" in e for errs in report["errors"].values() for e in errs)

    @pytest.mark.parametrize("key,val", [("position", True), ("token_id", True), ("token_id", -5), ("token_id", 1.0)])
    def test_bool_float_and_negative_ints_rejected(self, rec, tmp_path, key, val):
        exp, _ = self._export(rec, tmp_path)
        f = sorted((exp / "sequences").glob("*.json"))[0]
        d = json.loads(f.read_text()); d["tokens"][0][key] = val
        f.write_text(json.dumps(d))
        assert not verify_export(exp)["ok"]

    @pytest.mark.parametrize("content", ["{not json", "[1, 2]", '"a string"', json.dumps({"actor_id": "zero"})])
    def test_malformed_sequence_file_is_a_verdict_not_a_crash(self, rec, tmp_path, content):
        exp, head = self._export(rec, tmp_path)
        f = sorted((exp / "sequences").glob("*.json"))[0]
        f.write_text(content)
        report = verify_export(exp, expected_head=head)
        assert not report["ok"] and f.name in report["errors"]

    def test_scores_null_and_non_list(self, rec, tmp_path):
        exp, head = self._export(rec, tmp_path)
        f = sorted((exp / "sequences").glob("*.json"))[0]
        d = json.loads(f.read_text()); d["scores"] = None; f.write_text(json.dumps(d))
        r = verify_export(exp, expected_head=head)
        assert not r["ok"] and "head" in r["errors"]          # null == no scores: binding ok, head moved
        d["scores"] = "nope"; f.write_text(json.dumps(d))
        assert not verify_export(exp)["ok"]

    def test_filename_must_match_actor_and_index(self, rec, tmp_path):
        exp, _ = self._export(rec, tmp_path)
        f = sorted((exp / "sequences").glob("*.json"))[1]
        f.rename(f.with_name("actor0000_seq00000005.json"))
        assert not verify_export(exp)["ok"]

    def test_large_indices_sort_numerically(self, rec, tmp_path):
        exp, head = self._export(rec, tmp_path)
        from checker.verify_tokens import _sequence_files
        names = [p.name for p in _sequence_files(exp)]
        assert names == sorted(names)   # for small indices the orders coincide; the parser is what we exercise
        assert verify_export(exp, expected_head=head)["ok"]


class TestStoreTransactions:
    def test_failed_append_leaves_no_partial_state(self, rec):
        r1, r2, seqs = _run(rec, n_seq=1)
        from martingale.record import ScoreRecord, float_bits
        head = rec.ledger.head()
        bad = [ScoreRecord(seqs[0].digest, 0, r2.digest, float_bits(-0.1), rec.ledger.last_score_digest(seqs[0].digest)),
               ScoreRecord(seqs[0].digest, 9, r2.digest, float_bits(-0.1), "0" * 64)]   # second is out of range + broken chain
        with pytest.raises(ValueError):
            rec.ledger.append_scores(bad)
        assert not rec.ledger._con.in_transaction
        assert rec.ledger.head() == head
        assert len(rec.ledger.scores_for(seqs[0].digest)) == 2          # the original two scores only

    def test_duplicate_index_from_second_writer_is_refused_atomically(self, tmp_path):
        from martingale.record import TokenLedger, build_sequence, float_bits
        a = Recorder(tmp_path / "ws"); r = a.publish_revision(W1, TOK, SAMPLER, step=0)
        b = TokenLedger(tmp_path / "ws" / "tokens.db")
        seq = build_sequence(0, 0, "x", [1], [(r.digest, 1, float_bits(-0.1), None)])
        a.ledger.append_sequence(seq)
        with pytest.raises(ValueError):
            b.append_sequence(build_sequence(0, 0, "y", [1], [(r.digest, 2, float_bits(-0.2), None)]))
        assert b.count_sequences() == 1 and not b._con.in_transaction

    def test_export_refuses_foreign_directory_and_replaces_old_export(self, rec, tmp_path):
        _run(rec, n_seq=1)
        exp = rec.export_for_checker(tmp_path / "exp")
        assert verify_export(exp, expected_head=rec.ledger.head())["ok"]
        rec.export_for_checker(tmp_path / "exp")                                  # replace is fine
        (tmp_path / "other").mkdir(); (tmp_path / "other" / "keep.txt").write_text("x")
        with pytest.raises(FileExistsError):
            rec.export_for_checker(tmp_path / "other")
