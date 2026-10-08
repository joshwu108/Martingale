"""
campaigns/mutation_tokens.py — forgery campaign against checker/verify_tokens.py.

Builds a small two-actor token record (three revisions, mixed-revision
sequences, top-k entries, trainer scores, two replayed rows: a copy of a
fresh row and an explicit row without an origin), exports it, then applies
>= 60 single-fault mutations to copies of the export and runs the
independent checker on each. Two verdict columns:

  rejected_anchored    checker given the true ledger head (the deployment mode:
                       TokenLedger.head() is published out of band, e.g. in the run log)
  rejected_unanchored  checker given only the files

The re-signed class (tamper, then recompute every downstream digest) is the
reason anchors exist: without a trusted head a consistent chain is a
consistent chain. One exception the campaign shows: a replayed copy binds
its origin's digest, so re-signing anything inside the origin (tokens,
reward, header; the last fresh row of actor 1 here) or deleting it is caught
without an anchor; the origin's scores hang off its digest and are not
witnessed. The kill rule is 100% rejection in the anchored column; the
unanchored column is reported, not asserted.

    uv run python -m campaigns.mutation_tokens
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Callable

from checker.verify_tokens import verify_export
from martingale.record import Recorder, SamplerConfig, float_bits
from martingale.record.bits import digest_json
from martingale.record.replay import REPLAY_KEYS

RESULTS_DIR = Path(__file__).parent.parent / "results"
W = ["1" * 64, "2" * 64, "3" * 64]
TOK = "f" * 64
SAMPLER = SamplerConfig(temperature=1.0, top_p=1.0, engine="fake", engine_version="0", logprobs_mode="raw")


def build_fixture(ws: Path) -> tuple[Recorder, str]:
    rec = Recorder(ws)
    r0 = rec.publish_revision(W[0], TOK, SAMPLER, step=0)
    r1 = rec.publish_revision(W[1], TOK, SAMPLER, step=1)
    rec.publish_revision(W[2], TOK, SamplerConfig(temperature=0.5), step=2)   # different sampler
    seqs = []
    for i in range(6):
        with rec.sequence(actor_id=i % 2, sequence_id=f"p{i}", prompt_ids=[1, 2, 3 + i]) as s:
            s.token(r0.digest, 10 + i, -0.5 - i / 10)
            s.token(r1.digest if i % 3 else r0.digest, 20 + i, -1.5, topk=[(20 + i, -1.5), (7, -2.5)])
            s.token(r1.digest, 30 + i, -0.25)
            s.reward(float(i % 2))
        seqs.append(s.record)
    for s in seqs:
        rec.score(s.digest, r1.digest, [-0.45, -1.6, -0.3])
    # replayed rows (record/replay.py): a copy of actor 1's last fresh row, and an explicit row without an origin
    copied = rec.replay_from(seqs[5].digest, actor_id=0, sequence_id="step1/replay0", draw_id="4/0",
                             content_digest="c" * 64, is_weight=Fraction(1, 2), rescale=Fraction(3, 4), reward=1.0)
    rec.score(copied.digest, r1.digest, [-0.45, -1.6, -0.3])
    explicit = rec.replayed(actor_id=0, sequence_id="restored", prompt_ids=[4, 4, 4],
                            steps=[(r0.digest, 40, -0.5), (r1.digest, 41, -1.5, [(41, -1.5), (7, -2.5)]), (r1.digest, 42, -0.25)],
                            draw_id="4/1", content_digest="d" * 64, is_weight=1)
    rec.score(explicit.digest, r1.digest, [-0.4, -1.4, -0.2])
    return rec, rec.ledger.head()


COPY, EXPLICIT = 3, 4     # positions in _seq_files(): actor 0 holds seq 0..4 (3 = the copy, 4 = the explicit row)


Mutation = Callable[[Path], None]


def _seq_files(exp: Path) -> list[Path]:
    return sorted((exp / "sequences").glob("*.json"))


def _rev_files(exp: Path) -> list[Path]:
    return sorted((exp / "revisions").glob("*.json"))


def _edit(path: Path, fn: Callable[[dict], None]) -> None:
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d))


def _flip_nibble(h: str, i: int = 5) -> str:
    c = h[i]
    return h[:i] + ("0" if c != "0" else "1") + h[i + 1:]


def _resign(d: dict) -> None:
    """Recompute every digest in a sequence file after tampering (the re-signed forgery)."""
    prev = d["prev_sequence_digest"]
    for t in d["tokens"]:
        t["prev_digest"] = prev
        t["digest"] = digest_json({k: t[k] for k in ("revision_digest", "position", "token_id", "logprob_bits", "topk", "prev_digest")})
        prev = t["digest"]
    d["token_digests"] = [t["digest"] for t in d["tokens"]]
    d["terminal_digest"] = prev
    canon = {k: d[k] for k in ("actor_id", "sequence_index", "sequence_id", "prompt_digest", "prompt_len",
                               "token_digests", "reward_bits", "prev_sequence_digest", "terminal_digest")}
    if d.get("replay") is not None:                      # a replayed row digests its provenance and replay block
        canon["provenance"] = "replayed"
        canon["replay"] = {k: d["replay"][k] for k in REPLAY_KEYS}
    d["digest"] = digest_json(canon)
    sprev = d["digest"]
    for s in d.get("scores", []):
        s["sequence_digest"] = d["digest"]
        s["prev_digest"] = sprev
        s["digest"] = digest_json({k: s[k] for k in ("sequence_digest", "position", "train_revision_digest", "logprob_bits", "prev_digest")})
        sprev = s["digest"]


def mutations() -> dict[str, Mutation]:
    m: dict[str, Mutation] = {}

    def seq_field(name, where, idx, key, val):
        def fn(exp):
            def ed(d):
                target = d if where is None else d[where][idx]
                target[key] = val(target[key]) if callable(val) else val
            _edit(_seq_files(exp)[1], ed)
        m[name] = fn

    # token fields (first, middle, last token of one sequence)
    for pos in (0, 1, 2):
        seq_field(f"token{pos}_id+1", "tokens", pos, "token_id", lambda v: v + 1)
        seq_field(f"token{pos}_logprob_ulp", "tokens", pos, "logprob_bits", lambda b: b[:-1] + ("0" if b[-1] != "0" else "1"))
        seq_field(f"token{pos}_revision_nibble", "tokens", pos, "revision_digest", _flip_nibble)
        seq_field(f"token{pos}_prev_nibble", "tokens", pos, "prev_digest", _flip_nibble)
        seq_field(f"token{pos}_digest_nibble", "tokens", pos, "digest", _flip_nibble)
        seq_field(f"token{pos}_position+1", "tokens", pos, "position", lambda v: v + 1)
    seq_field("token1_topk_id", "tokens", 1, "topk", lambda tk: [[tk[0][0] + 1, tk[0][1]], tk[1]])
    seq_field("token1_topk_bits", "tokens", 1, "topk", lambda tk: [tk[0], [tk[1][0], float_bits(-2.4)]])
    seq_field("token1_topk_dropped", "tokens", 1, "topk", None)
    seq_field("token0_topk_added", "tokens", 0, "topk", [[1, float_bits(-0.1)]])
    seq_field("token1_bits_nan", "tokens", 1, "logprob_bits", "f32:7fc00000")
    seq_field("token1_bits_f64_same_value", "tokens", 1, "logprob_bits", float_bits(-1.5, "f64"))
    # revision swap to a different revision with the SAME sampler (the subtle class)
    def swap_same_sampler(exp):
        def ed(d):
            d["tokens"][0]["revision_digest"] = d["tokens"][2]["revision_digest"]
        _edit(_seq_files(exp)[1], ed)
    m["token0_revision_swapped_same_sampler"] = swap_same_sampler
    # sequence header fields
    for key, val in [("reward_bits", float_bits(5.0)), ("reward_bits", None), ("prompt_len", 99),
                     ("prompt_digest", _flip_nibble), ("sequence_index", 9), ("actor_id", lambda a: 1 - a),
                     ("sequence_id", "forged"), ("prev_sequence_digest", _flip_nibble), ("terminal_digest", _flip_nibble),
                     ("digest", _flip_nibble)]:
        seq_field(f"seq_{key}_{'fn' if callable(val) else val}", None, None, key, val)
    seq_field("seq_token_digests_reordered", None, None, "token_digests", lambda l: list(reversed(l)))
    # structural
    def drop_token(exp):
        _edit(_seq_files(exp)[1], lambda d: d["tokens"].pop(1))
    def dup_token(exp):
        _edit(_seq_files(exp)[1], lambda d: d["tokens"].append(copy.deepcopy(d["tokens"][-1])))
    def swap_tokens(exp):
        def ed(d): d["tokens"][0], d["tokens"][1] = d["tokens"][1], d["tokens"][0]
        _edit(_seq_files(exp)[1], ed)
    def truncate_tokens(exp):
        _edit(_seq_files(exp)[1], lambda d: d.__setitem__("tokens", d["tokens"][:2]))
    def drop_sequence(exp):
        _seq_files(exp)[0].unlink()
    def drop_last_sequence(exp):
        _seq_files(exp)[-1].unlink()
    def swap_sequence_files(exp):
        a, b = _seq_files(exp)[0], _seq_files(exp)[2]
        ta, tb = a.read_text(), b.read_text(); a.write_text(tb); b.write_text(ta)
    def replay_sequence(exp):
        f = _seq_files(exp)[0]
        shutil.copy(f, f.with_name("actor0000_seq00000009.json"))
    m.update(drop_token=drop_token, dup_token=dup_token, swap_tokens=swap_tokens, truncate_tokens=truncate_tokens,
             drop_first_sequence=drop_sequence, drop_last_sequence=drop_last_sequence,
             swap_sequence_files=swap_sequence_files, replay_sequence_as_new=replay_sequence)
    # scores
    for key, val in [("logprob_bits", float_bits(0.0)), ("position", 7), ("train_revision_digest", _flip_nibble),
                     ("prev_digest", _flip_nibble), ("digest", _flip_nibble), ("sequence_digest", _flip_nibble)]:
        seq_field(f"score0_{key}", "scores", 0, key, val)
    seq_field("score2_position_dup", "scores", 2, "position", 1)
    def drop_score(exp): _edit(_seq_files(exp)[1], lambda d: d["scores"].pop(0))
    def swap_scores(exp):
        def ed(d): d["scores"][0], d["scores"][1] = d["scores"][1], d["scores"][0]
        _edit(_seq_files(exp)[1], ed)
    def move_scores(exp):
        f0, f1 = _seq_files(exp)[0], _seq_files(exp)[1]
        s = json.loads(f1.read_text())["scores"]
        _edit(f0, lambda d: d.__setitem__("scores", s))
    m.update(drop_score=drop_score, swap_scores=swap_scores, move_scores_to_other_sequence=move_scores)
    # manifests
    for key, val in [("step", lambda v: v + 1), ("weights_digest", _flip_nibble), ("tokenizer_digest", _flip_nibble),
                     ("parent_digest", _flip_nibble), ("digest", _flip_nibble), ("kind", "rational")]:
        def mf(exp, key=key, val=val):
            _edit(_rev_files(exp)[0], lambda d: d.__setitem__(key, val(d[key]) if callable(val) else val))
        m[f"manifest_{key}"] = mf
    for skey in ("temperature", "top_k", "logprobs_mode", "engine_version"):
        def ms(exp, skey=skey):
            _edit(_rev_files(exp)[0], lambda d: d["sampler"].__setitem__(skey, 0.9 if skey == "temperature" else "x"))
        m[f"manifest_sampler_{skey}"] = ms
    def drop_manifest(exp):
        """Drop the manifest of a revision tokens reference."""
        used = json.loads(_seq_files(exp)[0].read_text())["tokens"][0]["revision_digest"]
        (exp / "revisions" / f"{used}.json").unlink()
    def drop_unreferenced_manifest(exp):
        """Drop a revision no token references: only the anchored revision set catches it."""
        used = {t["revision_digest"] for f in _seq_files(exp) for t in json.loads(f.read_text())["tokens"]}
        used |= {s["train_revision_digest"] for f in _seq_files(exp) for s in json.loads(f.read_text())["scores"]}
        for f in _rev_files(exp):
            if f.stem not in used:
                f.unlink(); return
        raise RuntimeError("fixture has no unreferenced revision")
    def rename_manifest(exp):
        f = _rev_files(exp)[0]; f.rename(f.with_name("0" * 64 + ".json"))
    m.update(drop_manifest=drop_manifest, drop_unreferenced_manifest=drop_unreferenced_manifest,
             rename_manifest=rename_manifest)
    # unbound content, type confusion, malformed files (2026-10-06 review)
    seq_field("seq_extra_key", None, None, "smuggled", "payload")
    seq_field("token0_extra_key", "tokens", 0, "note", "x")
    seq_field("score0_extra_key", "scores", 0, "weight", 1.0)
    def manifest_extra_key(exp):
        _edit(_rev_files(exp)[0], lambda d: d.__setitem__("comment", "x"))
    m["manifest_extra_key"] = manifest_extra_key
    seq_field("token0_position_bool", "tokens", 0, "position", False)
    seq_field("token1_token_id_bool", "tokens", 1, "token_id", True)
    seq_field("token1_token_id_negative", "tokens", 1, "token_id", -1)
    seq_field("token1_token_id_float", "tokens", 1, "token_id", 21.0)
    seq_field("seq_actor_id_string", None, None, "actor_id", "0")
    seq_field("seq_scores_null", None, None, "scores", None)
    seq_field("seq_scores_string", None, None, "scores", "")
    seq_field("score1_position_bool", "scores", 1, "position", True)
    def manifest_step_bool(exp):
        _edit(_rev_files(exp)[0], lambda d: d.__setitem__("step", True))
    m["manifest_step_bool"] = manifest_step_bool
    def malformed_json(exp): _seq_files(exp)[1].write_text("{not json")
    def non_object_file(exp): _seq_files(exp)[1].write_text("[1,2,3]")
    def manifest_malformed(exp): _rev_files(exp)[0].write_text("null")
    def rename_sequence_file(exp):
        f = _seq_files(exp)[1]; f.rename(f.with_name("actor0000_seq00000007.json"))
    m.update(malformed_json=malformed_json, non_object_file=non_object_file, manifest_malformed=manifest_malformed,
             rename_sequence_file=rename_sequence_file)
    # re-signed forgeries: tamper then recompute all digests downstream
    def resigned(key, fn):
        def mut(exp):
            def ed(d): fn(d); _resign(d)
            _edit(_seq_files(exp)[-1], ed)
        m[f"resigned_{key}"] = mut
    resigned("token_id", lambda d: d["tokens"][0].__setitem__("token_id", 999))
    resigned("logprob", lambda d: d["tokens"][1].__setitem__("logprob_bits", float_bits(-0.01)))
    resigned("reward", lambda d: d.__setitem__("reward_bits", float_bits(2.0)))
    resigned("revision_same_sampler", lambda d: d["tokens"][0].__setitem__("revision_digest", d["tokens"][2]["revision_digest"]))
    resigned("drop_token", lambda d: (d["tokens"].pop(), [t.__setitem__("position", i) for i, t in enumerate(d["tokens"])]))
    resigned("score", lambda d: d["scores"][0].__setitem__("logprob_bits", float_bits(-9.0)))
    # replayed rows: the replay block, its provenance flag, and the origin binding
    def rep_field(name, key, val, resign=False, target=COPY):
        def fn(exp):
            def ed(d):
                d["replay"][key] = val(d["replay"][key]) if callable(val) else val
                if resign:
                    _resign(d)
            _edit(_seq_files(exp)[target], ed)
        m[name] = fn
    rep_field("replay_draw_id", "draw_id", "forged")
    rep_field("replay_draw_id_empty", "draw_id", "")
    rep_field("replay_content_digest_nibble", "content_digest", _flip_nibble)
    rep_field("replay_is_weight_num+1", "is_weight_num", lambda v: v + 1)
    rep_field("replay_is_weight_den+1", "is_weight_den", lambda v: v + 1)
    rep_field("replay_rescale_num+1", "rescale_num", lambda v: v + 1)
    rep_field("replay_origin_nibble", "origin_digest", _flip_nibble)
    rep_field("replay_extra_key", "note", "x")
    rep_field("replay_is_weight_bool", "is_weight_num", True)
    rep_field("replay_is_weight_negative", "is_weight_num", -1)
    rep_field("replay_is_weight_den_zero", "is_weight_den", 0)
    rep_field("replay_rescale_zero", "rescale_num", 0)
    def prov(name, fn, target=COPY):
        m[name] = lambda exp: _edit(_seq_files(exp)[target], fn)
    prov("replay_provenance_fresh_with_block", lambda d: d.__setitem__("provenance", "fresh"))
    prov("replay_provenance_bogus", lambda d: d.__setitem__("provenance", "cached"))
    prov("replay_provenance_missing", lambda d: d.pop("provenance"))
    prov("replay_block_null_with_provenance", lambda d: d.__setitem__("replay", None))
    prov("replay_block_string", lambda d: d.__setitem__("replay", "4/0"))
    # re-signed: the block and the digests agree, so only the origin binding or the anchor can catch it
    rep_field("resigned_replay_weight_unreduced", "is_weight_num", lambda v: v * 2, resign=True)       # 1/2 -> 2/2
    rep_field("resigned_replay_origin_missing", "origin_digest", _flip_nibble, resign=True)
    def resigned_origin_other_fresh(exp):
        other = json.loads(_seq_files(exp)[0].read_text())["digest"]
        _edit(_seq_files(exp)[COPY], lambda d: (d["replay"].__setitem__("origin_digest", other), _resign(d)))
    def resigned_copy_token_tampered(exp):
        """The copy no longer equals its origin: caught without an anchor."""
        _edit(_seq_files(exp)[COPY], lambda d: (d["tokens"][1].__setitem__("logprob_bits", float_bits(-1.4)), _resign(d)))
    def resigned_copy_prompt_tampered(exp):
        def ed(d):
            d["prompt_ids"] = None; d["prompt_digest"] = _flip_nibble(d["prompt_digest"]); _resign(d)
        _edit(_seq_files(exp)[COPY], ed)
    def resigned_copy_origin_dropped(exp):
        """The copy keeps its tokens but no longer names its origin: a weaker claim; only the anchor catches it."""
        def ed(d):
            d["replay"]["origin_digest"] = None; _resign(d)
        f3, f4 = _seq_files(exp)[COPY], _seq_files(exp)[EXPLICIT]
        _edit(f3, ed)
        new_prev = json.loads(f3.read_text())["digest"]
        _edit(f4, lambda d: (d.__setitem__("prev_sequence_digest", new_prev), _resign(d)))   # keep actor 0's chain consistent
    def resigned_replay_block_removed(exp):
        """The explicit row pretends to be fresh: a consistent chain; the anchor catches it."""
        _edit(_seq_files(exp)[EXPLICIT], lambda d: (d.pop("provenance"), d.pop("replay"), _resign(d)))
    def resigned_explicit_row_given_origin(exp):
        """The explicit row claims a fresh origin whose tokens differ: caught without an anchor."""
        origin = json.loads(_seq_files(exp)[0].read_text())["digest"]
        _edit(_seq_files(exp)[EXPLICIT], lambda d: (d["replay"].__setitem__("origin_digest", origin), _resign(d)))
    def resigned_fresh_row_given_replay_block(exp):
        """Actor 1's last fresh row (the copy's origin) claims to be a replay of another fresh row."""
        origin = json.loads(_seq_files(exp)[1].read_text())["digest"]
        block = {"draw_id": "x", "content_digest": "e" * 64, "is_weight_num": 1, "is_weight_den": 1,
                 "rescale_num": 1, "rescale_den": 1, "origin_digest": origin}
        _edit(_seq_files(exp)[-1], lambda d: (d.__setitem__("provenance", "replayed"), d.__setitem__("replay", block), _resign(d)))
    def resigned_origin_token_tampered(exp):
        """The origin (last fresh row of actor 1) is re-signed with a different token: its copy witnesses it."""
        _edit(_seq_files(exp)[-1], lambda d: (d["tokens"][2].__setitem__("token_id", 777), _resign(d)))
    def resigned_origin_reward_tampered(exp):
        """The origin's reward is not witnessed by the copy: re-signed, only the anchor catches it."""
        _edit(_seq_files(exp)[-1], lambda d: (d.__setitem__("reward_bits", float_bits(3.0)), _resign(d)))
    seq_field("seq_fresh_provenance_explicit", None, None, "provenance", "fresh")   # fresh rows carry no such key
    seq_field("seq_fresh_replay_null", None, None, "replay", None)
    m.update(resigned_replay_origin_other_fresh=resigned_origin_other_fresh,
             resigned_copy_token_tampered=resigned_copy_token_tampered,
             resigned_copy_prompt_tampered=resigned_copy_prompt_tampered,
             resigned_copy_origin_dropped=resigned_copy_origin_dropped,
             resigned_replay_block_removed=resigned_replay_block_removed,
             resigned_explicit_row_given_origin=resigned_explicit_row_given_origin,
             resigned_fresh_row_given_replay_block=resigned_fresh_row_given_replay_block,
             resigned_origin_token_tampered=resigned_origin_token_tampered,
             resigned_origin_reward_tampered=resigned_origin_reward_tampered)
    return m


def run(verbose: bool = True) -> dict:
    with tempfile.TemporaryDirectory(prefix="martingale_mut_tokens_") as tmp:
        base = Path(tmp)
        rec, head = build_fixture(base / "ws")
        export = rec.export_for_checker(base / "export")
        clean = verify_export(export, expected_head=head)
        assert clean["head"] == head
        assert clean["ok"], clean["errors"]
        muts = mutations()
        rows = []
        for name, fn in muts.items():
            work = base / "mut" / name
            shutil.copytree(export, work)
            fn(work)
            anchored = verify_export(work, expected_head=head)        # must return a verdict, never raise
            unanchored = verify_export(work)
            rows.append({"mutant": name, "rejected_anchored": not anchored["ok"],
                         "rejected_unanchored": not unanchored["ok"],
                         "first_error": next(iter(anchored["errors"].values()), ["(accepted)"])[0]})
            if verbose:
                print(f"{name:40s} anchored={'REJECT' if not anchored['ok'] else 'accept':6s} "
                      f"unanchored={'REJECT' if not unanchored['ok'] else 'accept'}")
    n = len(rows)
    n_anch = sum(r["rejected_anchored"] for r in rows)
    n_unanch = sum(r["rejected_unanchored"] for r in rows)
    accepted_unanchored = [r["mutant"] for r in rows if not r["rejected_unanchored"]]
    return {"campaign": "mutation_tokens", "n_mutants": n, "rejected_anchored": n_anch,
            "rejected_unanchored": n_unanch, "accepted_unanchored": accepted_unanchored,
            "all_rejected_anchored": n_anch == n, "mutants": rows}


def main() -> None:
    out = run()
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "mutation_tokens_report.json").write_text(json.dumps(out, indent=1))
    print(f"\n{out['rejected_anchored']}/{out['n_mutants']} rejected with anchored heads, "
          f"{out['rejected_unanchored']}/{out['n_mutants']} without; "
          f"accepted only without anchors: {out['accepted_unanchored']}")
    if not out["all_rejected_anchored"]:
        print("SURVIVING MUTANTS (anchored):", [r["mutant"] for r in out["mutants"] if not r["rejected_anchored"]])
        sys.exit(1)


if __name__ == "__main__":
    main()
