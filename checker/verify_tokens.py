"""
checker/verify_tokens.py — independent tier-1 verifier for the token record.

Imports nothing from martingale (CI-enforced). Re-implements the bit-string
format, canonical JSON and every digest from the design (docs/design.md 7),
then checks an export produced by TokenLedger.export_for_checker():

  revisions/<digest>.json          LLM revision manifests
  sequences/actorNNNN_seqNNNNNNNN.json   sequence + tokens + scores

Tier 1 verifies BINDING: digests recompute, chains are unbroken, every
referenced revision exists, positions are contiguous, bits are finite,
and every token in a sequence was sampled under the same tokenizer and
sampler config. It cannot and does not verify that a token was actually
drawn from the recorded distribution (docs/nonclaims.md). Written to reject.

    python -m checker.verify_tokens <export_dir> --expected-head <ledger head>
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

GENESIS = "0" * 64
_FORMATS = {"f32": (">f", 8), "f64": (">d", 16)}
SAMPLER_KEYS = (
    "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "max_tokens",
    "logprobs_mode", "dtype", "engine", "engine_version", "tensor_parallel", "seed_policy",
)


MANIFEST_KEYS = frozenset({"kind", "weights_digest", "tokenizer_digest", "sampler", "parent_digest", "step",
                           "digest", "weights_uri"})
TOKEN_KEYS = frozenset({"revision_digest", "position", "token_id", "logprob_bits", "topk", "prev_digest", "digest"})
SEQUENCE_KEYS = frozenset({"actor_id", "sequence_index", "sequence_id", "prompt_digest", "prompt_len",
                           "token_digests", "reward_bits", "prev_sequence_digest", "terminal_digest",
                           "digest", "tokens", "engine_request_id", "scores"})
SCORE_KEYS = frozenset({"sequence_digest", "position", "train_revision_digest", "logprob_bits", "prev_digest", "digest"})


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _is_int(v, lo: int = 0) -> bool:
    """A real int (bool is excluded: JSON true == 1 would otherwise pass) that is >= lo."""
    return type(v) is int and v >= lo


def _extra_keys(d: dict, allowed: frozenset) -> list[str]:
    return sorted(set(d) - allowed)


def _digest(obj) -> str:
    return hashlib.blake2b(_canonical(obj), digest_size=32).hexdigest()


def _is_hex64(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _bits_ok(bits) -> bool:
    if not isinstance(bits, str) or ":" not in bits:
        return False
    dtype, hexpart = bits.split(":", 1)
    if dtype not in _FORMATS:
        return False
    fmt, width = _FORMATS[dtype]
    if len(hexpart) != width or any(c not in "0123456789abcdef" for c in hexpart):
        return False
    v = struct.unpack(fmt, bytes.fromhex(hexpart))[0]
    return not (math.isinf(v) or math.isnan(v))


# ---- revisions ------------------------------------------------------------------

def verify_revision_manifest(d: dict) -> list[str]:
    errs: list[str] = []
    if not isinstance(d, dict):
        return ["manifest is not an object"]
    if d.get("kind") != "llm":
        return [f"manifest kind is {d.get('kind')!r}, not 'llm'"]
    extra = _extra_keys(d, MANIFEST_KEYS)
    if extra:
        errs.append(f"manifest has unbound keys {extra}")
    for k in ("weights_digest", "tokenizer_digest", "parent_digest"):
        if not _is_hex64(d.get(k)):
            errs.append(f"manifest {k} malformed")
    if not _is_int(d.get("step")):
        errs.append("manifest step malformed")
    sampler = d.get("sampler")
    if not isinstance(sampler, dict) or set(sampler) != set(SAMPLER_KEYS):
        errs.append("manifest sampler keys malformed")
    if errs:
        return errs
    computed = _digest({"kind": "llm", "weights_digest": d["weights_digest"],
                        "tokenizer_digest": d["tokenizer_digest"], "sampler": sampler,
                        "parent_digest": d["parent_digest"], "step": d["step"]})
    if computed != d.get("digest"):
        errs.append(f"manifest digest mismatch: stored {d.get('digest')}, computed {computed}")
    return errs


def load_revisions(export_dir: Path) -> tuple[dict[str, dict], list[str]]:
    revs: dict[str, dict] = {}
    errs: list[str] = []
    for p in sorted((export_dir / "revisions").glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (ValueError, OSError) as exc:
            errs.append(f"{p.name}: unreadable ({exc.__class__.__name__})")
            continue
        e = verify_revision_manifest(d)
        if e:
            errs.extend(f"{p.name}: {m}" for m in e)
            continue
        if p.stem != d["digest"]:
            errs.append(f"{p.name}: filename does not match digest")
            continue
        revs[d["digest"]] = d
    for d in revs.values():
        if d["parent_digest"] != GENESIS and d["parent_digest"] not in revs:
            errs.append(f"revision {d['digest'][:12]}: parent {d['parent_digest'][:12]} missing")
    return revs, errs


# ---- tokens and sequences -----------------------------------------------------------

def _token_digest(t: dict) -> str:
    return _digest({"revision_digest": t["revision_digest"], "position": t["position"],
                    "token_id": t["token_id"], "logprob_bits": t["logprob_bits"],
                    "topk": t.get("topk"), "prev_digest": t["prev_digest"]})


def _verify_token(t: dict, pos: int, prev: str, revs: dict[str, dict]) -> list[str]:
    errs = []
    if not isinstance(t, dict):
        return [f"token {pos}: not an object"]
    for k in ("revision_digest", "position", "token_id", "logprob_bits", "prev_digest", "digest"):
        if k not in t:
            return [f"token {pos}: missing {k}"]
    extra = _extra_keys(t, TOKEN_KEYS)
    if extra:
        errs.append(f"token {pos}: unbound keys {extra}")
    if not _is_int(t["position"]) or t["position"] != pos:
        errs.append(f"token {pos}: position is {t['position']!r}")
    if t["prev_digest"] != prev:
        errs.append(f"token {pos}: chain broken")
    if not _bits_ok(t["logprob_bits"]):
        errs.append(f"token {pos}: logprob_bits malformed or non-finite")
    if t.get("topk") is not None:
        if not (isinstance(t["topk"], list) and all(isinstance(e, list) and len(e) == 2
                                                  and _is_int(e[0]) and _bits_ok(e[1]) for e in t["topk"])):
            errs.append(f"token {pos}: topk malformed")
    if not _is_hex64(t["revision_digest"]) or t["revision_digest"] not in revs:
        errs.append(f"token {pos}: revision {str(t['revision_digest'])[:12]} not in store")
    if not _is_int(t["token_id"]):
        errs.append(f"token {pos}: token_id not a non-negative int")
    if not _is_hex64(t["prev_digest"]) or not _is_hex64(t["digest"]):
        errs.append(f"token {pos}: digest fields malformed")
    if errs:
        return errs
    if _token_digest(t) != t["digest"]:
        errs.append(f"token {pos}: digest mismatch")
    return errs


def _verify_scores(seq: dict, revs: dict[str, dict]) -> list[str]:
    errs: list[str] = []
    prev = seq["digest"]
    n = len(seq["tokens"])
    scores = seq.get("scores", [])
    if scores is None:
        scores = []
    if not isinstance(scores, list):
        return ["scores is not a list"]
    seen: set = set()
    for i, s in enumerate(scores):
        if not isinstance(s, dict):
            return errs + [f"score {i}: not an object"]
        for k in ("sequence_digest", "position", "train_revision_digest", "logprob_bits", "prev_digest", "digest"):
            if k not in s:
                errs.append(f"score {i}: missing {k}")
                return errs
        extra = _extra_keys(s, SCORE_KEYS)
        if extra:
            errs.append(f"score {i}: unbound keys {extra}")
        if s["sequence_digest"] != seq["digest"]:
            errs.append(f"score {i}: wrong sequence digest")
        if not (_is_int(s["position"]) and s["position"] < n):
            errs.append(f"score {i}: position {s['position']!r} outside sequence")
        if not _is_hex64(s["train_revision_digest"]) or s["train_revision_digest"] not in revs:
            errs.append(f"score {i}: train revision not in store")
        if errs:
            return errs
        key = (s["position"], s["train_revision_digest"])
        if key in seen:
            errs.append(f"score {i}: duplicate (position, train revision)")
        seen.add(key)
        if not _bits_ok(s["logprob_bits"]):
            errs.append(f"score {i}: logprob_bits malformed")
        if s["prev_digest"] != prev:
            errs.append(f"score {i}: chain broken")
        if errs:
            return errs
        computed = _digest({"sequence_digest": s["sequence_digest"], "position": s["position"],
                            "train_revision_digest": s["train_revision_digest"],
                            "logprob_bits": s["logprob_bits"], "prev_digest": s["prev_digest"]})
        if computed != s["digest"]:
            return errs + [f"score {i}: digest mismatch"]
        prev = s["digest"]
    return errs


def verify_sequence(seq: dict, revs: dict[str, dict], expected_prev: str, expected_index: int) -> list[str]:
    errs: list[str] = []
    if not isinstance(seq, dict):
        return ["sequence file is not an object"]
    for k in ("actor_id", "sequence_index", "sequence_id", "prompt_digest", "prompt_len", "tokens",
              "prev_sequence_digest", "terminal_digest", "reward_bits", "token_digests", "digest"):
        if k not in seq:
            return [f"missing field {k}"]
    extra = _extra_keys(seq, SEQUENCE_KEYS)
    if extra:
        errs.append(f"unbound keys {extra}")
    if not _is_int(seq["actor_id"]) or not isinstance(seq["sequence_id"], str):
        errs.append("actor_id or sequence_id malformed")
    if not _is_int(seq["sequence_index"]) or seq["sequence_index"] != expected_index:
        errs.append(f"sequence_index {seq['sequence_index']!r} != expected {expected_index}")
    if seq["prev_sequence_digest"] != expected_prev:
        errs.append("prev_sequence_digest does not match the previous sequence")
    if not _is_hex64(seq["prompt_digest"]):
        errs.append("prompt_digest malformed")
    if not _is_int(seq["prompt_len"]):
        errs.append("prompt_len malformed")
    if not isinstance(seq["token_digests"], list) or not _is_hex64(seq["prev_sequence_digest"]) \
            or not _is_hex64(seq["terminal_digest"]) or not _is_hex64(seq["digest"]):
        errs.append("digest fields malformed")
    if seq["reward_bits"] is not None and not _bits_ok(seq["reward_bits"]):
        errs.append("reward_bits malformed")
    tokens = seq["tokens"]
    if not isinstance(tokens, list) or not tokens:
        return errs + ["no tokens"]
    prev = seq["prev_sequence_digest"]
    for pos, t in enumerate(tokens):
        e = _verify_token(t, pos, prev, revs)
        if e:
            errs.extend(e)
            break
        prev = t["digest"]
    if errs:
        return errs
    if seq["token_digests"] != [t["digest"] for t in tokens]:
        errs.append("token_digests list does not match tokens")
    if seq["terminal_digest"] != tokens[-1]["digest"]:
        errs.append("terminal_digest is not the last token digest")
    first = revs[tokens[0]["revision_digest"]]
    for t in tokens[1:]:
        r = revs[t["revision_digest"]]
        if r["tokenizer_digest"] != first["tokenizer_digest"]:
            errs.append(f"token {t['position']}: tokenizer changed mid-sequence")
        if r["sampler"] != first["sampler"]:
            errs.append(f"token {t['position']}: sampler config changed mid-sequence")
    computed = _digest({"actor_id": seq["actor_id"], "sequence_index": seq["sequence_index"],
                        "sequence_id": seq["sequence_id"], "prompt_digest": seq["prompt_digest"],
                        "prompt_len": seq["prompt_len"], "token_digests": seq["token_digests"],
                        "reward_bits": seq["reward_bits"], "prev_sequence_digest": seq["prev_sequence_digest"],
                        "terminal_digest": seq["terminal_digest"]})
    if computed != seq["digest"]:
        errs.append("sequence digest mismatch")
    if errs:
        return errs
    return _verify_scores(seq, revs)


def ledger_head(sequence_heads: dict, score_heads: dict, revision_digests: list) -> str:
    """Must match martingale.record.store.ledger_head byte for byte."""
    return _digest({"sequence_heads": {str(a): h for a, h in sorted(sequence_heads.items())},
                    "score_heads": dict(sorted(score_heads.items())),
                    "revisions": sorted(revision_digests)})


def verify_export(export_dir: Path, expected_head: str | None = None) -> dict:
    """Verify a whole export.

    Returns {"ok", "n_revisions", "n_sequences", "n_tokens", "head", "heads": {actor: digest},
    "errors": {file: [...]}}. `expected_head` is the ledger head the producer published out
    of band (run log, W&B field). Without it a forger can tamper and re-sign a whole chain,
    delete a trailing sequence, or drop an unreferenced revision; with it any change moves
    the head and is caught. The head is only computed over files that verified.
    """
    export_dir = Path(export_dir)
    revs, rev_errs = load_revisions(export_dir)
    errors: dict[str, list[str]] = {}
    if rev_errs:
        errors["revisions"] = rev_errs
    heads: dict[int, tuple[str, int]] = {}
    score_heads: dict[str, str] = {}
    n_tokens = 0
    n_seq = 0
    for p in _sequence_files(export_dir):
        n_seq += 1
        try:
            seq = json.loads(p.read_text())
            if not isinstance(seq, dict):
                raise ValueError("not an object")
            if not _is_int(seq.get("actor_id")):
                raise ValueError("actor_id malformed")
            actor = seq["actor_id"]
            prev, idx = heads.get(actor, (GENESIS, 0))
            m = _SEQ_NAME.match(p.name)
            if m is None or int(m.group(1)) != actor or int(m.group(2)) != seq.get("sequence_index"):
                raise ValueError("filename does not match actor_id/sequence_index")
            e = verify_sequence(seq, revs, prev, idx)
        except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
            errors[p.name] = [f"malformed: {exc}"]
            continue
        if e:
            errors[p.name] = e
            continue
        n_tokens += len(seq["tokens"])
        heads[actor] = (seq["digest"], idx + 1)
        if seq.get("scores"):
            score_heads[seq["digest"]] = seq["scores"][-1]["digest"]
    head_digests = {actor: h for actor, (h, _i) in heads.items()}
    head = ledger_head(head_digests, score_heads, list(revs))
    if expected_head is not None and head != expected_head:
        errors.setdefault("head", []).append(f"ledger head {head} != anchored {expected_head}")
    return {"ok": not errors, "anchored": expected_head is not None, "n_revisions": len(revs),
            "n_sequences": n_seq, "n_tokens": n_tokens, "head": head, "heads": head_digests, "errors": errors}


_SEQ_NAME = __import__("re").compile(r"^actor(\d+)_seq(\d+)\.json$")


def _sequence_files(export_dir: Path) -> list[Path]:
    """Sequence files ordered by parsed (actor, index), not lexicographically; unparseable names last."""
    files = list((export_dir / "sequences").glob("*.json"))

    def key(p: Path):
        m = _SEQ_NAME.match(p.name)
        return (0, int(m.group(1)), int(m.group(2)), p.name) if m else (1, 0, 0, p.name)
    return sorted(files, key=key)


def main(argv: list[str]) -> int:
    """python -m checker.verify_tokens <export_dir> (--expected-head HEAD | --unanchored)

    Without an anchor the checker cannot detect re-signed chains, trailing deletions or
    dropped unreferenced revisions, so it refuses to run unless --unanchored is explicit.
    """
    args = argv[1:]
    expected = None
    unanchored = False
    paths = []
    i = 0
    while i < len(args):
        if args[i] == "--expected-head" and i + 1 < len(args):
            expected = args[i + 1]
            i += 2
        elif args[i] == "--unanchored":
            unanchored = True
            i += 1
        else:
            paths.append(args[i])
            i += 1
    if len(paths) != 1 or (expected is None and not unanchored) or (expected is not None and unanchored):
        print(main.__doc__, file=sys.stderr)
        return 2
    report = verify_export(Path(paths[0]), expected_head=expected)
    if not report["anchored"]:
        report["warning"] = "UNANCHORED: re-signed chains, trailing deletions and dropped unreferenced revisions are not detectable"
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
