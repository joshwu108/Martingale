"""
record/tokens.py — TokenRecord, SequenceRecord, ScoreRecord (design.md 7.4, 7.5).

Chains:
  tokens    per sequence; the first token's prev_digest is the sequence's
            prev_sequence_digest (GENESIS for an actor's first sequence), so
            sequence order is bound into the token chain.
  sequences per actor via prev_sequence_digest and sequence_index.
  scores    per sequence: the trainer's log-prob for each recorded token under
            the revision it trained with, chained from the sequence digest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from martingale.record.bits import GENESIS_DIGEST, digest_bytes, digest_json, parse_bits


def prompt_digest(prompt_ids: Sequence[int]) -> str:
    """BLAKE2b-256 over prompt token ids as little-endian signed 32-bit ints."""
    raw = b"".join(int(t).to_bytes(4, "little", signed=True) for t in prompt_ids)
    return digest_bytes(raw)


def _check_int(name: str, v: Any, lo: int = 0) -> None:
    if type(v) is not int or v < lo:
        raise ValueError(f"{name} must be an int >= {lo}, got {v!r}")


def _check_hex(name: str, v: Any) -> None:
    if not (isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)):
        raise ValueError(f"{name} must be 64 lowercase hex chars, got {v!r}")


@dataclass(frozen=True)
class TokenRecord:
    revision_digest: str
    position: int
    token_id: int
    logprob_bits: str
    prev_digest: str
    topk: tuple[tuple[int, str], ...] | None = None
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _check_hex("revision_digest", self.revision_digest)
        _check_hex("prev_digest", self.prev_digest)
        _check_int("position", self.position)
        _check_int("token_id", self.token_id)
        parse_bits(self.logprob_bits)
        if self.topk is not None:
            object.__setattr__(self, "topk", tuple((t, b) for t, b in self.topk))
            for t, b in self.topk:
                _check_int("topk token id", t)
                parse_bits(b)
        object.__setattr__(self, "digest", digest_json(self.canonical()))

    def canonical(self) -> dict[str, Any]:
        return {
            "revision_digest": self.revision_digest,
            "position": self.position,
            "token_id": self.token_id,
            "logprob_bits": self.logprob_bits,
            "topk": None if self.topk is None else [[t, b] for t, b in self.topk],
            "prev_digest": self.prev_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        d = self.canonical()
        d["digest"] = self.digest
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TokenRecord:
        rec = cls(revision_digest=d["revision_digest"], position=int(d["position"]),
                  token_id=int(d["token_id"]), logprob_bits=d["logprob_bits"],
                  prev_digest=d["prev_digest"],
                  topk=None if d.get("topk") is None else tuple((t, b) for t, b in d["topk"]))
        if "digest" in d and d["digest"] != rec.digest:
            raise ValueError("token record digest mismatch")
        return rec


@dataclass(frozen=True)
class SequenceRecord:
    actor_id: int
    sequence_index: int               # per-actor counter, 0-based
    sequence_id: str                  # caller-supplied label (prompt id / sample id)
    prompt_digest: str
    prompt_len: int
    tokens: tuple[TokenRecord, ...]
    prev_sequence_digest: str
    reward_bits: str | None = None
    engine_request_id: str | None = None   # outside the digest
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _check_hex("prompt_digest", self.prompt_digest)
        _check_hex("prev_sequence_digest", self.prev_sequence_digest)
        _check_int("actor_id", self.actor_id)
        _check_int("sequence_index", self.sequence_index)
        _check_int("prompt_len", self.prompt_len)
        if not isinstance(self.sequence_id, str):
            raise ValueError("sequence_id must be a str")
        if not self.tokens:
            raise ValueError("a sequence needs at least one token")
        object.__setattr__(self, "tokens", tuple(self.tokens))
        if self.reward_bits is not None:
            parse_bits(self.reward_bits)
        prev = self.prev_sequence_digest
        for i, t in enumerate(self.tokens):
            if t.position != i:
                raise ValueError(f"token {i} has position {t.position}")
            if t.prev_digest != prev:
                raise ValueError(f"token {i} prev_digest does not chain")
            prev = t.digest
        object.__setattr__(self, "digest", digest_json(self.canonical()))

    @property
    def terminal_digest(self) -> str:
        return self.tokens[-1].digest

    @property
    def revision_digests(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(t.revision_digest for t in self.tokens))

    @property
    def is_mixed_revision(self) -> bool:
        return len(self.revision_digests) > 1

    def canonical(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "sequence_index": self.sequence_index,
            "sequence_id": self.sequence_id,
            "prompt_digest": self.prompt_digest,
            "prompt_len": self.prompt_len,
            "token_digests": [t.digest for t in self.tokens],
            "reward_bits": self.reward_bits,
            "prev_sequence_digest": self.prev_sequence_digest,
            "terminal_digest": self.terminal_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        d = self.canonical()
        d["digest"] = self.digest
        d["engine_request_id"] = self.engine_request_id
        d["tokens"] = [t.to_dict() for t in self.tokens]
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SequenceRecord:
        rec = cls(actor_id=int(d["actor_id"]), sequence_index=int(d["sequence_index"]),
                  sequence_id=str(d["sequence_id"]), prompt_digest=d["prompt_digest"],
                  prompt_len=int(d["prompt_len"]),
                  tokens=tuple(TokenRecord.from_dict(t) for t in d["tokens"]),
                  prev_sequence_digest=d["prev_sequence_digest"],
                  reward_bits=d.get("reward_bits"), engine_request_id=d.get("engine_request_id"))
        if "digest" in d and d["digest"] != rec.digest:
            raise ValueError("sequence record digest mismatch")
        return rec


@dataclass(frozen=True)
class ScoreRecord:
    """The trainer's log-prob for one recorded token under the revision it trained with."""
    sequence_digest: str
    position: int
    train_revision_digest: str
    logprob_bits: str
    prev_digest: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        _check_hex("sequence_digest", self.sequence_digest)
        _check_hex("train_revision_digest", self.train_revision_digest)
        _check_hex("prev_digest", self.prev_digest)
        _check_int("position", self.position)
        parse_bits(self.logprob_bits)
        object.__setattr__(self, "digest", digest_json(self.canonical()))

    def canonical(self) -> dict[str, Any]:
        return {"sequence_digest": self.sequence_digest, "position": self.position,
                "train_revision_digest": self.train_revision_digest,
                "logprob_bits": self.logprob_bits, "prev_digest": self.prev_digest}

    def to_dict(self) -> dict[str, Any]:
        d = self.canonical()
        d["digest"] = self.digest
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ScoreRecord:
        rec = cls(sequence_digest=d["sequence_digest"], position=int(d["position"]),
                  train_revision_digest=d["train_revision_digest"],
                  logprob_bits=d["logprob_bits"], prev_digest=d["prev_digest"])
        if "digest" in d and d["digest"] != rec.digest:
            raise ValueError("score record digest mismatch")
        return rec


def build_sequence(
    actor_id: int, sequence_index: int, sequence_id: str, prompt_ids: Sequence[int],
    steps: Sequence[tuple[str, int, str, tuple[tuple[int, str], ...] | None]],
    prev_sequence_digest: str = GENESIS_DIGEST, reward_bits: str | None = None,
    engine_request_id: str | None = None,
) -> SequenceRecord:
    """Chain (revision_digest, token_id, logprob_bits, topk) steps into a SequenceRecord."""
    tokens: list[TokenRecord] = []
    prev = prev_sequence_digest
    for pos, (rev, tok, bits, topk) in enumerate(steps):
        rec = TokenRecord(revision_digest=rev, position=pos, token_id=tok,
                          logprob_bits=bits, prev_digest=prev, topk=topk)
        tokens.append(rec)
        prev = rec.digest
    return SequenceRecord(actor_id=actor_id, sequence_index=sequence_index, sequence_id=sequence_id,
                          prompt_digest=prompt_digest(prompt_ids), prompt_len=len(prompt_ids),
                          tokens=tuple(tokens), prev_sequence_digest=prev_sequence_digest,
                          reward_bits=reward_bits, engine_request_id=engine_request_id)
