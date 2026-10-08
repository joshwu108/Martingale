"""
record/replay.py — provenance of a replayed row.

A replay buffer trains on rows the current policy did not generate. When it
hands such a row to the trainer it registers it here, so the row sits in the
token record next to the fresh rows with provenance "replayed": its tokens
carry the behaviour log-prob bits and the revision that produced them (copied
from the fresh row they came from, when that row is in this record), and a
ReplayProvenance block binds the buffer's draw id, its content digest of the
row, the importance weight it applied and any policy rescale, all inside the
sequence digest. Weights are stored as exact reduced fractions (a float is a
rational; Fraction(float) is exact), never as rounded decimals.

What this binds and what it does not: docs/replay-provenance.md and
docs/nonclaims.md ("Replayed rows").
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from martingale.record.tokens import SequenceRecord

PROVENANCE_FRESH = "fresh"
PROVENANCE_REPLAYED = "replayed"
PROVENANCES: tuple[str, ...] = (PROVENANCE_FRESH, PROVENANCE_REPLAYED)
REPLAY_KEYS: tuple[str, ...] = ("draw_id", "content_digest", "is_weight_num", "is_weight_den",
                                "rescale_num", "rescale_den", "origin_digest")


class OriginNotFound(LookupError):
    """No fresh sequence in this record can serve as the replayed row's origin (none matches the prompt,
    completion and behaviour step; or the named digest is missing or is itself a replayed row)."""


class AmbiguousOrigin(OriginNotFound):
    """Several fresh sequences match the replayed row and nothing tells them apart (pass the behaviour
    log-probs the buffer stored, or the origin digest)."""


def _is_hex64(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _exact(name: str, value: Any) -> Fraction:
    """An exact non-negative rational from a Fraction, int or finite float; bools and non-finite floats refused."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got a bool")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    try:
        f = Fraction(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a Fraction, int or finite float, got {value!r}") from exc
    if f < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return f


def _check_ratio(name: str, num: Any, den: Any, positive: bool = False) -> None:
    for k, v in ((f"{name}_num", num), (f"{name}_den", den)):
        if type(v) is not int:
            raise ValueError(f"{k} must be an int, got {v!r}")
    if den < 1 or num < 0 or (positive and num == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'} with a positive denominator, "
                         f"got {num}/{den}")
    if math.gcd(num, den) != 1:
        raise ValueError(f"{name} must be a reduced fraction, got {num}/{den}")


@dataclass(frozen=True)
class ReplayProvenance:
    """Everything a replay buffer declares about a row it replays; every field is inside the sequence digest."""
    draw_id: str                        # the buffer's identifier of this draw (opaque, non-empty)
    content_digest: str                 # the buffer's content digest of the row (64 hex)
    is_weight_num: int                  # importance weight the buffer applied, exact and reduced
    is_weight_den: int
    rescale_num: int = 1                # any policy rescale of that weight (1/1: none), exact and reduced
    rescale_den: int = 1
    origin_digest: str | None = None    # the fresh sequence in this record the row was copied from, when known

    def __post_init__(self) -> None:
        if not isinstance(self.draw_id, str) or not self.draw_id:
            raise ValueError(f"draw_id must be a non-empty str, got {self.draw_id!r}")
        if not _is_hex64(self.content_digest):
            raise ValueError(f"content_digest must be 64 lowercase hex chars, got {self.content_digest!r}")
        _check_ratio("is_weight", self.is_weight_num, self.is_weight_den)
        _check_ratio("rescale", self.rescale_num, self.rescale_den, positive=True)
        if self.origin_digest is not None and not _is_hex64(self.origin_digest):
            raise ValueError(f"origin_digest must be None or 64 lowercase hex chars, got {self.origin_digest!r}")

    @classmethod
    def of(cls, draw_id: str, content_digest: str, *, is_weight: Fraction | float | int,
           rescale: Fraction | float | int = 1, origin_digest: str | None = None) -> ReplayProvenance:
        """Build from the buffer's numbers; a float weight is stored as its exact dyadic value."""
        w = _exact("is_weight", is_weight)
        r = _exact("rescale", rescale)
        if r == 0:
            raise ValueError("rescale must be positive")
        return cls(draw_id=draw_id, content_digest=content_digest,
                   is_weight_num=w.numerator, is_weight_den=w.denominator,
                   rescale_num=r.numerator, rescale_den=r.denominator, origin_digest=origin_digest)

    @property
    def is_weight(self) -> Fraction:
        return Fraction(self.is_weight_num, self.is_weight_den)

    @property
    def rescale(self) -> Fraction:
        return Fraction(self.rescale_num, self.rescale_den)

    @property
    def applied_weight(self) -> Fraction:
        """The weight the loss saw: importance weight times the policy rescale."""
        return self.is_weight * self.rescale

    def canonical(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in REPLAY_KEYS}

    def to_dict(self) -> dict[str, Any]:
        return self.canonical()

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ReplayProvenance:
        if not isinstance(d, Mapping):
            raise ValueError(f"replay block must be an object, got {type(d).__name__}")
        unknown = sorted(set(d) - set(REPLAY_KEYS))
        missing = sorted(set(REPLAY_KEYS) - set(d))
        if unknown or missing:
            raise ValueError(f"replay block keys: unknown {unknown}, missing {missing}")
        return cls(**{k: d[k] for k in REPLAY_KEYS})


def check_replay_origin(seq: SequenceRecord, origin: SequenceRecord) -> None:
    """Refuse a replayed row whose prompt or tokens are not those of the fresh sequence it claims to copy."""
    if origin.replay is not None:
        raise ValueError(f"replay origin {origin.digest[:12]} must be a fresh sequence")
    if seq.prompt_digest != origin.prompt_digest or seq.prompt_len != origin.prompt_len:
        raise ValueError("replayed row and its origin differ: prompt")
    if len(seq.tokens) != len(origin.tokens):
        raise ValueError(f"replayed row and its origin differ: {len(seq.tokens)} vs {len(origin.tokens)} tokens")
    for s, o in zip(seq.tokens, origin.tokens):
        if (s.token_id, s.revision_digest, s.logprob_bits, s.topk) != (o.token_id, o.revision_digest, o.logprob_bits, o.topk):
            raise ValueError(f"replayed row and its origin differ at token {s.position}")
