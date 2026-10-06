"""
record/revision.py — LLMRevision: a content-addressed snapshot of (weights,
tokenizer, sampler config, learner step, parent).

The digest is over the manifest, not the weights themselves; weights live
wherever the trainer put them (`weights_uri`, outside the digest) and are
identified by `weights_digest`, which is over raw tensor bytes (design.md 7.2),
not over torch.save output (whose pickle framing changes across versions).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from martingale.record.bits import GENESIS_DIGEST, digest_bytes, digest_json

SAMPLER_KEYS: tuple[str, ...] = (
    "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "max_tokens",
    "logprobs_mode", "dtype", "engine", "engine_version", "tensor_parallel", "seed_policy",
)


@dataclass(frozen=True)
class SamplerConfig:
    """Every key that changes the sampling distribution or the reported log-probs.

    Missing values are stored as null; any change to any key is a new revision.
    `logprobs_mode` records whether the engine reports raw or post-filter log-probs
    (vLLM's `logprobs_mode`), because the ratio semantics differ.
    """
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = None
    logprobs_mode: str | None = None
    dtype: str | None = None
    engine: str | None = None
    engine_version: str | None = None
    tensor_parallel: int | None = None
    seed_policy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in SAMPLER_KEYS}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SamplerConfig:
        unknown = set(d) - set(SAMPLER_KEYS)
        if unknown:
            raise ValueError(f"unknown sampler keys {sorted(unknown)}")
        return cls(**{k: d.get(k) for k in SAMPLER_KEYS})


def _tensor_parts(name: str, value: Any) -> tuple[str, tuple[int, ...], bytes]:
    """(dtype string, shape, raw little-endian bytes) for a torch tensor, numpy array,
    or an explicit (dtype, shape, bytes) triple. Keeps the stored dtype; never casts."""
    if isinstance(value, tuple) and len(value) == 3:
        dtype, shape, raw = value
        return str(dtype), tuple(int(x) for x in shape), bytes(raw)
    if hasattr(value, "detach") and hasattr(value, "untyped_storage"):     # torch.Tensor
        t = value.detach().cpu().contiguous()
        dtype = str(t.dtype).removeprefix("torch.")
        flat = t.reshape(-1)
        raw = flat.view(_uint8_dtype_for(t)).numpy().tobytes() if flat.numel() else b""
        return dtype, tuple(t.shape), raw
    if hasattr(value, "tobytes") and hasattr(value, "dtype"):                # numpy.ndarray
        import numpy as np
        arr = np.ascontiguousarray(value)
        return str(arr.dtype), tuple(arr.shape), arr.astype(arr.dtype.newbyteorder("<")).tobytes()
    raise TypeError(f"cannot digest parameter {name!r} of type {type(value).__name__}")


def _uint8_dtype_for(t: Any):
    import torch
    return torch.uint8


def weights_digest(state_dict: Mapping[str, Any]) -> str:
    """BLAKE2b-256 over sorted (name, dtype, shape, raw bytes) of every parameter (design.md 7.2)."""
    h = hashlib.blake2b(digest_size=32)
    for name in sorted(state_dict):
        dtype, shape, raw = _tensor_parts(name, state_dict[name])
        h.update(name.encode("utf-8") + b"\x00" + dtype.encode("ascii") + b"\x00"
                 + ",".join(str(d) for d in shape).encode("ascii") + b"\x00" + raw + b"\xff")
    return h.hexdigest()


def tokenizer_digest(data: bytes | Iterable[bytes]) -> str:
    """BLAKE2b-256 of tokenizer.json bytes, or of several files concatenated in the order given."""
    if isinstance(data, (bytes, bytearray)):
        return digest_bytes(bytes(data))
    h = hashlib.blake2b(digest_size=32)
    for chunk in data:
        h.update(bytes(chunk))
    return h.hexdigest()


@dataclass(frozen=True)
class LLMRevision:
    weights_digest: str
    tokenizer_digest: str
    sampler: SamplerConfig
    step: int
    parent_digest: str = GENESIS_DIGEST
    weights_uri: str | None = None          # outside the digest
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("weights_digest", "tokenizer_digest", "parent_digest"):
            v = getattr(self, name)
            if not (isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)):
                raise ValueError(f"{name} must be 64 lowercase hex chars, got {v!r}")
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a non-negative int")
        object.__setattr__(self, "digest", digest_json(self.manifest()))

    def manifest(self) -> dict[str, Any]:
        """The digested fields, exactly (design.md 7.1)."""
        return {
            "kind": "llm",
            "weights_digest": self.weights_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "sampler": self.sampler.to_dict(),
            "parent_digest": self.parent_digest,
            "step": self.step,
        }

    def to_dict(self) -> dict[str, Any]:
        d = self.manifest()
        d["digest"] = self.digest
        d["weights_uri"] = self.weights_uri
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LLMRevision:
        if d.get("kind") != "llm":
            raise ValueError(f"not an llm revision manifest: kind={d.get('kind')!r}")
        rev = cls(
            weights_digest=d["weights_digest"], tokenizer_digest=d["tokenizer_digest"],
            sampler=SamplerConfig.from_dict(d["sampler"]), step=int(d["step"]),
            parent_digest=d.get("parent_digest", GENESIS_DIGEST), weights_uri=d.get("weights_uri"),
        )
        if "digest" in d and d["digest"] != rev.digest:
            raise ValueError(f"manifest digest mismatch: stored {d['digest']}, computed {rev.digest}")
        return rev
