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
# Added 2026-10-10. Emitted only when set, so manifests written before then digest unchanged.
# An absent key means "unknown"; an explicit null is not a canonical form and the checker rejects it.
SAMPLER_OPTIONAL_KEYS: tuple[str, ...] = ("logprobs_source", "temperature_applied")

# vLLM's `logprobs_mode` vocabulary (vllm/config/model.py, v0.28.0 line 250). "raw" is the
# log-softmax of the model's logits; "processed" is after every sampling processor, including
# temperature and top-k/top-p (vllm/v1/sample/sampler.py lines 88-105 and 277-287).
LOGPROBS_MODES: tuple[str, ...] = ("raw_logprobs", "processed_logprobs", "raw_logits", "processed_logits")
PROCESSED_LOGPROBS_MODES: frozenset[str] = frozenset({"processed_logprobs", "processed_logits"})
RAW_LOGPROBS_MODES: frozenset[str] = frozenset({"raw_logprobs", "raw_logits"})


def _is_hex64(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def temperature_applied_for(logprobs_mode: str | None) -> bool | None:
    """Whether an engine reporting in `logprobs_mode` divided by temperature before the log-softmax.
    None when the mode is unknown or not in vLLM's vocabulary (a record written before 2026-10-10
    carries the source name here)."""
    if logprobs_mode in PROCESSED_LOGPROBS_MODES:
        return True
    if logprobs_mode in RAW_LOGPROBS_MODES:
        return False
    return None


@dataclass(frozen=True)
class SamplerConfig:
    """Every key that changes the sampling distribution or the reported log-probs.

    Missing values are stored as null; any change to any key is a new revision.
    `logprobs_mode` is the engine's reporting mode in vLLM's vocabulary (`LOGPROBS_MODES`):
    raw log-probs are the log-softmax of the logits, processed ones have temperature and the
    sampling filters applied, so the ratio semantics differ. `logprobs_source` says which pass
    produced the bits (engine_sampling, trainer_old_logps, trainer_recompute, trainer_scores)
    and `temperature_applied` whether that pass divided the logits by `temperature` first;
    the diagnostics compare lag-0 log-probs only when both sides agree on it. Records written
    before 2026-10-10 carry the source name in `logprobs_mode` and lack the other two keys.
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
    logprobs_source: str | None = None
    temperature_applied: bool | None = None

    def __post_init__(self) -> None:
        if self.temperature_applied is not None and type(self.temperature_applied) is not bool:
            raise ValueError(f"temperature_applied must be a bool or None, got {self.temperature_applied!r}")

    def to_dict(self) -> dict[str, Any]:
        """Canonical form: the twelve original keys always, the optional keys only when set."""
        d: dict[str, Any] = {k: getattr(self, k) for k in SAMPLER_KEYS}
        d.update({k: getattr(self, k) for k in SAMPLER_OPTIONAL_KEYS if getattr(self, k) is not None})
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SamplerConfig:
        unknown = set(d) - set(SAMPLER_KEYS) - set(SAMPLER_OPTIONAL_KEYS)
        if unknown:
            raise ValueError(f"unknown sampler keys {sorted(unknown)}")
        return cls(**{k: d.get(k) for k in SAMPLER_KEYS + SAMPLER_OPTIONAL_KEYS})


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


def _weights_entry(name: str, value: Any) -> bytes:
    """One parameter's bytes in the weights digest: name, dtype, shape, raw bytes (design.md 7.2)."""
    dtype, shape, raw = _tensor_parts(name, value)
    return (name.encode("utf-8") + b"\x00" + dtype.encode("ascii") + b"\x00"
            + ",".join(str(d) for d in shape).encode("ascii") + b"\x00" + raw + b"\xff")


def weights_digest(state_dict: Mapping[str, Any]) -> str:
    """BLAKE2b-256 over sorted (name, dtype, shape, raw bytes) of every parameter (design.md 7.2)."""
    h = hashlib.blake2b(digest_size=32)
    for name in sorted(state_dict):
        h.update(_weights_entry(name, state_dict[name]))
    return h.hexdigest()


class ReceivedWeights:
    """`weights_digest` over (name, tensor) pairs as a rollout replica receives them: in buckets, in
    the sender's order. Each tensor's bytes are copied on arrival (a transfer buffer is reused after
    its callback), so the replica holds one CPU copy of the payload until `digest()`.

    It digests what arrived, before the engine loads it: equality with the sender's digest proves the
    sync delivered those tensors, not that the engine's loader applied them."""

    def __init__(self) -> None:
        self._entries: dict[str, bytes] = {}

    @property
    def tensors(self) -> int:
        return len(self._entries)

    def add(self, named_tensors: Iterable[tuple[str, Any]]) -> None:
        for name, value in named_tensors:
            if name in self._entries:
                raise ValueError(f"parameter {name!r} received twice in one sync")
            self._entries[name] = _weights_entry(name, value)

    def digest(self) -> str | None:
        """The weights digest of everything received; None when nothing arrived (no claim)."""
        if not self._entries:
            return None
        h = hashlib.blake2b(digest_size=32)
        for name in sorted(self._entries):
            h.update(self._entries[name])
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
    received_weights_digest: str | None = None   # inside the digest when set; see sync_verified
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        names: tuple[str, ...] = ("weights_digest", "tokenizer_digest", "parent_digest")
        if self.received_weights_digest is not None:
            names += ("received_weights_digest",)
        for name in names:
            v = getattr(self, name)
            if not _is_hex64(v):
                raise ValueError(f"{name} must be 64 lowercase hex chars, got {v!r}")
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a non-negative int")
        object.__setattr__(self, "digest", digest_json(self.manifest()))

    def manifest(self) -> dict[str, Any]:
        """The digested fields, exactly (design.md 7.1). `received_weights_digest` only when the
        replica reported one, so manifests written before 2026-10-13 digest unchanged."""
        m: dict[str, Any] = {
            "kind": "llm",
            "weights_digest": self.weights_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "sampler": self.sampler.to_dict(),
            "parent_digest": self.parent_digest,
            "step": self.step,
        }
        if self.received_weights_digest is not None:
            m["received_weights_digest"] = self.received_weights_digest
        return m

    @property
    def sync_verified(self) -> bool | None:
        """Whether the rollout replica received exactly the weights this revision names: True or
        False when it reported a digest of what its last sync delivered, None when it did not (no
        sync was observed, which is not a match)."""
        if self.received_weights_digest is None:
            return None
        return self.received_weights_digest == self.weights_digest

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
            received_weights_digest=d.get("received_weights_digest"),
        )
        if "digest" in d and d["digest"] != rev.digest:
            raise ValueError(f"manifest digest mismatch: stored {d['digest']}, computed {rev.digest}")
        return rev
