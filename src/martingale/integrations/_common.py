"""Pieces the TRL and verl adapters share: the row-id contract and the tensor/weights helpers.

A row id is ``rank << 40 | local``: the rank of the recorder that allocated it and that
recorder's own counter. Every id is bound to its sequence digest in the ledger's ``row_ids``
table, so any recorder over the same workspace (another rank, a Ray worker process, a
recorder whose in-memory maps have rolled over) resolves it (docs/replay-provenance.md,
"More than one process"). Imports nothing from either framework or adapter.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from martingale.record import SamplerConfig, temperature_applied_for

if TYPE_CHECKING:
    import torch

    from martingale.record.store import TokenLedger

ROW_ID_KEY = "martingale_row_id"
KEEP_GENERATIONS = 2          # row maps kept for the last N generation batches
ROW_ID_RANK_BITS = 40         # row id = rank << 40 | local counter; rank 0 ids are the plain counter
_MAX_RANK = 1 << 23           # int64 leaves 23 bits for the rank
_LOCAL_MASK = (1 << ROW_ID_RANK_BITS) - 1
ENGINE_ONLY_OVERRIDES = frozenset({"logprobs_mode", "temperature_applied"})   # describe the engine's bits, not the trainer's

ContentKey = tuple[tuple[int, ...], tuple[int, ...]]                      # (prompt ids, completion ids)
Generation = tuple[dict[int, str], dict[ContentKey, list[str]]]           # (row id -> digest, content -> digests)


def pack_row_id(rank: int, local_id: int) -> int:
    """The ``martingale_row_id`` for the ``local_id``-th row the recorder on ``rank`` allocated."""
    if not 0 <= rank < _MAX_RANK:
        raise ValueError(f"rank must be in [0, {_MAX_RANK}), got {rank}")
    if not 0 <= local_id <= _LOCAL_MASK:
        raise ValueError(f"local row id must be in [0, 2**{ROW_ID_RANK_BITS}), got {local_id}")
    return (rank << ROW_ID_RANK_BITS) | local_id


def unpack_row_id(row_id: int) -> tuple[int, int]:
    """(rank, local id) of a packed row id; a negative id (the skip sentinel) unpacks to (-1, -1)."""
    if row_id < 0:
        return -1, -1
    return row_id >> ROW_ID_RANK_BITS, row_id & _LOCAL_MASK


def resolve_row_id(generations: list[Generation], ledger: TokenLedger, row_id: int) -> str | None:
    """The recorded sequence behind a row id: the in-memory maps of the last KEEP_GENERATIONS
    batches first, then the ledger's row-id table (any rank, any age, any process)."""
    for by_id, _by_content in reversed(generations):
        if row_id in by_id:
            return by_id[row_id]
    rank, local = unpack_row_id(int(row_id))
    return None if rank < 0 else ledger.row_digest(rank, local)


def next_by_content(generations: list[Generation], key: ContentKey) -> str | None:
    """Content fallback for a row without an id: newest batch first, round-robin over duplicates
    so identical rows in one batch map to distinct sequences."""
    for _by_id, by_content in reversed(generations):
        lst = by_content.get(key)
        if lst:
            digest = lst.pop(0)
            lst.append(digest)
            return digest
    return None


def _tokenizer_bytes(tokenizer: Any) -> bytes | str:
    """tokenizer.json bytes from a HF tokenizer, raw bytes, or a precomputed digest."""
    if isinstance(tokenizer, (bytes, str)):
        return tokenizer
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        return backend.to_str().encode("utf-8")
    vocab = getattr(tokenizer, "get_vocab", None)
    if vocab is not None:
        import json
        return json.dumps(vocab(), sort_keys=True).encode("utf-8")
    raise TypeError("tokenizer must be a HF tokenizer, tokenizer.json bytes, or a 64-hex digest")


def _prefix_lengths(mask: torch.Tensor, side: str) -> list[int]:
    """Lengths of a right-padded ("prefix": 1..1 0..0) or left-padded ("suffix": 0..0 1..1) mask.
    Refuses a mask that is not pure padding, because slicing by sum() would misalign tokens silently."""
    import torch

    m = mask.to("cpu").long()
    lengths = m.sum(dim=1)
    n = m.size(1)
    idx = torch.arange(n).unsqueeze(0)
    expected = (idx < lengths.unsqueeze(1)) if side == "prefix" else (idx >= (n - lengths).unsqueeze(1))
    if not bool((m.bool() == expected).all()):
        raise ValueError(f"{side} mask is not pure padding; cannot slice tokens safely")
    return [int(x) for x in lengths.tolist()]


def _hole_rows(mask: torch.Tensor) -> list[bool]:
    """Per row of a right-padded mask: True when it is not 1..1 0..0 (a multi-turn response whose
    tool text is masked out inside the completion)."""
    import torch

    m = mask.to("cpu").long()
    expected = (torch.arange(m.size(1)).unsqueeze(0) < m.sum(dim=1, keepdim=True)).long()
    return [bool(x) for x in (m != expected).any(dim=1).tolist()]


def _check_digestable(state_dict: dict) -> None:
    for name, t in state_dict.items():
        if getattr(t, "is_meta", False) or (hasattr(t, "numel") and t.numel() == 0):
            raise RuntimeError(
                f"parameter {name!r} is sharded or on the meta device; the weights digest would be meaningless. "
                "Gather the full state dict (FSDP/ZeRO-3) before recording.")


def check_sampler_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Validate user overrides once, at construction: unknown keys and `logprobs_source` are refused
    (the source is what the adapter observed; relabelling it would merge behavior and trainer revisions)."""
    if "logprobs_source" in overrides:
        raise ValueError("sampler_overrides cannot set logprobs_source; the adapter records where the bits came from")
    SamplerConfig.from_dict(overrides)
    return dict(overrides)


def apply_sampler_overrides(sampler: SamplerConfig, overrides: Mapping[str, Any], logprobs_source: str) -> SamplerConfig:
    """User overrides on top of what the adapter read. `logprobs_mode` and `temperature_applied` say what
    the engine reported, so they are applied to engine-sourced revisions only; the trainer-side values
    come from the framework's own code and an override must not make the two sides agree by accident.
    An overridden `logprobs_mode` re-derives `temperature_applied` unless that is overridden too."""
    if not overrides:
        return sampler
    skip = () if logprobs_source == "engine_sampling" else ENGINE_ONLY_OVERRIDES
    merged = {**sampler.to_dict(), **{k: v for k, v in overrides.items() if k not in skip}}
    if "logprobs_mode" in overrides and "temperature_applied" not in overrides and not skip:
        merged["temperature_applied"] = temperature_applied_for(merged["logprobs_mode"])
    return SamplerConfig.from_dict(merged)
