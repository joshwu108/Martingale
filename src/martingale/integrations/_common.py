"""Pieces the TRL and verl adapters share: the row-id contract and the tensor/weights helpers.

A row id is ``rank << 40 | local``: the rank of the recorder that allocated it and that
recorder's own counter. Every id is bound to its sequence digest in the ledger's ``row_ids``
table, so any recorder over the same workspace (another rank, a Ray worker process, a
recorder whose in-memory maps have rolled over) resolves it (docs/replay-provenance.md,
"More than one process"). Imports nothing from either framework or adapter.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from martingale.record.store import TokenLedger

ROW_ID_KEY = "martingale_row_id"
KEEP_GENERATIONS = 2          # row maps kept for the last N generation batches
ROW_ID_RANK_BITS = 40         # row id = rank << 40 | local counter; rank 0 ids are the plain counter
_MAX_RANK = 1 << 23           # int64 leaves 23 bits for the rank
_LOCAL_MASK = (1 << ROW_ID_RANK_BITS) - 1

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


def _check_digestable(state_dict: dict) -> None:
    for name, t in state_dict.items():
        if getattr(t, "is_meta", False) or (hasattr(t, "numel") and t.numel() == 0):
            raise RuntimeError(
                f"parameter {name!r} is sharded or on the meta device; the weights digest would be meaningless. "
                "Gather the full state dict (FSDP/ZeRO-3) before recording.")
