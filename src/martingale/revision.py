"""
revision.py — content-addressed revision store.

A revision is a snapshot of a policy's probability table, identified by a
BLAKE2b-256 digest of its canonical serialization.

Canonical serialization: sorted-key JSON of (state, action) → "num/den" strings.
This ensures that the same policy table always produces the same digest,
regardless of key insertion order.

The revision store persists revisions to disk for durability and for the
independent checker (checker/verify.py) to re-derive probabilities without
trusting the ledger.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterator

from martingale.rational import fraction_to_str, to_fraction


PolicyTable = dict[int, dict[int, Fraction]]


def _canonical_bytes(table: PolicyTable) -> bytes:
    """
    Produce canonical bytes for a policy table: sorted-key JSON of num/den strings.

    States and actions are sorted numerically to ensure determinism.
    """
    canonical = {}
    for s in sorted(table.keys()):
        canonical[str(s)] = {}
        for a in sorted(table[s].keys()):
            canonical[str(s)][str(a)] = fraction_to_str(to_fraction(table[s][a]))
    return json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class Revision:
    """An immutable snapshot of a policy table with its content-addressed digest."""
    table: dict  # {state: {action: Fraction}}
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        # Normalize the table to Fractions and compute digest
        normalized = {}
        for s, action_probs in self.table.items():
            normalized[int(s)] = {int(a): to_fraction(p) for a, p in action_probs.items()}
        # Use object.__setattr__ because frozen=True
        object.__setattr__(self, "table", normalized)
        digest = hashlib.blake2b(_canonical_bytes(normalized), digest_size=32).hexdigest()
        object.__setattr__(self, "digest", digest)

    def prob(self, state: int, action: int) -> Fraction:
        """Return the exact probability for (state, action) in this revision."""
        return self.table[state][action]


class RevisionStore:
    """
    Persistent content-addressed store of policy revisions.

    Revisions are stored as JSON files in a directory, named by their digest.
    Lookup is O(1) via digest. Supports iteration over all stored digests.
    """

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        # In-memory cache
        self._cache: dict[str, Revision] = {}

    def _path(self, digest: str) -> Path:
        return self._dir / f"{digest}.json"

    def publish(self, table: PolicyTable) -> Revision:
        """
        Store a policy table and return its Revision.

        Idempotent: publishing the same table twice yields the same revision.
        Writes are durable: uses atomic rename with fsync.
        """
        rev = Revision(table)
        if rev.digest in self._cache:
            return self._cache[rev.digest]
        target = self._path(rev.digest)
        if target.exists():
            self._cache[rev.digest] = rev
            return rev

        # Serialize the revision for disk storage
        payload = {
            "digest": rev.digest,
            "table": {
                str(s): {str(a): fraction_to_str(p) for a, p in action_probs.items()}
                for s, action_probs in rev.table.items()
            },
        }
        data = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")

        # Atomic write: temp file → fsync → rename → fsync parent
        tmp = target.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            _fsync_file(f)
        os.rename(tmp, target)
        _fsync_dir(self._dir)

        self._cache[rev.digest] = rev
        return rev

    def get(self, digest: str) -> Revision:
        """Retrieve a Revision by its digest. Raises KeyError if not found."""
        if digest in self._cache:
            return self._cache[digest]
        path = self._path(digest)
        if not path.exists():
            raise KeyError(f"Revision digest not found: {digest}")
        with open(path, "rb") as f:
            payload = json.loads(f.read())
        table = {
            int(s): {int(a): Fraction(p_str) for a, p_str in action_probs.items()}
            for s, action_probs in payload["table"].items()
        }
        rev = Revision(table)
        assert rev.digest == digest, (
            f"Stored digest mismatch: file says {digest}, computed {rev.digest}"
        )
        self._cache[digest] = rev
        return rev

    def __contains__(self, digest: str) -> bool:
        return digest in self._cache or self._path(digest).exists()

    def all_digests(self) -> Iterator[str]:
        """Iterate over all stored revision digests."""
        for p in self._dir.glob("*.json"):
            yield p.stem


def _fsync_file(f) -> None:
    """Platform-aware fsync for durability."""
    import sys
    if sys.platform == "darwin":
        import fcntl
        fcntl.fcntl(f.fileno(), fcntl.F_FULLFSYNC)
    else:
        os.fsync(f.fileno())


def _fsync_dir(directory: Path) -> None:
    """fsync a directory to ensure rename is durable."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        pass  # Best-effort on platforms that don't support directory fsync
