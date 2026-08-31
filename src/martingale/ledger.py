"""
ledger.py — hash-chained trajectory attestations, mixed-revision aware.

Each action record pins:
  - revision_digest: the content-addressed digest of the behavior policy used
  - state, action: the (s, a) pair
  - behavior_prob: exact rational probability π(a|s) under the behavior revision
  - draw: the raw draw integer and rejection count (for checker re-derivation)
  - env_next_state, env_reward: observed environment transition
  - prev_digest: digest of the previous record (hash chain)

Each trajectory links a sequence of action records into a per-actor chain.

The checker (checker/verify.py) re-derives all probabilities from the revision
store and verifies draw_integer + rejection_count against the full simplex —
this catches the subtle forgery class where the same action probability appears
in two revisions but the draw integers differ.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional

from martingale.draw import DrawResult
from martingale.rational import fraction_to_str, to_fraction
from martingale.revision import RevisionStore, _fsync_file, _fsync_dir


GENESIS_DIGEST = "0" * 64  # sentinel prev_digest for the first record


def _record_canonical_bytes(
    revision_digest: str,
    state: int,
    action: int,
    behavior_prob: Fraction,
    draw_integer: int,
    rejection_count: int,
    env_next_state: int,
    env_reward: Fraction,
    prev_digest: str,
) -> bytes:
    """Canonical serialization of an action record for hashing."""
    obj = {
        "revision_digest": revision_digest,
        "state": state,
        "action": action,
        "behavior_prob": fraction_to_str(behavior_prob),
        "draw_integer": draw_integer,
        "rejection_count": rejection_count,
        "env_next_state": env_next_state,
        "env_reward": fraction_to_str(env_reward),
        "prev_digest": prev_digest,
    }
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class ActionRecord:
    """An immutable ledger record for a single action draw."""
    revision_digest: str
    state: int
    action: int
    behavior_prob: Fraction
    draw: DrawResult
    env_next_state: int
    env_reward: Fraction
    prev_digest: str
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        digest = hashlib.blake2b(
            _record_canonical_bytes(
                self.revision_digest, self.state, self.action,
                self.behavior_prob, self.draw.draw_integer, self.draw.rejection_count,
                self.env_next_state, self.env_reward, self.prev_digest,
            ),
            digest_size=32,
        ).hexdigest()
        object.__setattr__(self, "digest", digest)

    def to_dict(self) -> dict:
        return {
            "revision_digest": self.revision_digest,
            "state": self.state,
            "action": self.action,
            "behavior_prob": fraction_to_str(self.behavior_prob),
            "draw_integer": self.draw.draw_integer,
            "rejection_count": self.draw.rejection_count,
            "env_next_state": self.env_next_state,
            "env_reward": fraction_to_str(self.env_reward),
            "prev_digest": self.prev_digest,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ActionRecord:
        return cls(
            revision_digest=d["revision_digest"],
            state=d["state"],
            action=d["action"],
            behavior_prob=Fraction(d["behavior_prob"]),
            draw=DrawResult(
                action=d["action"],
                draw_integer=d["draw_integer"],
                rejection_count=d["rejection_count"],
            ),
            env_next_state=d["env_next_state"],
            env_reward=Fraction(d["env_reward"]),
            prev_digest=d["prev_digest"],
        )


@dataclass(frozen=True)
class TrajectoryRecord:
    """A complete trajectory: ordered sequence of action records + chain info."""
    actor_id: int
    episode_id: int
    action_records: tuple  # tuple[ActionRecord, ...]
    terminal_digest: str   # digest of the last action record
    prev_trajectory_digest: str  # digest of the prior trajectory in per-actor chain

    def to_dict(self) -> dict:
        return {
            "actor_id": self.actor_id,
            "episode_id": self.episode_id,
            "action_records": [r.to_dict() for r in self.action_records],
            "terminal_digest": self.terminal_digest,
            "prev_trajectory_digest": self.prev_trajectory_digest,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrajectoryRecord:
        recs = tuple(ActionRecord.from_dict(r) for r in d["action_records"])
        return cls(
            actor_id=d["actor_id"],
            episode_id=d["episode_id"],
            action_records=recs,
            terminal_digest=d["terminal_digest"],
            prev_trajectory_digest=d["prev_trajectory_digest"],
        )


class Ledger:
    """
    Persistent hash-chained ledger of trajectory attestations.

    One ledger per actor (identified by actor_id). Trajectories are stored
    as JSON files named by actor_id + episode_id. Writes are durable.
    """

    def __init__(self, directory: Path, revision_store: RevisionStore) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._store = revision_store
        # Per-actor last trajectory digest (for chaining)
        self._prev_traj_digest: dict[int, str] = {}

    @property
    def genesis_digest(self) -> str:
        return GENESIS_DIGEST

    def _traj_path(self, actor_id: int, episode_id: int) -> Path:
        return self._dir / f"actor{actor_id:04d}_ep{episode_id:08d}.json"

    def append_trajectory(
        self,
        actor_id: int,
        episode_id: int,
        records: list[ActionRecord],
    ) -> TrajectoryRecord:
        """
        Append a trajectory to the ledger. Returns the TrajectoryRecord.

        Validates chain integrity: each record's prev_digest must equal
        the digest of the previous record (or genesis_digest for the first).
        Writes durably using atomic rename + fsync.
        """
        if not records:
            raise ValueError("Trajectory must contain at least one action record.")

        # Validate chain
        prev = records[0].prev_digest  # could be genesis or prior record
        for i, rec in enumerate(records):
            if rec.prev_digest != prev:
                raise ValueError(
                    f"Chain broken at record {i}: "
                    f"expected prev_digest={prev}, got {rec.prev_digest}"
                )
            prev = rec.digest

        prev_traj = self._prev_traj_digest.get(actor_id, GENESIS_DIGEST)
        traj = TrajectoryRecord(
            actor_id=actor_id,
            episode_id=episode_id,
            action_records=tuple(records),
            terminal_digest=records[-1].digest,
            prev_trajectory_digest=prev_traj,
        )

        # Durably write trajectory
        path = self._traj_path(actor_id, episode_id)
        tmp = path.with_suffix(".tmp")
        data = json.dumps(traj.to_dict(), indent=2).encode("utf-8")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            _fsync_file(f)
        os.rename(tmp, path)
        _fsync_dir(self._dir)

        self._prev_traj_digest[actor_id] = traj.terminal_digest
        return traj

    def load_trajectory(self, actor_id: int, episode_id: int) -> TrajectoryRecord:
        """Load a trajectory from disk."""
        path = self._traj_path(actor_id, episode_id)
        if not path.exists():
            raise KeyError(f"Trajectory not found: actor={actor_id}, episode={episode_id}")
        with open(path, "rb") as f:
            d = json.loads(f.read())
        return TrajectoryRecord.from_dict(d)

    def all_trajectories(self):
        """Iterate over all stored trajectories."""
        for p in sorted(self._dir.glob("actor*.json")):
            with open(p, "rb") as f:
                d = json.loads(f.read())
            yield TrajectoryRecord.from_dict(d)
