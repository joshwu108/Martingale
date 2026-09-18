"""
prod/actor.py — AsyncActor with pin-before-draw context manager.

The PinnedContext enforces the protocol structurally:
  - sample_and_record() requires an active pin (raises ProtocolViolation otherwise)
  - The pinned revision digest is recorded in every action record
  - commit_episode() flushes the accumulated records to the ledger

Usage:
    with actor.pin_revision(checkpoint_digest) as ctx:
        for step in range(horizon):
            result = actor.sample_and_record(obs, log_probs, episode_id, step)
        ctx.commit_episode(episode_id)
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

import torch

from martingale.draw import DrawResult
from martingale.ledger import ActionRecord, GENESIS_DIGEST
from martingale.store.sqlite import SQLiteLedger


class ProtocolViolation(RuntimeError):
    """Raised when the pin-before-draw protocol is violated."""


@dataclass
class _SampledStep:
    revision_digest: str
    state_hash: int
    action: int
    log_prob_behavior: float   # float32 log-prob under behavior policy
    env_next_state_hash: int
    step: int


class PinnedContext:
    """
    Context object returned by actor.pin_revision(). Holds the pinned revision
    digest for the duration of the with-block. Steps are accumulated in the
    actor's shared episode buffer (keyed by episode_id), so mixed-revision
    episodes work naturally. commit_episode() flushes the shared buffer.
    """

    def __init__(self, revision_digest: str, actor: "AsyncActor") -> None:
        self.revision_digest = revision_digest
        self._actor = actor

    def commit_episode(self, episode_id: int) -> None:
        """Flush all accumulated steps for this episode to the ledger."""
        steps = self._actor._episode_buffer.pop(episode_id, [])
        self._actor._flush_episode(episode_id, steps)


class AsyncActor:
    """
    Production actor that enforces the pin-before-draw protocol.

    Each call to sample_and_record() must occur inside a pin_revision() context.
    Records are written to the SQLite ledger via commit_episode().
    """

    def __init__(
        self,
        actor_id: int,
        ledger: SQLiteLedger,
        seed: bytes = b"",
    ) -> None:
        self._actor_id = actor_id
        self._ledger = ledger
        self._seed = seed
        self._pinned: Optional[PinnedContext] = None
        self._prev_digest = GENESIS_DIGEST
        self._episode_buffer: dict[int, list[_SampledStep]] = {}

    def pin_revision(self, revision_digest: str) -> PinnedContext:
        """Return a context manager that pins the given revision."""
        return _PinContextManager(self, revision_digest)

    def sample_and_record(
        self,
        obs: torch.Tensor,
        log_probs: torch.Tensor,
        episode_id: int = 0,
        step: int = 0,
    ) -> dict:
        """
        Sample an action from log_probs and record it.

        Must be called inside a pin_revision() context.
        Raises ProtocolViolation if no revision is pinned.

        Args:
            obs: observation tensor (used as state hash)
            log_probs: log-probabilities over actions under the pinned policy
            episode_id: episode identifier
            step: step within the episode

        Returns:
            dict with action, log_prob, revision_digest
        """
        if self._pinned is None:
            raise ProtocolViolation(
                "sample_and_record() called without an active pin. "
                "Use 'with actor.pin_revision(digest) as ctx:' first."
            )

        # Sample action from the distribution
        probs = torch.softmax(log_probs, dim=-1)
        action = torch.multinomial(probs, num_samples=1).item()
        log_prob_b = log_probs[action].item()

        # Encode state as a hash of the observation tensor
        import struct
        obs_list = obs.detach().cpu().float().tolist()
        obs_bytes = struct.pack(f"{len(obs_list)}f", *obs_list)
        state_hash = int.from_bytes(
            hashlib.blake2b(obs_bytes, digest_size=4).digest(), "big"
        )

        rec = _SampledStep(
            revision_digest=self._pinned.revision_digest,
            state_hash=state_hash,
            action=int(action),
            log_prob_behavior=log_prob_b,
            env_next_state_hash=0,  # set by environment feedback
            step=step,
        )
        # Append to shared episode buffer (supports mixed-revision episodes)
        self._episode_buffer.setdefault(episode_id, []).append(rec)

        return {
            "action": int(action),
            "log_prob": log_prob_b,
            "revision_digest": self._pinned.revision_digest,
        }

    def _flush_episode(self, episode_id: int, steps: list[_SampledStep]) -> None:
        """Convert accumulated steps to ActionRecords and write to ledger."""
        if not steps:
            return

        records = []
        prev = self._prev_digest
        for i, step in enumerate(steps):
            # Encode log_prob as a Fraction for ledger compatibility
            # Use a rational approximation of the float log-prob
            lp = Fraction(step.log_prob_behavior).limit_denominator(10**9)
            behavior_prob = torch.exp(torch.tensor(step.log_prob_behavior)).item()
            behavior_prob_frac = Fraction(behavior_prob).limit_denominator(10**9)

            draw = DrawResult(
                action=step.action,
                draw_integer=step.state_hash,  # encode state as draw integer
                rejection_count=0,
            )
            rec = ActionRecord(
                revision_digest=step.revision_digest,
                state=step.state_hash % 2**31,
                action=step.action,
                behavior_prob=behavior_prob_frac,
                draw=draw,
                env_next_state=step.env_next_state_hash % 2**31,
                env_reward=Fraction(0),
                prev_digest=prev,
            )
            records.append(rec)
            prev = rec.digest

        self._ledger.append_trajectory(
            actor_id=self._actor_id,
            episode_id=episode_id,
            records=records,
        )
        self._prev_digest = prev


class _PinContextManager:
    """Internal context manager returned by AsyncActor.pin_revision()."""

    def __init__(self, actor: AsyncActor, revision_digest: str) -> None:
        self._actor = actor
        self._revision_digest = revision_digest
        self._ctx: Optional[PinnedContext] = None

    def __enter__(self) -> PinnedContext:
        self._ctx = PinnedContext(self._revision_digest, self._actor)
        self._actor._pinned = self._ctx
        return self._ctx

    def __exit__(self, *_) -> None:
        self._actor._pinned = None
        return False
