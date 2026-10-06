"""
prod/actor.py — reference AsyncActor over the token record.

Pin-before-draw is enforced structurally: sample_and_record() needs an active
pin, and every recorded token carries the pinned revision digest, so an
episode whose pin changes mid-way is a mixed-revision sequence (first class).

    actor = AsyncActor(actor_id=0, recorder=rec)
    with actor.pin_revision(rev.digest) as ctx:
        for step in range(horizon):
            out = actor.sample_and_record(log_probs, episode_id=0, step=step)
        ctx.commit_episode(0, reward=1.0)

Sampling is done here with torch.multinomial because this is a reference
actor; in a real stack the inference engine samples and reports log-probs.
Nothing on this path claims the draw itself is verifiable (docs/nonclaims.md).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from martingale.record.recorder import ProtocolViolation, Recorder, SequenceBuilder

if TYPE_CHECKING:
    import torch

__all__ = ["AsyncActor", "PinnedContext", "ProtocolViolation"]


class PinnedContext:
    def __init__(self, revision_digest: str, actor: AsyncActor) -> None:
        self.revision_digest = revision_digest
        self._actor = actor

    def commit_episode(self, episode_id: int, reward: float | None = None) -> None:
        self._actor.commit_episode(episode_id, reward=reward)


class AsyncActor:
    def __init__(self, actor_id: int, recorder: Recorder, logprob_dtype: str = "f32") -> None:
        self._actor_id = actor_id
        self._rec = recorder
        self._dtype = logprob_dtype
        self._pinned: PinnedContext | None = None
        self._open: dict[int, SequenceBuilder] = {}

    @property
    def actor_id(self) -> int:
        return self._actor_id

    def pin_revision(self, revision_digest: str) -> _PinContextManager:
        if not self._rec.ledger.has_revision(revision_digest):
            raise ProtocolViolation(f"cannot pin unpublished revision {revision_digest}")
        return _PinContextManager(self, revision_digest)

    def sample_and_record(self, log_probs: torch.Tensor, episode_id: int = 0, step: int = 0,
                          prompt_ids: Sequence[int] = (), obs: Any = None) -> dict:
        """Sample an action from log_probs under the pinned revision and record it.

        `obs` is accepted for API compatibility and ignored; a real stack records the
        prompt via `prompt_ids` on the first step of the episode.
        """
        import torch

        if self._pinned is None:
            raise ProtocolViolation(
                "sample_and_record() called without an active pin. "
                "Use 'with actor.pin_revision(digest) as ctx:' first."
            )
        builder = self._open.get(episode_id)
        if builder is None:
            builder = self._rec.sequence(self._actor_id, f"episode-{episode_id}", list(prompt_ids))
            builder.__enter__()
            self._open[episode_id] = builder
        probs = torch.softmax(log_probs.detach().float(), dim=-1)
        action = int(torch.multinomial(probs, num_samples=1).item())
        log_prob_b = float(torch.log_softmax(log_probs.detach().float(), dim=-1)[action].item())
        builder.token(self._pinned.revision_digest, action, log_prob_b, dtype=self._dtype)
        return {"action": action, "log_prob": log_prob_b, "revision_digest": self._pinned.revision_digest,
                "step": step}

    def commit_episode(self, episode_id: int, reward: float | None = None) -> None:
        builder = self._open.pop(episode_id, None)
        if builder is None:
            raise ProtocolViolation(f"episode {episode_id} has no recorded steps")
        if reward is not None:
            builder.reward(reward, dtype=self._dtype)
        builder.__exit__(None, None, None)

    def abort_episode(self, episode_id: int) -> None:
        """Discard an open episode without writing it (e.g. the engine died)."""
        self._open.pop(episode_id, None)


class _PinContextManager:
    def __init__(self, actor: AsyncActor, revision_digest: str) -> None:
        self._actor = actor
        self._revision_digest = revision_digest

    def __enter__(self) -> PinnedContext:
        ctx = PinnedContext(self._revision_digest, self._actor)
        self._actor._pinned = ctx
        return ctx

    def __exit__(self, *_) -> None:
        self._actor._pinned = None
