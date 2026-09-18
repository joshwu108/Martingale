"""
integrations/verl.py — veRL MartingaleVeRLCallback.

A thin callback adapter for veRL's actor/learner training loop that publishes
model checkpoints as content-addressed revisions after each policy update.

Does NOT depend on veRL being installed — works standalone as a revision
coordinator and can be passed to any training loop that provides a model
handle after each optimizer step.

Usage in a veRL TrainLoop:
    from martingale.integrations.verl import MartingaleVeRLCallback

    callback = MartingaleVeRLCallback(publisher=publisher)

    # After each policy update:
    callback.on_update_actor(actor_model, global_step=step)

    # Before each rollout batch (get revision to pin in actors):
    revision_digest = callback.on_rollout_start(actor_model)
"""
from __future__ import annotations

from typing import Optional

from martingale.prod.revision_publisher import RevisionPublisher


class MartingaleVeRLCallback:
    """
    veRL training loop callback for Martingale provenance tracking.

    Hooks:
      on_update_actor(model, global_step)  — call after each policy optimizer step
      on_rollout_start(model)              — call before each rollout batch

    Both hooks publish the current model weights as a revision and return the
    revision digest. Actors can use this digest to pin the current revision
    before sampling actions.
    """

    def __init__(self, publisher: RevisionPublisher) -> None:
        self.publisher = publisher
        self._current_revision: Optional[str] = None
        self._global_step: int = 0

    @property
    def current_revision(self) -> Optional[str]:
        """The digest of the most recently published revision, or None."""
        return self._current_revision

    @property
    def global_step(self) -> int:
        """The global_step from the most recent on_update_actor call."""
        return self._global_step

    def on_update_actor(self, model, global_step: int = 0) -> str:
        """
        Publish model weights as a new revision after a policy optimizer step.

        Call this immediately after `optimizer.step()` in the learner loop.

        Args:
            model: The actor model (nn.Module or state dict).
            global_step: The learner's global training step counter.

        Returns:
            The new revision digest (hex string).
        """
        self._global_step = global_step
        rev = self.publisher.publish(model)
        self._current_revision = rev.digest
        return rev.digest

    def on_rollout_start(self, model) -> str:
        """
        Publish current model weights and return the revision digest to pin.

        Call this before dispatching a rollout batch to actors so each actor
        can call `actor.pin_revision(digest)` before sampling.

        Returns:
            The revision digest actors should pin for this rollout.
        """
        rev = self.publisher.publish(model)
        self._current_revision = rev.digest
        return rev.digest
