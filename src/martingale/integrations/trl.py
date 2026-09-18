"""
integrations/trl.py — HuggingFace TRL MartingalePPOTrainer.

A thin wrapper that adds revision publishing to TRL's PPO training loop.
Does NOT depend on trl being installed — fails gracefully if not available.

Usage (with trl installed):
    from trl import PPOTrainer
    from martingale.integrations.trl import MartingalePPOTrainer

    trainer = MartingalePPOTrainer(publisher=publisher, **trl_kwargs)
    for batch in dataloader:
        with actor.pin_revision(trainer.current_revision) as ctx:
            ...
        trainer.step(...)   # publishes new revision automatically
"""
from __future__ import annotations

from martingale.prod.revision_publisher import RevisionPublisher


class MartingalePPOTrainer:
    """
    Lightweight mixin/wrapper that adds revision publishing to a PPO training loop.

    Can be used standalone (without trl) as a revision-publishing coordinator,
    or subclassed from trl.PPOTrainer when trl is available.

    Usage without trl:
        trainer = MartingalePPOTrainer(publisher=publisher)
        trainer.on_step_end(model)   # call after each optimizer step

    Usage with trl (subclass):
        class MyTrainer(MartingalePPOTrainer, PPOTrainer):
            def step(self, *args, **kwargs):
                result = super().step(*args, **kwargs)
                self.on_step_end(self.model)
                return result
    """

    def __init__(self, publisher: RevisionPublisher, **kwargs) -> None:
        self.publisher = publisher
        self._current_revision: str | None = None
        # Pass remaining kwargs to super() if used as a mixin
        try:
            super().__init__(**kwargs)
        except TypeError:
            pass

    @property
    def current_revision(self) -> str | None:
        """The most recently published revision digest."""
        return self._current_revision

    def on_step_end(self, model) -> str:
        """
        Publish a new revision after an optimizer step.
        Call this after model weights have been updated.
        Returns the new revision digest.
        """
        rev = self.publisher.publish(model)
        self._current_revision = rev.digest
        return rev.digest

    def publish_current(self, model) -> str:
        """Alias for on_step_end — publish without implying a step occurred."""
        return self.on_step_end(model)
