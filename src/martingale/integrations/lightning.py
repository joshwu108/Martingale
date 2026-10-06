"""
integrations/lightning.py — PyTorch Lightning MartingaleCallback.

Publishes a new revision after each optimizer step, so the ledger always
knows which checkpoint generated which trajectory.

Usage:
    from martingale.integrations.lightning import MartingaleCallback
    from martingale.prod.revision_publisher import RevisionPublisher
    from martingale.record import Recorder

    publisher = RevisionPublisher(Recorder("./martingale_ws"))
    trainer = pl.Trainer(callbacks=[MartingaleCallback(publisher)])
"""
from __future__ import annotations

from martingale.prod.revision_publisher import RevisionPublisher


class MartingaleCallback:
    """
    Lightning callback that publishes model checkpoints as revisions.

    Hooks used:
      - on_train_batch_start: publish the weights the batch will be generated under
      - on_train_batch_end: publish the updated weights (after optimizer.step())
    Publishing is idempotent per checkpoint, so unchanged weights do not mint revisions.
    """

    def __init__(self, publisher: RevisionPublisher) -> None:
        self.publisher = publisher

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, **kwargs):
        """Publish current model weights as a revision."""
        self.publisher.publish(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs=None, batch=None, batch_idx=0, **kwargs):
        """Publish after the optimizer step (weights have changed)."""
        self.publisher.publish(pl_module)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, **kwargs):
        """Kept for compatibility: weights have not changed yet, so this is a no-op publish."""
        self.publisher.publish(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module, **kwargs):
        """Publish at validation time for checkpointing."""
        self.publisher.publish(pl_module)
