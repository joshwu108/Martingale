"""
integrations/lightning.py — PyTorch Lightning MartingaleCallback.

Publishes a new revision after each optimizer step, so the ledger always
knows which checkpoint generated which trajectory.

Usage:
    from martingale.integrations.lightning import MartingaleCallback
    from martingale.prod.revision_publisher import RevisionPublisher
    from martingale.store.sqlite import SQLiteRevisionStore

    store = SQLiteRevisionStore("revisions.db")
    publisher = RevisionPublisher(store)
    trainer = pl.Trainer(callbacks=[MartingaleCallback(publisher)])
"""
from __future__ import annotations

from martingale.prod.revision_publisher import RevisionPublisher


class MartingaleCallback:
    """
    Lightning callback that publishes model checkpoints as revisions.

    Hooks used:
      - on_train_batch_start: publish at the start of each batch
      - on_before_optimizer_step: publish after weights are updated
    """

    def __init__(self, publisher: RevisionPublisher) -> None:
        self.publisher = publisher

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, **kwargs):
        """Publish current model weights as a revision."""
        self.publisher.publish(pl_module)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, **kwargs):
        """Publish after optimizer step (weights have changed)."""
        self.publisher.publish(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module, **kwargs):
        """Publish at validation time for checkpointing."""
        self.publisher.publish(pl_module)
