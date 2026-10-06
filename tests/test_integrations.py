"""Tests for integrations — Lightning, veRL, JAX utilities (TRL: tests/test_trl_integration.py)."""
from fractions import Fraction

import pytest
import torch
import torch.nn as nn


class TestMartingaleCallback:
    """Tests for the PyTorch Lightning MartingaleCallback."""

    def test_callback_publishes_revision_on_train_batch_start(self, tmp_path):
        from martingale.integrations.lightning import MartingaleCallback
        from martingale.prod.revision_publisher import RevisionPublisher
        from martingale.record import Recorder

        publisher = RevisionPublisher(Recorder(tmp_path / "ws"))
        callback = MartingaleCallback(publisher=publisher)

        # Simulate a Lightning trainer calling the hook
        model = nn.Linear(4, 2)
        callback.on_train_batch_start(
            trainer=None, pl_module=model, batch=None, batch_idx=0
        )
        assert publisher.latest_digest is not None
        assert publisher.recorder.ledger.has_revision(publisher.latest_digest)

    def test_callback_publishes_on_optimizer_step(self, tmp_path):
        from martingale.integrations.lightning import MartingaleCallback
        from martingale.prod.revision_publisher import RevisionPublisher
        from martingale.record import Recorder

        publisher = RevisionPublisher(Recorder(tmp_path / "ws"))
        callback = MartingaleCallback(publisher=publisher)

        model = nn.Linear(4, 2)
        # Simulate optimizer step changing weights
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        optimizer.step()

        callback.on_before_optimizer_step(trainer=None, pl_module=model, optimizer=optimizer)
        assert publisher.latest_digest is not None

    def test_revision_changes_after_weight_update(self, tmp_path):
        from martingale.integrations.lightning import MartingaleCallback
        from martingale.prod.revision_publisher import RevisionPublisher
        from martingale.record import Recorder

        publisher = RevisionPublisher(Recorder(tmp_path / "ws"))
        callback = MartingaleCallback(publisher=publisher)
        model = nn.Linear(4, 2)

        callback.on_train_batch_start(None, model, None, 0)
        rev1 = publisher.latest_digest

        # Update weights
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()

        callback.on_train_batch_start(None, model, None, 1)
        rev2 = publisher.latest_digest

        assert rev1 != rev2


class TestJaxUtils:
    """Tests for JAX-compatible pure-function IS weight utilities."""

    def test_compute_is_weights_pure(self):
        import math

        from martingale.integrations.jax_utils import compute_is_weights_numpy
        log_pi = [-0.5, -1.0]
        log_b  = [-1.0, -1.0]
        weights = compute_is_weights_numpy(log_pi, log_b)
        assert len(weights) == 2
        assert abs(weights[0] - math.exp(-0.5 - (-1.0))) < 1e-6

    def test_clip_is_weights_pure(self):
        from martingale.integrations.jax_utils import clip_is_weights_numpy
        weights = [0.7, 1.0, 1.4]
        clipped = clip_is_weights_numpy(weights, eps=0.2)
        assert clipped[0] == pytest.approx(0.8, abs=1e-6)
        assert clipped[1] == pytest.approx(1.0, abs=1e-6)
        assert clipped[2] == pytest.approx(1.2, abs=1e-6)

    def test_ppo_loss_pure(self):
        import math

        from martingale.integrations.jax_utils import ppo_loss_numpy
        log_pi = [-0.5, -0.8]
        log_b  = [-1.0, -1.0]
        advantages = [1.0, -0.5]
        loss = ppo_loss_numpy(log_pi, log_b, advantages, clip_eps=0.2)
        assert isinstance(loss, float)


class TestMartingaleVeRLCallback:
    """Tests for the veRL MartingaleVeRLCallback integration."""

    def test_callback_importable(self):
        from martingale.integrations.verl import MartingaleVeRLCallback
        assert MartingaleVeRLCallback is not None

    def test_on_update_actor_publishes_revision(self, tmp_path):
        from martingale.integrations.verl import MartingaleVeRLCallback
        from martingale.prod.revision_publisher import RevisionPublisher
        from martingale.record import Recorder

        publisher = RevisionPublisher(Recorder(tmp_path / "ws"))
        callback = MartingaleVeRLCallback(publisher=publisher)

        model = nn.Linear(4, 2)
        digest = callback.on_update_actor(model, global_step=1)
        assert digest is not None
        assert publisher.latest_digest == digest
        assert publisher.recorder.ledger.has_revision(digest)

    def test_on_rollout_start_returns_current_revision(self, tmp_path):
        from martingale.integrations.verl import MartingaleVeRLCallback
        from martingale.prod.revision_publisher import RevisionPublisher
        from martingale.record import Recorder

        publisher = RevisionPublisher(Recorder(tmp_path / "ws"))
        callback = MartingaleVeRLCallback(publisher=publisher)

        model = nn.Linear(4, 2)
        # First publish a revision via on_update_actor
        callback.on_update_actor(model, global_step=0)
        digest = callback.on_rollout_start(model)
        assert digest == publisher.latest_digest

    def test_revision_changes_between_updates(self, tmp_path):
        from martingale.integrations.verl import MartingaleVeRLCallback
        from martingale.prod.revision_publisher import RevisionPublisher
        from martingale.record import Recorder

        publisher = RevisionPublisher(Recorder(tmp_path / "ws"))
        callback = MartingaleVeRLCallback(publisher=publisher)
        model = nn.Linear(4, 2)

        d1 = callback.on_update_actor(model, global_step=0)

        # Change model weights
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()

        d2 = callback.on_update_actor(model, global_step=1)
        assert d1 != d2

    def test_global_step_tracked(self, tmp_path):
        from martingale.integrations.verl import MartingaleVeRLCallback
        from martingale.prod.revision_publisher import RevisionPublisher
        from martingale.record import Recorder

        publisher = RevisionPublisher(Recorder(tmp_path / "ws"))
        callback = MartingaleVeRLCallback(publisher=publisher)
        model = nn.Linear(4, 2)

        callback.on_update_actor(model, global_step=42)
        assert callback.global_step == 42

    def test_no_revision_before_first_update(self, tmp_path):
        from martingale.integrations.verl import MartingaleVeRLCallback
        from martingale.prod.revision_publisher import RevisionPublisher
        from martingale.record import Recorder

        publisher = RevisionPublisher(Recorder(tmp_path / "ws"))
        callback = MartingaleVeRLCallback(publisher=publisher)
        assert callback.current_revision is None


