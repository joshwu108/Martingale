"""Tests for prod/revision_publisher.py — checkpoint hashing and publishing."""
import pytest
import torch
import torch.nn as nn
from martingale.prod.revision_publisher import RevisionPublisher, checkpoint_digest
from martingale.store.sqlite import SQLiteRevisionStore


class TestCheckpointDigest:
    def test_same_model_same_digest(self):
        model = nn.Linear(4, 2)
        d1 = checkpoint_digest(model)
        d2 = checkpoint_digest(model)
        assert d1 == d2

    def test_different_weights_different_digest(self):
        m1 = nn.Linear(4, 2)
        m2 = nn.Linear(4, 2)
        # Different random init → different digest (overwhelmingly likely)
        assert checkpoint_digest(m1) != checkpoint_digest(m2)

    def test_digest_is_hex_string(self):
        d = checkpoint_digest(nn.Linear(2, 2))
        assert isinstance(d, str)
        assert len(d) == 64

    def test_state_dict_input(self):
        model = nn.Linear(4, 2)
        d1 = checkpoint_digest(model)
        d2 = checkpoint_digest(model.state_dict())
        assert d1 == d2


class TestRevisionPublisher:
    def test_publish_registers_revision(self, tmp_path):
        store = SQLiteRevisionStore(tmp_path / "rev.db")
        publisher = RevisionPublisher(store)
        model = nn.Linear(4, 2)
        rev = publisher.publish(model)
        assert rev.digest in store

    def test_publish_idempotent(self, tmp_path):
        store = SQLiteRevisionStore(tmp_path / "rev.db")
        publisher = RevisionPublisher(store)
        model = nn.Linear(4, 2)
        r1 = publisher.publish(model)
        r2 = publisher.publish(model)
        assert r1.digest == r2.digest

    def test_revision_uniquely_identifies_checkpoint(self, tmp_path):
        """Two different models produce different revisions."""
        store = SQLiteRevisionStore(tmp_path / "rev.db")
        publisher = RevisionPublisher(store)
        m1, m2 = nn.Linear(4, 2), nn.Linear(4, 2)
        r1 = publisher.publish(m1)
        r2 = publisher.publish(m2)
        # Different model weights → different revision digest
        assert r1.digest != r2.digest

    def test_latest_digest(self, tmp_path):
        store = SQLiteRevisionStore(tmp_path / "rev.db")
        publisher = RevisionPublisher(store)
        m1, m2 = nn.Linear(4, 2), nn.Linear(4, 2)
        r1 = publisher.publish(m1)
        r2 = publisher.publish(m2)
        assert publisher.latest_digest == r2.digest
