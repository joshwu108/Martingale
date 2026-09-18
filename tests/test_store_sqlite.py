"""Tests for store/sqlite.py — SQLite-backed RevisionStore and Ledger."""
import pytest
from fractions import Fraction
from pathlib import Path
from martingale.store.sqlite import SQLiteRevisionStore, SQLiteLedger
from martingale.store.base import AbstractRevisionStore, AbstractLedger


class TestSQLiteRevisionStore:
    def test_implements_abstract_interface(self, tmp_path):
        store = SQLiteRevisionStore(tmp_path / "revisions.db")
        assert isinstance(store, AbstractRevisionStore)

    def test_publish_and_retrieve(self, tmp_path):
        store = SQLiteRevisionStore(tmp_path / "revisions.db")
        table = {0: {0: Fraction(1, 2), 1: Fraction(1, 2)}}
        rev = store.publish(table)
        assert rev.digest in store
        retrieved = store.get(rev.digest)
        assert retrieved.table == rev.table

    def test_idempotent_publish(self, tmp_path):
        store = SQLiteRevisionStore(tmp_path / "revisions.db")
        table = {0: {0: Fraction(3, 4), 1: Fraction(1, 4)}}
        r1 = store.publish(table)
        r2 = store.publish(table)
        assert r1.digest == r2.digest
        assert len(list(store.all_digests())) == 1

    def test_missing_digest_raises(self, tmp_path):
        store = SQLiteRevisionStore(tmp_path / "revisions.db")
        with pytest.raises(KeyError):
            store.get("0" * 64)

    def test_multiple_revisions_stored(self, tmp_path):
        store = SQLiteRevisionStore(tmp_path / "revisions.db")
        tables = [{0: {0: Fraction(i, 4), 1: Fraction(4 - i, 4)}} for i in range(1, 4)]
        revs = [store.publish(t) for t in tables]
        assert len({r.digest for r in revs}) == 3
        assert len(list(store.all_digests())) == 3

    def test_survives_reopen(self, tmp_path):
        """Revisions persist across store re-opens (durability)."""
        db = tmp_path / "revisions.db"
        table = {0: {0: Fraction(1, 3), 1: Fraction(2, 3)}}
        rev = SQLiteRevisionStore(db).publish(table)
        # Re-open fresh instance
        store2 = SQLiteRevisionStore(db)
        assert rev.digest in store2
        assert store2.get(rev.digest).table == rev.table

    def test_wal_mode_enabled(self, tmp_path):
        """WAL mode is set for concurrent reader support."""
        store = SQLiteRevisionStore(tmp_path / "revisions.db")
        import sqlite3
        con = sqlite3.connect(str(tmp_path / "revisions.db"))
        mode = con.execute("PRAGMA journal_mode").fetchone()[0]
        con.close()
        assert mode == "wal"


class TestSQLiteLedger:
    def _setup(self, tmp_path):
        store = SQLiteRevisionStore(tmp_path / "revisions.db")
        ledger = SQLiteLedger(tmp_path / "ledger.db", revision_store=store)
        rev = store.publish({0: {0: Fraction(1, 2), 1: Fraction(1, 2)}})
        return store, ledger, rev

    def test_implements_abstract_interface(self, tmp_path):
        store = SQLiteRevisionStore(tmp_path / "revisions.db")
        ledger = SQLiteLedger(tmp_path / "ledger.db", revision_store=store)
        assert isinstance(ledger, AbstractLedger)

    def test_append_and_load_trajectory(self, tmp_path):
        from martingale.draw import draw_action
        from martingale.ledger import ActionRecord, GENESIS_DIGEST
        store, ledger, rev = self._setup(tmp_path)
        probs = {0: Fraction(1, 2), 1: Fraction(1, 2)}
        draw = draw_action(probs, seed=b"test", actor_id=0, episode=0, step=0)
        rec = ActionRecord(
            revision_digest=rev.digest, state=0, action=draw.action,
            behavior_prob=probs[draw.action], draw=draw,
            env_next_state=0, env_reward=Fraction(1),
            prev_digest=GENESIS_DIGEST,
        )
        traj = ledger.append_trajectory(actor_id=0, episode_id=0, records=[rec])
        loaded = ledger.load_trajectory(actor_id=0, episode_id=0)
        assert loaded.terminal_digest == traj.terminal_digest
        assert len(loaded.action_records) == 1

    def test_survives_reopen(self, tmp_path):
        """Trajectories persist across ledger re-opens."""
        from martingale.draw import draw_action
        from martingale.ledger import ActionRecord, GENESIS_DIGEST
        store, ledger, rev = self._setup(tmp_path)
        probs = {0: Fraction(1, 2), 1: Fraction(1, 2)}
        draw = draw_action(probs, seed=b"test", actor_id=0, episode=0, step=0)
        rec = ActionRecord(
            revision_digest=rev.digest, state=0, action=draw.action,
            behavior_prob=probs[draw.action], draw=draw,
            env_next_state=0, env_reward=Fraction(1),
            prev_digest=GENESIS_DIGEST,
        )
        traj = ledger.append_trajectory(actor_id=0, episode_id=0, records=[rec])

        # Re-open
        ledger2 = SQLiteLedger(tmp_path / "ledger.db",
                               revision_store=SQLiteRevisionStore(tmp_path / "revisions.db"))
        loaded = ledger2.load_trajectory(actor_id=0, episode_id=0)
        assert loaded.terminal_digest == traj.terminal_digest

    def test_all_trajectories_iterates(self, tmp_path):
        from martingale.draw import draw_action
        from martingale.ledger import ActionRecord, GENESIS_DIGEST
        store, ledger, rev = self._setup(tmp_path)
        probs = {0: Fraction(1, 2), 1: Fraction(1, 2)}
        for ep in range(3):
            draw = draw_action(probs, seed=b"t", actor_id=0, episode=ep, step=0)
            rec = ActionRecord(
                revision_digest=rev.digest, state=0, action=draw.action,
                behavior_prob=probs[draw.action], draw=draw,
                env_next_state=0, env_reward=Fraction(0),
                prev_digest=GENESIS_DIGEST,
            )
            ledger.append_trajectory(actor_id=0, episode_id=ep, records=[rec])
        trajs = list(ledger.all_trajectories())
        assert len(trajs) == 3

    def test_checker_works_with_sqlite_ledger(self, tmp_path):
        """checker/verify.py can verify trajectories stored in SQLite."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from checker.verify import verify_ledger
        from martingale.draw import draw_action
        from martingale.ledger import ActionRecord, GENESIS_DIGEST
        store, ledger, rev = self._setup(tmp_path)
        probs = {0: Fraction(1, 2), 1: Fraction(1, 2)}
        draw = draw_action(probs, seed=b"chk", actor_id=0, episode=0, step=0)
        rec = ActionRecord(
            revision_digest=rev.digest, state=0, action=draw.action,
            behavior_prob=probs[draw.action], draw=draw,
            env_next_state=0, env_reward=Fraction(1),
            prev_digest=GENESIS_DIGEST,
        )
        ledger.append_trajectory(actor_id=0, episode_id=0, records=[rec])

        # Export to file-based format for checker compatibility
        export_dir = tmp_path / "ledger_export"
        store_dir = tmp_path / "store_export"
        ledger.export_for_checker(export_dir, store_dir)

        report = verify_ledger(export_dir, store_dir, seed=b"chk")
        assert report["failed"] == 0
