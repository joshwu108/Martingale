"""
store/sqlite.py — SQLite-backed RevisionStore and Ledger.

Production storage backend with:
  - WAL mode for concurrent readers
  - Single-writer serialization via connection-level locking
  - Full compatibility with checker/verify.py (via export_for_checker)
  - Durability: SQLite WAL + fsync on commit
"""
from __future__ import annotations

import json
import os
import sqlite3
from fractions import Fraction
from pathlib import Path
from typing import Iterator

from martingale.ledger import ActionRecord, TrajectoryRecord, GENESIS_DIGEST
from martingale.rational import fraction_to_str, to_fraction
from martingale.revision import Revision, _canonical_bytes, _fsync_file, _fsync_dir
from martingale.store.base import AbstractLedger, AbstractRevisionStore


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a WAL-mode SQLite connection with sensible defaults."""
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


class SQLiteRevisionStore(AbstractRevisionStore):
    """SQLite-backed content-addressed revision store."""

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._con = _connect(self._path)
        self._create_schema()
        self._cache: dict[str, Revision] = {}

    def _create_schema(self) -> None:
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS revisions (
                digest TEXT PRIMARY KEY,
                table_json TEXT NOT NULL,
                created_at REAL DEFAULT (unixepoch('now', 'subsec'))
            )
        """)
        self._con.commit()

    def publish(self, table: dict) -> Revision:
        rev = Revision(table)
        if rev.digest in self._cache:
            return self._cache[rev.digest]
        # Serialize: {state: {action: "num/den"}}
        table_json = json.dumps({
            str(s): {str(a): fraction_to_str(to_fraction(p)) for a, p in ap.items()}
            for s, ap in rev.table.items()
        }, sort_keys=True)
        self._con.execute(
            "INSERT OR IGNORE INTO revisions (digest, table_json) VALUES (?, ?)",
            (rev.digest, table_json),
        )
        self._con.commit()
        self._cache[rev.digest] = rev
        return rev

    def get(self, digest: str) -> Revision:
        if digest in self._cache:
            return self._cache[digest]
        row = self._con.execute(
            "SELECT table_json FROM revisions WHERE digest = ?", (digest,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Revision not found: {digest}")
        raw = json.loads(row[0])
        table = {int(s): {int(a): Fraction(p) for a, p in ap.items()}
                 for s, ap in raw.items()}
        rev = Revision(table)
        assert rev.digest == digest
        self._cache[digest] = rev
        return rev

    def __contains__(self, digest: str) -> bool:
        if digest in self._cache:
            return True
        row = self._con.execute(
            "SELECT 1 FROM revisions WHERE digest = ?", (digest,)
        ).fetchone()
        return row is not None

    def all_digests(self) -> Iterator[str]:
        for (digest,) in self._con.execute("SELECT digest FROM revisions"):
            yield digest

    def revisions_ordered(self) -> list[tuple[int, str, float]]:
        """Return [(seq, digest, created_at)] ordered by creation time."""
        rows = self._con.execute(
            "SELECT digest, created_at FROM revisions ORDER BY created_at ASC"
        ).fetchall()
        return [(i, digest, created_at) for i, (digest, created_at) in enumerate(rows)]

    def export_for_checker(self, store_dir: Path) -> None:
        """Export all revisions to the file-based format expected by checker/verify.py."""
        store_dir.mkdir(parents=True, exist_ok=True)
        for digest in self.all_digests():
            rev = self.get(digest)
            target = store_dir / f"{digest}.json"
            if not target.exists():
                payload = {
                    "digest": digest,
                    "table": {
                        str(s): {str(a): fraction_to_str(p) for a, p in ap.items()}
                        for s, ap in rev.table.items()
                    },
                }
                target.write_text(json.dumps(payload, indent=2))


class SQLiteLedger(AbstractLedger):
    """SQLite-backed hash-chained ledger."""

    def __init__(self, db_path: Path, revision_store: SQLiteRevisionStore) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._store = revision_store
        self._con = _connect(self._path)
        self._create_schema()
        self._prev_traj_digest: dict[int, str] = {}

    def _create_schema(self) -> None:
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS trajectories (
                actor_id INTEGER NOT NULL,
                episode_id INTEGER NOT NULL,
                records_json TEXT NOT NULL,
                terminal_digest TEXT NOT NULL,
                prev_trajectory_digest TEXT NOT NULL,
                created_at REAL DEFAULT (unixepoch('now', 'subsec')),
                PRIMARY KEY (actor_id, episode_id)
            )
        """)
        self._con.commit()

    @property
    def genesis_digest(self) -> str:
        return GENESIS_DIGEST

    def append_trajectory(
        self,
        actor_id: int,
        episode_id: int,
        records: list[ActionRecord],
    ) -> TrajectoryRecord:
        if not records:
            raise ValueError("Trajectory must contain at least one record.")

        prev_traj = self._prev_traj_digest.get(actor_id, GENESIS_DIGEST)
        traj = TrajectoryRecord(
            actor_id=actor_id,
            episode_id=episode_id,
            action_records=tuple(records),
            terminal_digest=records[-1].digest,
            prev_trajectory_digest=prev_traj,
        )
        records_json = json.dumps([r.to_dict() for r in records])
        self._con.execute(
            """INSERT OR REPLACE INTO trajectories
               (actor_id, episode_id, records_json, terminal_digest, prev_trajectory_digest)
               VALUES (?, ?, ?, ?, ?)""",
            (actor_id, episode_id, records_json,
             traj.terminal_digest, traj.prev_trajectory_digest),
        )
        self._con.commit()
        self._prev_traj_digest[actor_id] = traj.terminal_digest
        return traj

    def load_trajectory(self, actor_id: int, episode_id: int) -> TrajectoryRecord:
        row = self._con.execute(
            "SELECT records_json, terminal_digest, prev_trajectory_digest "
            "FROM trajectories WHERE actor_id=? AND episode_id=?",
            (actor_id, episode_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Trajectory not found: actor={actor_id}, episode={episode_id}")
        records_raw, terminal_digest, prev_traj_digest = row
        records = tuple(ActionRecord.from_dict(r) for r in json.loads(records_raw))
        return TrajectoryRecord(
            actor_id=actor_id,
            episode_id=episode_id,
            action_records=records,
            terminal_digest=terminal_digest,
            prev_trajectory_digest=prev_traj_digest,
        )

    def all_trajectories(self) -> Iterator[TrajectoryRecord]:
        rows = self._con.execute(
            "SELECT actor_id, episode_id, records_json, terminal_digest, "
            "prev_trajectory_digest FROM trajectories ORDER BY actor_id, episode_id"
        ).fetchall()
        for actor_id, episode_id, records_json, terminal_digest, prev_traj in rows:
            records = tuple(ActionRecord.from_dict(r) for r in json.loads(records_json))
            yield TrajectoryRecord(
                actor_id=actor_id,
                episode_id=episode_id,
                action_records=records,
                terminal_digest=terminal_digest,
                prev_trajectory_digest=prev_traj,
            )

    def export_for_checker(self, ledger_dir: Path, store_dir: Path) -> None:
        """Export to file-based format expected by checker/verify.py."""
        ledger_dir.mkdir(parents=True, exist_ok=True)
        self._store.export_for_checker(store_dir)
        for traj in self.all_trajectories():
            path = ledger_dir / f"actor{traj.actor_id:04d}_ep{traj.episode_id:08d}.json"
            path.write_text(json.dumps(traj.to_dict(), indent=2))
