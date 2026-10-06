"""
record/store.py — TokenLedger: SQLite store for LLM revisions, sequences, scores.

One file (`tokens.db`), WAL mode, synchronous=FULL and fullfsync=ON (the
latter makes SQLite use F_FULLFSYNC on Darwin; ignored elsewhere). A sequence
is written in one transaction after its last token, so a SIGKILL mid-sequence
loses that sequence and nothing else. export_for_checker() writes the
file-based form checker/verify_tokens.py reads.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterator

from martingale.record.bits import GENESIS_DIGEST, digest_json
from martingale.record.revision import LLMRevision
from martingale.record.tokens import ScoreRecord, SequenceRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_revisions (
    digest TEXT PRIMARY KEY,
    manifest_json TEXT NOT NULL,
    weights_uri TEXT,
    step INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sequences (
    actor_id INTEGER NOT NULL,
    sequence_index INTEGER NOT NULL,
    digest TEXT NOT NULL UNIQUE,
    record_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (actor_id, sequence_index)
);
CREATE TABLE IF NOT EXISTS scores (
    sequence_digest TEXT NOT NULL,
    position INTEGER NOT NULL,
    train_revision_digest TEXT NOT NULL,
    record_json TEXT NOT NULL,
    PRIMARY KEY (sequence_digest, position, train_revision_digest)
);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("PRAGMA fullfsync=ON")
    return con


class TokenLedger:
    """Every mutation runs inside one BEGIN IMMEDIATE transaction (check-then-insert
    included) under a process-local lock; a failure rolls back, so a sequence or a score
    batch is written entirely or not at all, and two writers on one file serialise."""

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._con = _connect(self._path)
        self._lock = threading.RLock()
        self._con.executescript(_SCHEMA)

    @contextlib.contextmanager
    def _tx(self, immediate: bool = True):
        with self._lock:
            self._con.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
            self._con.execute("COMMIT")

    # ---- revisions ------------------------------------------------------------

    def publish_revision(self, rev: LLMRevision) -> LLMRevision:
        """Idempotent: the same manifest yields the same row."""
        with self._tx():
            self._con.execute(
                "INSERT OR IGNORE INTO llm_revisions (digest, manifest_json, weights_uri, step, created_at) "
                "VALUES (?,?,?,?,?)",
                (rev.digest, json.dumps(rev.to_dict(), sort_keys=True), rev.weights_uri, rev.step, time.time()),
            )
        return rev

    def get_revision(self, digest: str) -> LLMRevision:
        row = self._con.execute("SELECT manifest_json FROM llm_revisions WHERE digest=?", (digest,)).fetchone()
        if row is None:
            raise KeyError(f"LLM revision not found: {digest}")
        return LLMRevision.from_dict(json.loads(row[0]))

    def has_revision(self, digest: str) -> bool:
        return self._con.execute("SELECT 1 FROM llm_revisions WHERE digest=?", (digest,)).fetchone() is not None

    def all_revisions(self) -> Iterator[LLMRevision]:
        for (m,) in self._con.execute("SELECT manifest_json FROM llm_revisions ORDER BY step, created_at"):
            yield LLMRevision.from_dict(json.loads(m))

    def max_step(self) -> int | None:
        return self._con.execute("SELECT MAX(step) FROM llm_revisions").fetchone()[0]

    # ---- sequences ------------------------------------------------------------

    def next_sequence_index(self, actor_id: int) -> int:
        row = self._con.execute("SELECT MAX(sequence_index) FROM sequences WHERE actor_id=?", (actor_id,)).fetchone()
        return 0 if row[0] is None else row[0] + 1

    def last_sequence_digest(self, actor_id: int) -> str:
        row = self._con.execute(
            "SELECT digest FROM sequences WHERE actor_id=? ORDER BY sequence_index DESC LIMIT 1", (actor_id,)
        ).fetchone()
        return GENESIS_DIGEST if row is None else row[0]

    def append_sequence(self, seq: SequenceRecord) -> SequenceRecord:
        """Fail closed: refuses an unknown revision, a gap in the per-actor index, or a broken chain.
        The checks and the insert are one transaction."""
        with self._tx():
            for d in seq.revision_digests:
                if not self.has_revision(d):
                    raise KeyError(f"sequence references unpublished revision {d}")
            expected_index = self.next_sequence_index(seq.actor_id)
            if seq.sequence_index != expected_index:
                raise ValueError(f"actor {seq.actor_id}: expected sequence_index {expected_index}, got {seq.sequence_index}")
            expected_prev = self.last_sequence_digest(seq.actor_id)
            if seq.prev_sequence_digest != expected_prev:
                raise ValueError(f"actor {seq.actor_id}: prev_sequence_digest does not match the ledger head")
            self._con.execute(
                "INSERT INTO sequences (actor_id, sequence_index, digest, record_json, created_at) VALUES (?,?,?,?,?)",
                (seq.actor_id, seq.sequence_index, seq.digest, json.dumps(seq.to_dict(), sort_keys=True), time.time()),
            )
        return seq

    def get_sequence(self, digest: str) -> SequenceRecord:
        row = self._con.execute("SELECT record_json FROM sequences WHERE digest=?", (digest,)).fetchone()
        if row is None:
            raise KeyError(f"sequence not found: {digest}")
        return SequenceRecord.from_dict(json.loads(row[0]))

    def all_sequences(self) -> Iterator[SequenceRecord]:
        for (r,) in self._con.execute("SELECT record_json FROM sequences ORDER BY actor_id, sequence_index"):
            yield SequenceRecord.from_dict(json.loads(r))

    def count_sequences(self) -> int:
        return self._con.execute("SELECT COUNT(*) FROM sequences").fetchone()[0]

    # ---- scores ---------------------------------------------------------------

    def last_score_digest(self, sequence_digest: str) -> str:
        row = self._con.execute(
            "SELECT record_json FROM scores WHERE sequence_digest=? ORDER BY rowid DESC LIMIT 1", (sequence_digest,)
        ).fetchone()
        return sequence_digest if row is None else json.loads(row[0])["digest"]

    def append_scores(self, records: list[ScoreRecord]) -> None:
        """Append a chained batch of scores for one sequence in one transaction."""
        if not records:
            return
        seq_digest = records[0].sequence_digest
        if any(r.sequence_digest != seq_digest for r in records):
            raise ValueError("a score batch must belong to one sequence")
        with self._tx():
            seq = self.get_sequence(seq_digest)
            prev = self.last_score_digest(seq_digest)
            for r in records:
                if not self.has_revision(r.train_revision_digest):
                    raise KeyError(f"score references unpublished revision {r.train_revision_digest}")
                if not (0 <= r.position < len(seq.tokens)):
                    raise ValueError(f"score position {r.position} outside sequence of length {len(seq.tokens)}")
                if r.prev_digest != prev:
                    raise ValueError("score chain does not continue from the ledger head")
                prev = r.digest
            self._con.executemany(
                "INSERT INTO scores (sequence_digest, position, train_revision_digest, record_json) VALUES (?,?,?,?)",
                [(r.sequence_digest, r.position, r.train_revision_digest, json.dumps(r.to_dict(), sort_keys=True))
                 for r in records],
            )

    def scores_for(self, sequence_digest: str) -> list[ScoreRecord]:
        rows = self._con.execute(
            "SELECT record_json FROM scores WHERE sequence_digest=? ORDER BY rowid", (sequence_digest,)
        ).fetchall()
        return [ScoreRecord.from_dict(json.loads(r)) for (r,) in rows]

    # ---- head -----------------------------------------------------------------

    def head(self) -> str:
        """One digest covering every chain head and the revision set (design.md 7.6).

        Publish it out of band (run log, W&B field) and hand it to the checker as
        `expected_head`; then any tampering anywhere, re-signed or not, moves it.
        """
        with self._tx(immediate=False):   # one read snapshot
            sequence_heads = {row[0]: row[1] for row in self._con.execute(
                "SELECT actor_id, digest FROM sequences s WHERE sequence_index = "
                "(SELECT MAX(sequence_index) FROM sequences WHERE actor_id = s.actor_id)")}
            score_heads = {d: json.loads(rj)["digest"] for d, rj in self._con.execute(
                "SELECT sequence_digest, record_json FROM scores s WHERE rowid = "
                "(SELECT MAX(rowid) FROM scores WHERE sequence_digest = s.sequence_digest)")}
            revisions = [d for (d,) in self._con.execute("SELECT digest FROM llm_revisions")]
        return ledger_head(sequence_heads, score_heads, revisions)

    # ---- export ---------------------------------------------------------------

    def export_for_checker(self, out_dir: Path) -> Path:
        """Write revisions/<digest>.json and sequences/actorNNNN_seqNNNNNNNN.json (with scores).

        Written to a temporary sibling directory, fsynced, then renamed into place. A
        previous export at `out_dir` is replaced; anything else there is refused.
        """
        out = Path(out_dir)
        if out.exists():
            if set(p.name for p in out.iterdir()) - {"revisions", "sequences"}:
                raise FileExistsError(f"{out} exists and is not a previous export; refusing to overwrite")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=out.name + ".", dir=out.parent))
        (tmp / "revisions").mkdir()
        (tmp / "sequences").mkdir()
        with self._tx(immediate=False):
            for rev in self.all_revisions():
                _write_fsync(tmp / "revisions" / f"{rev.digest}.json", json.dumps(rev.to_dict(), indent=1, sort_keys=True))
            scores_by_seq: dict[str, list[dict]] = {}
            for d, rj in self._con.execute("SELECT sequence_digest, record_json FROM scores ORDER BY rowid"):
                scores_by_seq.setdefault(d, []).append(json.loads(rj))
            for seq in self.all_sequences():
                payload = seq.to_dict()
                payload["scores"] = scores_by_seq.get(seq.digest, [])
                name = f"actor{seq.actor_id:04d}_seq{seq.sequence_index:08d}.json"
                _write_fsync(tmp / "sequences" / name, json.dumps(payload, indent=1, sort_keys=True))
        for sub in ("revisions", "sequences"):
            _fsync_dir(tmp / sub)
        if out.exists():
            shutil.rmtree(out)
        os.rename(tmp, out)
        _fsync_dir(out.parent)
        return out

    def close(self) -> None:
        self._con.close()


def _write_fsync(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        pass


def ledger_head(sequence_heads: dict[int, str], score_heads: dict[str, str], revision_digests: list[str]) -> str:
    """Canonical head digest; checker/verify_tokens.py recomputes the same from an export."""
    return digest_json({
        "sequence_heads": {str(a): h for a, h in sorted(sequence_heads.items())},
        "score_heads": dict(sorted(score_heads.items())),
        "revisions": sorted(revision_digests),
    })
