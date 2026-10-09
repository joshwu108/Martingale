"""A record opened for reading is left exactly as it was found: no -wal/-shm sidecar next to it, no byte
of tokens.db changed, no mutation accepted. A record a writer still holds open is read through its WAL.

Background: tokens.db is a WAL-mode file. Opening it read-write creates -wal/-shm and a clean close removes
them; a plain mode=ro connection creates them too but cannot checkpoint, so it leaves them behind. That is
how the committed results grew stray sidecars. `TokenLedger(path, readonly=True)` opens a quiescent record
immutable, so nothing is created at all."""
import hashlib
import sqlite3
from pathlib import Path

import pytest

from martingale.record import Recorder, SamplerConfig, TokenLedger
from tests.conftest import REAL_RECORD, needs_real_record

S = SamplerConfig(temperature=1.0)


def _sidecars(db: Path) -> list[str]:
    return sorted(p.name for p in db.parent.glob(db.name + "-*"))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _written_record(ws: Path, n: int = 3) -> Path:
    """A small record written by a Recorder and closed: tokens.db alone, header in WAL mode, no sidecars."""
    rec = Recorder(ws)
    rev = rec.publish_revision("a" * 64, "b" * 64, S, step=0)
    for i in range(n):
        with rec.sequence(0, f"s{i}", [1, 2, i]) as seq:
            seq.token(rev.digest, 10 + i, -0.5)
            seq.token(rev.digest, 20 + i, -0.25)
    rec.score(seq.record.digest, rev.digest, [-0.5, -0.25])
    rec.ledger.close()
    db = ws / "tokens.db"
    assert _sidecars(db) == []
    return db


def _read_everything(ledger: TokenLedger, tmp: Path) -> None:
    """Every read path a reader uses, including the two that create something on a writable ledger."""
    ledger.count_sequences(), ledger.count_scores(), ledger.count_revisions(), ledger.head()
    list(ledger.all_sequences()), list(ledger.all_revisions())
    seq = next(iter(ledger.all_sequences()))
    ledger.scores_for(seq.digest)
    ledger.find_fresh_sequences(seq.prompt_digest)       # creates an expression index on a writable ledger
    assert ledger.row_digest(0, 0) is None               # the row_ids table is created on first bind only
    assert ledger.count_replayed_sequences() == 0
    ledger.export_for_checker(tmp / "export")            # writes the export, never the record


def test_reading_a_closed_record_leaves_no_sidecar_and_no_changed_byte(tmp_path):
    db = _written_record(tmp_path / "ws")
    before = _sha(db)
    ledger = TokenLedger(db, readonly=True)
    _read_everything(ledger, tmp_path)
    assert _sidecars(db) == []                           # not even while it is open
    ledger.close()
    assert _sidecars(db) == [] and _sha(db) == before


def test_readonly_refuses_every_mutation(tmp_path):
    db = _written_record(tmp_path / "ws")
    before = _sha(db)
    ledger = TokenLedger(db, readonly=True)
    rev = next(iter(ledger.all_revisions()))
    with pytest.raises(sqlite3.OperationalError, match="read-only"):
        ledger.publish_revision(rev)
    with pytest.raises(sqlite3.OperationalError, match="read-only"):
        ledger.bind_row_ids([(0, 0, "d" * 64)])
    with pytest.raises(sqlite3.OperationalError, match="read-only"):
        with ledger._tx():
            pass
    ledger.checkpoint_wal()                              # a no-op for a reader, not an error
    ledger.close()
    assert _sidecars(db) == [] and _sha(db) == before


def test_readonly_open_needs_an_existing_record(tmp_path):
    with pytest.raises(FileNotFoundError):
        TokenLedger(tmp_path / "missing" / "tokens.db", readonly=True)
    assert not (tmp_path / "missing").exists()           # nothing scaffolded


def test_reader_over_a_record_a_writer_holds_open_sees_its_writes(tmp_path):
    """The inspector over a live run: the writer's -wal exists, so the reader goes through the WAL (not
    immutable) and sees rows committed after it opened; the writer removes the sidecars when it closes."""
    rec = Recorder(tmp_path / "ws")
    rev = rec.publish_revision("a" * 64, "b" * 64, S, step=0)
    db = tmp_path / "ws" / "tokens.db"
    assert "tokens.db-wal" in _sidecars(db)
    reader = TokenLedger(db, readonly=True)
    assert reader.count_sequences() == 0
    with rec.sequence(0, "live", [1, 2, 3]) as seq:
        seq.token(rev.digest, 7, -0.5)
    assert reader.count_sequences() == 1 and reader.get_sequence(seq.record.digest).sequence_id == "live"
    reader.close()
    rec.ledger.close()
    assert _sidecars(db) == []


@needs_real_record
def test_reading_the_committed_record_leaves_it_untouched(tmp_path):
    db = REAL_RECORD / "tokens.db"
    assert _sidecars(db) == [], "stale sidecars next to the committed record before the test ran"
    before = _sha(db)
    ledger = TokenLedger(db, readonly=True)
    _read_everything(ledger, tmp_path)
    assert ledger.head() == (REAL_RECORD / "head.txt").read_text().strip()
    assert _sidecars(db) == []
    ledger.close()
    assert _sidecars(db) == [] and _sha(db) == before


@needs_real_record
def test_the_inspector_opens_the_committed_record_read_only():
    from martingale.server import create_app
    app = create_app(REAL_RECORD)
    assert app.state.inspector.ledger.readonly is True
    assert _sidecars(REAL_RECORD / "tokens.db") == []
