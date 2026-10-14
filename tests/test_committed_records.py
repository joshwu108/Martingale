"""Every committed real-run record verifies against the head it published, and reading it changes nothing.

Each workspace under benchmarks/modal/results/ holds tokens.db and head.txt (the ledger head the run printed).
The record is opened read-only, exported, and run through the independent checker anchored on that head. The
workspace's files are hashed before and after: a reader must not add a -wal/-shm sidecar or change a byte.

A workspace whose record was lost is listed in NO_RECORD with the reason; benchmarks/modal/results/README.md says
the same. The list is checked both ways, so a restored record cannot stay excused."""
import hashlib
from pathlib import Path

import pytest

from checker.verify_tokens import verify_export
from martingale.record import TokenLedger

RESULTS = Path(__file__).resolve().parent.parent / "benchmarks" / "modal" / "results"
ANCHORED = sorted(p.parent for p in RESULTS.glob("*/head.txt"))

# The t0.7 run's tokens.db was copied out before its WAL was checkpointed, so the committed file never held the run
# (0 bytes when added, then an empty schema with a different head). Its verify.json is the anchored verdict the run
# produced from its export inside the container; the record itself is gone.
NO_RECORD = {"trl_grpo_vllm_a10g_16steps_seed0_t0.7": "tokens.db lost: copied before WAL checkpoint"}
WORKSPACES = [ws for ws in ANCHORED if ws.name not in NO_RECORD]


def _snapshot(ws: Path) -> dict[str, str]:
    """{file name: sha256} for every file in the workspace, so a new sidecar shows up as well as a changed byte."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ws.iterdir()) if p.is_file()}


def test_results_directory_has_committed_records():
    assert len(WORKSPACES) >= 5, f"expected the committed real runs under {RESULTS}, found {WORKSPACES}"


def test_every_committed_record_has_its_published_head():
    """A record without head.txt would never be opened above, and could only be verified unanchored."""
    unanchored = sorted(p.parent.name for p in RESULTS.glob("*/tokens.db") if not (p.parent / "head.txt").exists())
    assert unanchored == [], f"records without head.txt: {unanchored}"


@pytest.mark.parametrize("name", sorted(NO_RECORD))
def test_excused_workspace_really_has_no_record(name: str):
    ws = RESULTS / name
    assert (ws / "head.txt").exists(), f"{name} is excused but is not a committed workspace"
    assert not (ws / "tokens.db").exists(), f"{name} has a tokens.db again: verify it and drop it from NO_RECORD"


@pytest.mark.parametrize("ws", WORKSPACES, ids=lambda p: p.name)
def test_committed_record_verifies_anchored_and_is_left_untouched(ws: Path, tmp_path: Path):
    # A hard failure, not a skip like needs_real_record: these files are committed, so a missing one is a defect.
    assert (ws / "tokens.db").exists(), f"{ws.name} has head.txt but no tokens.db"
    head = (ws / "head.txt").read_text().strip()
    before = _snapshot(ws)

    ledger = TokenLedger(ws / "tokens.db", readonly=True)
    try:
        report = verify_export(ledger.export_for_checker(tmp_path / "export"), expected_head=head)
        during = _snapshot(ws)
    finally:
        ledger.close()
    after = _snapshot(ws)

    assert during == before, f"{ws.name} changed while open: {sorted(set(during.items()) ^ set(before.items()))}"
    assert after == before, f"{ws.name} changed after close: {sorted(set(after.items()) ^ set(before.items()))}"
    assert report["errors"] == {}, f"{ws.name}: {report['errors']}"
    assert report["ok"] and report["anchored"] and report["head"] == head
    assert report["n_sequences"] > 0 and report["n_tokens"] > 0
