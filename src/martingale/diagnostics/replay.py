"""
diagnostics/replay.py — run the live monitor over a finished record.

The trainer's callback folds each optimizer step's new scores into an
IncrementalDiagnosis and evaluates the alarms. A record holds the same rows
in the same insertion order, so the steps can be replayed after the fact:
scores are grouped into contiguous runs by their train revision's step, and
run k is folded in as step k+1 (TRL increments global_step before the
callback fires, so the generation at step g scored at step g is reported at
step g+1, exactly as `alarms.json` from a live run numbers them).

Only rowids bound a step; nothing is re-derived with floats.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from martingale.diagnostics.alarms import Alarm, AlarmConfig, AlarmHistory, evaluate
from martingale.diagnostics.incremental import IncrementalDiagnosis, StepDiagnosis
from martingale.record.revision import LLMRevision
from martingale.record.store import TokenLedger
from martingale.record.tokens import ScoreRecord, SequenceRecord


@dataclass
class ReplayStep:
    diagnosis: StepDiagnosis
    alarms: list[Alarm] = field(default_factory=list)


class _BoundedLedger:
    """The ledger as the monitor saw it at one step: rows at or below the step's rowid limits."""

    def __init__(self, ledger: TokenLedger) -> None:
        self._ledger = ledger
        self.sequence_limit = 0
        self.score_limit = 0

    def sequences_since(self, rowid: int) -> list[tuple[int, SequenceRecord]]:
        return self._ledger.sequences_since(rowid, upto=self.sequence_limit)

    def scores_since(self, rowid: int) -> list[tuple[int, ScoreRecord]]:
        return self._ledger.scores_since(rowid, upto=self.score_limit)

    def get_sequence(self, digest: str) -> SequenceRecord:
        return self._ledger.get_sequence(digest)

    def get_revision(self, digest: str) -> LLMRevision:
        return self._ledger.get_revision(digest)


def _score_runs(ledger: TokenLedger, step_of: dict[str, int]) -> list[tuple[int, int]]:
    """(train step, last rowid) of every contiguous run of scores sharing a train step."""
    runs: list[tuple[int, int]] = []
    for rowid, digest in ledger.score_revisions():
        step = step_of[digest]
        if runs and runs[-1][0] == step:
            runs[-1] = (step, rowid)
        else:
            runs.append((step, rowid))
    return runs


def replay(ledger: TokenLedger, config: AlarmConfig | None = None) -> list[ReplayStep]:
    """Every optimizer step's StepDiagnosis and alarms, as MartingaleMonitor would have produced them."""
    cfg = config or AlarmConfig()
    step_of = {rev.digest: rev.step for rev in ledger.all_revisions()}
    seq_rows = [(rowid, min(step_of[d] for d in seq.revision_digests)) for rowid, seq in ledger.sequences_since(0)]
    bounded = _BoundedLedger(ledger)
    inc = IncrementalDiagnosis(bounded)  # type: ignore[arg-type]
    history = AlarmHistory()
    out: list[ReplayStep] = []
    for train_step, last_rowid in _score_runs(ledger, step_of):
        bounded.sequence_limit = max([r for r, g in seq_rows if g <= train_step] + [bounded.sequence_limit])
        bounded.score_limit = last_rowid
        diag = inc.update(train_step + 1)
        out.append(ReplayStep(diag, evaluate(diag, history, cfg)))
    trailing = [r for r, _g in seq_rows if r > bounded.sequence_limit]
    if trailing:   # a generation that was never scored (run cut short): report it as the monitor would
        bounded.sequence_limit = max(trailing)
        last_gen = max(g for r, g in seq_rows if r in set(trailing))
        diag = inc.update(last_gen + 1)
        out.append(ReplayStep(diag, evaluate(diag, history, cfg)))
    return out
