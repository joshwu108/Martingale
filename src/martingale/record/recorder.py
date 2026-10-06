"""
record/recorder.py — the API an actor or trainer calls.

    rec = Recorder(workspace)                     # creates <workspace>/tokens.db
    rev = rec.publish_revision(state_dict, tokenizer_bytes, SamplerConfig(...), step=12)

    with rec.sequence(actor_id=0, sequence_id="gsm8k-412/3", prompt_ids=ids) as seq:
        for token_id, logprob in engine_stream():
            seq.token(rev.digest, token_id, logprob)         # revision may change mid-sequence
        seq.reward(1.0)
    # committed to the ledger at the end of the with-block

    rec.score(seq.record.digest, train_rev.digest, train_logprobs)   # at update time

Every token must carry a pinned revision: token() raises ProtocolViolation
without one. Nothing is written until the block exits normally; an exception
inside the block discards the sequence (and is re-raised).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from martingale.record.bits import float_bits
from martingale.record.revision import GENESIS_DIGEST, LLMRevision, SamplerConfig, tokenizer_digest, weights_digest
from martingale.record.store import TokenLedger
from martingale.record.tokens import ScoreRecord, SequenceRecord, build_sequence


class ProtocolViolation(RuntimeError):
    """A token was recorded without a pinned revision, or outside a sequence."""


class SequenceBuilder:
    def __init__(self, recorder: Recorder, actor_id: int, sequence_id: str,
                 prompt_ids: Sequence[int], engine_request_id: str | None) -> None:
        self._rec = recorder
        self._actor_id = actor_id
        self._sequence_id = sequence_id
        self._prompt_ids = list(prompt_ids)
        self._engine_request_id = engine_request_id
        self._steps: list[tuple[str, int, str, tuple[tuple[int, str], ...] | None]] = []
        self._reward_bits: str | None = None
        self.record: SequenceRecord | None = None

    def token(self, revision_digest: str | None, token_id: int, logprob: float, *,
              dtype: str = "f32", topk: Iterable[tuple[int, float]] | None = None) -> None:
        if not revision_digest:
            raise ProtocolViolation("token() called without a pinned revision digest")
        if not self._rec.ledger.has_revision(revision_digest):
            raise ProtocolViolation(f"revision {revision_digest} has not been published")
        tk = None if topk is None else tuple((int(t), float_bits(lp, dtype)) for t, lp in topk)
        self._steps.append((revision_digest, int(token_id), float_bits(logprob, dtype), tk))

    def reward(self, value: float, dtype: str = "f32") -> None:
        self._reward_bits = float_bits(value, dtype)

    def __enter__(self) -> SequenceBuilder:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            return  # discard; re-raise
        if not self._steps:
            raise ProtocolViolation("sequence closed without any token")
        ledger = self._rec.ledger
        seq = build_sequence(
            actor_id=self._actor_id, sequence_index=ledger.next_sequence_index(self._actor_id),
            sequence_id=self._sequence_id, prompt_ids=self._prompt_ids, steps=self._steps,
            prev_sequence_digest=ledger.last_sequence_digest(self._actor_id),
            reward_bits=self._reward_bits, engine_request_id=self._engine_request_id,
        )
        self.record = ledger.append_sequence(seq)


class Recorder:
    def __init__(self, workspace: Path | str, db_name: str = "tokens.db") -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.ledger = TokenLedger(self.workspace / db_name)
        self._latest: LLMRevision | None = None

    @property
    def latest_revision(self) -> LLMRevision | None:
        return self._latest

    def publish_revision(self, weights: Mapping[str, Any] | str, tokenizer: bytes | str,
                         sampler: SamplerConfig, step: int, *, parent_digest: str | None = None,
                         weights_uri: str | None = None) -> LLMRevision:
        """`weights` is a state dict (digested here) or a precomputed weights digest;
        `tokenizer` is tokenizer.json bytes or a precomputed digest. Parent defaults to the
        previously published revision."""
        wd = weights if isinstance(weights, str) else weights_digest(weights)
        td = tokenizer if isinstance(tokenizer, str) else tokenizer_digest(tokenizer)
        parent = parent_digest if parent_digest is not None else (
            self._latest.digest if self._latest is not None else GENESIS_DIGEST)
        rev = LLMRevision(weights_digest=wd, tokenizer_digest=td, sampler=sampler, step=step,
                          parent_digest=parent, weights_uri=weights_uri)
        self.ledger.publish_revision(rev)
        self._latest = rev
        return rev

    def sequence(self, actor_id: int, sequence_id: str, prompt_ids: Sequence[int],
                 engine_request_id: str | None = None) -> SequenceBuilder:
        return SequenceBuilder(self, actor_id, sequence_id, prompt_ids, engine_request_id)

    def score(self, sequence_digest: str, train_revision_digest: str,
              logprobs: Sequence[float], dtype: str = "f32") -> list[ScoreRecord]:
        """Record the trainer's log-prob for every recorded token of a sequence, chained."""
        prev = self.ledger.last_score_digest(sequence_digest)
        records: list[ScoreRecord] = []
        for pos, lp in enumerate(logprobs):
            r = ScoreRecord(sequence_digest=sequence_digest, position=pos,
                            train_revision_digest=train_revision_digest,
                            logprob_bits=float_bits(lp, dtype), prev_digest=prev)
            records.append(r)
            prev = r.digest
        self.ledger.append_scores(records)
        return records

    def export_for_checker(self, out_dir: Path | str) -> Path:
        return self.ledger.export_for_checker(Path(out_dir))
