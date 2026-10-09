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

    # a replay buffer hands the trainer a stored row again (docs/replay-provenance.md):
    rec.replay_from(seq.record.digest, actor_id=0, sequence_id="step40/replay3", draw_id="17/3",
                    content_digest=buffer_digest, is_weight=Fraction(3, 5), rescale=Fraction(1, 2))

Every token must carry a pinned revision: token() raises ProtocolViolation
without one. Nothing is written until the block exits normally; an exception
inside the block discards the sequence (and is re-raised).
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from martingale.record.bits import float_bits
from martingale.record.replay import ReplayProvenance
from martingale.record.revision import GENESIS_DIGEST, LLMRevision, SamplerConfig, tokenizer_digest, weights_digest
from martingale.record.store import TokenLedger
from martingale.record.tokens import ScoreRecord, SequenceRecord, TokenRecord, build_sequence, prompt_digest


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
            keep_prompt_ids=self._rec.keep_prompt_ids,
        )
        self.record = ledger.append_sequence(seq)


class Recorder:
    def __init__(self, workspace: Path | str, db_name: str = "tokens.db", keep_prompt_ids: bool = True) -> None:
        """`keep_prompt_ids=True` stores prompt token ids next to the digest so `martingale recompute`
        can re-score the record with a reference model; set False to keep prompts out of the record."""
        self.keep_prompt_ids = keep_prompt_ids
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
        """Record the trainer's log-prob for every recorded token of a sequence, chained.
        A sequence already scored under this train revision is not scored again (gradient
        accumulation re-runs the loss on the same rows under the same weights); returns []."""
        if self.ledger.has_scores(sequence_digest, train_revision_digest):
            return []
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

    # ---- replayed rows ----------------------------------------------------------

    def replay_from(self, origin_digest: str, *, actor_id: int, sequence_id: str, draw_id: str, content_digest: str,
                    is_weight: Fraction | float | int, rescale: Fraction | float | int = 1,
                    reward: float | None = None, dtype: str = "f32") -> SequenceRecord:
        """Register a row a replay buffer hands to the trainer, copying the behaviour log-prob bits and the
        revisions of the fresh sequence it came from. `reward` is the row's new advantage, if any."""
        try:
            origin = self.ledger.get_sequence(origin_digest)
        except KeyError:
            raise KeyError(f"replay origin sequence not in ledger: {origin_digest}") from None
        if origin.replay is not None:
            raise ValueError(f"replay origin {origin_digest[:12]} must be a fresh sequence")
        prov = ReplayProvenance.of(draw_id, content_digest, is_weight=is_weight, rescale=rescale, origin_digest=origin.digest)
        steps = [(t.revision_digest, t.token_id, t.logprob_bits, t.topk) for t in origin.tokens]
        return self._append_replayed(actor_id, sequence_id, (origin.prompt_digest, origin.prompt_len, origin.prompt_ids),
                                     steps, prov, reward, dtype)

    def replayed(self, actor_id: int, sequence_id: str, prompt_ids: Sequence[int], steps: Sequence[tuple], *,
                 draw_id: str, content_digest: str, is_weight: Fraction | float | int,
                 rescale: Fraction | float | int = 1, origin_digest: str | None = None,
                 reward: float | None = None, dtype: str = "f32") -> SequenceRecord:
        """Register a replayed row from the buffer's own copy of it.

        `steps` are (revision_digest, token_id, logprob[, topk]) with the behaviour log-prob as a float
        (encoded with `dtype`) or an already-encoded bits string; every revision must be published here.
        With `origin_digest` the row must equal that fresh sequence token for token (the ledger refuses it
        otherwise); without it the tokens are bound only to the buffer's claim (docs/nonclaims.md)."""
        if not steps:
            raise ProtocolViolation("a replayed row needs at least one token")
        encoded: list[tuple[str, int, str, tuple[tuple[int, str], ...] | None]] = []
        for st in steps:
            rev, tok, lp = st[0], st[1], st[2]
            topk = st[3] if len(st) > 3 else None
            if not rev:
                raise ProtocolViolation("replayed token without a pinned revision digest")
            if not self.ledger.has_revision(rev):
                raise ProtocolViolation(f"revision {rev} has not been published")
            tk = None if topk is None else tuple((int(t), b if isinstance(b, str) else float_bits(b, dtype)) for t, b in topk)
            encoded.append((rev, int(tok), lp if isinstance(lp, str) else float_bits(lp, dtype), tk))
        prov = ReplayProvenance.of(draw_id, content_digest, is_weight=is_weight, rescale=rescale, origin_digest=origin_digest)
        ids = tuple(int(t) for t in prompt_ids)
        return self._append_replayed(actor_id, sequence_id, (prompt_digest(ids), len(ids), ids), encoded, prov, reward, dtype)

    def _append_replayed(self, actor_id: int, sequence_id: str, prompt: tuple[str, int, tuple[int, ...] | None],
                         steps: Sequence[tuple[str, int, str, tuple[tuple[int, str], ...] | None]],
                         prov: ReplayProvenance, reward: float | None, dtype: str) -> SequenceRecord:
        ledger = self.ledger
        # index and head are read here and checked again inside append_sequence's transaction, as
        # SequenceBuilder does: a concurrent writer on this actor makes the append fail closed, never interleave
        prev_seq = ledger.last_sequence_digest(actor_id)
        tokens: list[TokenRecord] = []
        prev = prev_seq
        for pos, (rev, tok, bits, topk) in enumerate(steps):
            t = TokenRecord(revision_digest=rev, position=pos, token_id=tok, logprob_bits=bits, prev_digest=prev, topk=topk)
            tokens.append(t)
            prev = t.digest
        pdigest, plen, pids = prompt
        seq = SequenceRecord(actor_id=actor_id, sequence_index=ledger.next_sequence_index(actor_id),
                             sequence_id=sequence_id, prompt_digest=pdigest, prompt_len=plen, tokens=tuple(tokens),
                             prev_sequence_digest=prev_seq,
                             reward_bits=None if reward is None else float_bits(reward, dtype),
                             prompt_ids=pids if self.keep_prompt_ids else None, replay=prov)
        return ledger.append_sequence(seq)

    def export_for_checker(self, out_dir: Path | str) -> Path:
        return self.ledger.export_for_checker(Path(out_dir))
