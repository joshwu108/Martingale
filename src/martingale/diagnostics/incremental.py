"""
diagnostics/incremental.py — the doctor during training.

`IncrementalDiagnosis.update()` folds only the scores and sequences written
since the previous call into running per-lag sums, so it costs O(new rows)
per optimizer step instead of re-reading the record. Sums are exact
Fractions and equal what `decompose()` computes over the whole record
(tests/test_incremental.py pins that). Percentiles use a bounded,
deterministic reservoir per bucket instead of every token.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from fractions import Fraction

from martingale.diagnostics.staleness import CONFIDENT_GAP, CONFIDENT_LOGPROB, DEFAULT_EPS, MAX_EXP, _fs, _pct
from martingale.record.bits import bits_to_fraction
from martingale.record.store import TokenLedger
from martingale.record.tokens import SequenceRecord

RESERVOIR_SIZE = 4096


@dataclass
class RunningBucket:
    lag: int
    n_tokens: int = 0
    n_overflow: int = 0
    n_confident_disagreements: int = 0
    sum_log_ratio: Fraction = Fraction(0)
    sum_abs_log_ratio: Fraction = Fraction(0)
    max_abs_log_ratio: Fraction = Fraction(0)
    sum_ratio: float = 0.0          # declared float boundary (exp)
    sum_ratio_sq: float = 0.0
    n_clipped: dict[str, int] = field(default_factory=dict)
    reservoir: list[float] = field(default_factory=list)
    sequences: set = field(default_factory=set)
    mixed_sequences: set = field(default_factory=set)
    size: int = RESERVOIR_SIZE

    def add(self, seq_digest: str, position: int, log_ratio: Fraction, mixed: bool, eps: tuple[float, ...],
            behavior_logprob: Fraction | None = None) -> None:
        self.n_tokens += 1
        if behavior_logprob is not None and behavior_logprob > CONFIDENT_LOGPROB and log_ratio < -CONFIDENT_GAP:
            self.n_confident_disagreements += 1
        self.sum_log_ratio += log_ratio
        self.sum_abs_log_ratio += abs(log_ratio)
        self.max_abs_log_ratio = max(self.max_abs_log_ratio, abs(log_ratio))
        x = float(log_ratio)
        if abs(x) > MAX_EXP:
            self.n_overflow += 1
            for e in eps:
                self.n_clipped[str(e)] = self.n_clipped.get(str(e), 0) + 1
        else:
            r = math.exp(x)
            self.sum_ratio += r
            self.sum_ratio_sq += r * r
            for e in eps:
                if r < 1 - e or r > 1 + e:
                    self.n_clipped[str(e)] = self.n_clipped.get(str(e), 0) + 1
            self._reservoir_add(r, seq_digest, position)
        self.sequences.add(seq_digest)
        if mixed:
            self.mixed_sequences.add(seq_digest)

    def _reservoir_add(self, r: float, seq_digest: str, position: int) -> None:
        if len(self.reservoir) < self.size:
            self.reservoir.append(r)
            return
        # deterministic Algorithm R: the slot comes from a hash of the token identity
        h = int.from_bytes(hashlib.blake2b(f"{seq_digest}:{position}".encode(), digest_size=8).digest(), "big")
        j = h % (self.n_tokens - self.n_overflow)
        if j < self.size:
            self.reservoir[j] = r

    def summary(self, eps: tuple[float, ...] = DEFAULT_EPS) -> dict:
        n = self.n_tokens
        n_finite = n - self.n_overflow
        mean = self.sum_log_ratio / n if n else Fraction(0)
        mean_abs = self.sum_abs_log_ratio / n if n else Fraction(0)
        ess = (self.sum_ratio ** 2 / self.sum_ratio_sq) if self.sum_ratio_sq > 0 and math.isfinite(self.sum_ratio_sq) else 0.0
        srt = sorted(self.reservoir)
        return {
            "lag": self.lag, "n_tokens": n, "n_sequences": len(self.sequences),
            "n_mixed_sequences": len(self.mixed_sequences), "n_overflow_tokens": self.n_overflow,
            "n_confident_disagreements": self.n_confident_disagreements,
            "mean_log_ratio": _fs(mean), "mean_abs_log_ratio": _fs(mean_abs), "max_abs_log_ratio": _fs(self.max_abs_log_ratio),
            "float_informational": {
                "mean_log_ratio": float(mean), "mean_abs_log_ratio": float(mean_abs),
                "max_abs_log_ratio": float(self.max_abs_log_ratio),
                "ratio_mean": self.sum_ratio / n_finite if n_finite else 0.0,
                "ratio_p50": _pct(srt, 0.5), "ratio_p95": _pct(srt, 0.95), "ratio_max": srt[-1] if srt else 0.0,
                "ess": ess, "ess_fraction": ess / n if n else 0.0,
                "clipped_fraction": {str(e): (self.n_clipped.get(str(e), 0) / n if n else 0.0) for e in eps},
                "percentiles_from_reservoir_of": len(self.reservoir),
            },
        }


@dataclass
class StepDiagnosis:
    """What the doctor saw in one update() call: the step's own buckets plus the cumulative ones."""
    step: int
    n_new_scores: int
    n_new_sequences: int
    by_lag: list[dict]                       # this step only
    cumulative_by_lag: list[dict]
    attribution: dict                        # this step only
    floor_mean_abs: Fraction | None          # this step's lag-0 floor
    new_weights_digests: list[str]           # weights digest of each revision new sequences were generated under
    span_fraction: float                     # fraction of new sequences spanning > 2 revisions
    n_negative_lag: int
    unscored_new_sequences: int

    def metrics(self, prefix: str = "martingale/") -> dict[str, float]:
        """Flat float metrics for a trainer log (W&B, TensorBoard). Informational floats."""
        out: dict[str, float] = {}
        a = self.attribution["float_informational"]
        if a["staleness_share"] is not None:
            out[prefix + "staleness_share"] = a["staleness_share"]
            out[prefix + "engine_share"] = a["engine_share"]
        if self.floor_mean_abs is not None:
            out[prefix + "floor_mean_abs_log_ratio"] = float(self.floor_mean_abs)
        for b in self.by_lag:
            f = b["float_informational"]
            if b["lag"] == 0:
                out[prefix + "confident_disagreements_lag0"] = float(b.get("n_confident_disagreements", 0))
            out[prefix + f"ess_fraction_lag{b['lag']}"] = f["ess_fraction"]
            out[prefix + f"mean_abs_log_ratio_lag{b['lag']}"] = f["mean_abs_log_ratio"]
            out[prefix + f"tokens_lag{b['lag']}"] = float(b["n_tokens"])
        out[prefix + "span_fraction"] = self.span_fraction
        out[prefix + "negative_lag_tokens"] = float(self.n_negative_lag)
        return out


class IncrementalDiagnosis:
    def __init__(self, ledger: TokenLedger, eps: tuple[float, ...] = DEFAULT_EPS, reservoir_size: int = RESERVOIR_SIZE) -> None:
        self._ledger = ledger
        self._eps = eps
        self._size = reservoir_size
        self._score_rowid = 0
        self._seq_rowid = 0
        self._cumulative: dict[int, RunningBucket] = {}
        self._seq_cache: dict[str, SequenceRecord] = {}
        self._step_cache: dict[str, int] = {}
        self._weights_cache: dict[str, str] = {}
        self._unscored: set[str] = set()

    def _sequence(self, digest: str) -> SequenceRecord:
        if digest not in self._seq_cache:
            self._seq_cache[digest] = self._ledger.get_sequence(digest)
        return self._seq_cache[digest]

    def _revision(self, digest: str) -> tuple[int, str]:
        if digest not in self._step_cache:
            rev = self._ledger.get_revision(digest)
            self._step_cache[digest] = rev.step
            self._weights_cache[digest] = rev.weights_digest
        return self._step_cache[digest], self._weights_cache[digest]

    def update(self, step: int) -> StepDiagnosis:
        from martingale.diagnostics.staleness import attribute

        new_seqs = self._ledger.sequences_since(self._seq_rowid)
        weights: list[str] = []
        spanning = 0
        for rid, seq in new_seqs:
            self._seq_rowid = rid
            self._seq_cache[seq.digest] = seq
            self._unscored.add(seq.digest)
            for d in seq.revision_digests:
                _s, wd = self._revision(d)
                if wd not in weights:
                    weights.append(wd)
            if len(seq.revision_digests) > 2:
                spanning += 1
        step_buckets: dict[int, RunningBucket] = {}
        n_new = 0
        n_negative = 0
        for rid, sc in self._ledger.scores_since(self._score_rowid):
            self._score_rowid = rid
            n_new += 1
            seq = self._sequence(sc.sequence_digest)
            self._unscored.discard(seq.digest)
            tok = seq.tokens[sc.position]
            b_step, _ = self._revision(tok.revision_digest)
            t_step, _ = self._revision(sc.train_revision_digest)
            lag = t_step - b_step
            if lag < 0:
                n_negative += 1
            b_lp = bits_to_fraction(tok.logprob_bits)
            lr = bits_to_fraction(sc.logprob_bits) - b_lp
            for table in (step_buckets, self._cumulative):
                table.setdefault(lag, RunningBucket(lag, size=self._size)).add(
                    seq.digest, sc.position, lr, seq.is_mixed_revision, self._eps, behavior_logprob=b_lp)
        by_lag = [step_buckets[k].summary(self._eps) for k in sorted(step_buckets)]
        floor = Fraction(step_buckets[0].summary(self._eps)["mean_abs_log_ratio"]) if 0 in step_buckets else None
        partial: dict = {"by_lag": by_lag, "lag0_floor": step_buckets[0].summary(self._eps) if 0 in step_buckets else None}
        return StepDiagnosis(
            step=step, n_new_scores=n_new, n_new_sequences=len(new_seqs), by_lag=by_lag,
            cumulative_by_lag=[self._cumulative[k].summary(self._eps) for k in sorted(self._cumulative)],
            attribution=attribute(partial), floor_mean_abs=floor, new_weights_digests=weights,
            span_fraction=(spanning / len(new_seqs)) if new_seqs else 0.0, n_negative_lag=n_negative,
            unscored_new_sequences=len(self._unscored),
        )

    def cumulative_summary(self) -> list[dict]:
        return [self._cumulative[k].summary(self._eps) for k in sorted(self._cumulative)]
