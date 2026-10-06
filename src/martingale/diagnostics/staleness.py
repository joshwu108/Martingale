"""
diagnostics/staleness.py — staleness-versus-mismatch decomposition from the record.

For every scored token we know three things exactly: the behavior log-prob
bits (recorded at generation), the trainer's log-prob bits (recorded at the
update, under the revision it trained with), and both revisions' learner
steps. So each token has

    lag        = train_step - behavior_step            (0 = same weights)
    log_ratio  = train_logprob - behavior_logprob      (exact: both are dyadic rationals)

Tokens at lag 0 were scored by the same weights that generated them: any
nonzero log-ratio there is engine-versus-trainer mismatch, not staleness.
That bucket is the floor against which the lag >= 1 buckets are read.

Exactness: log-ratios are computed as Fractions. Everything that needs exp()
(ratios, ESS, clip fractions) is float64 at a declared boundary and labelled
as such in the report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable

from martingale.record.bits import bits_to_fraction
from martingale.record.store import TokenLedger

DEFAULT_EPS: tuple[float, ...] = (0.1, 0.2)


@dataclass(frozen=True)
class ScoredToken:
    actor_id: int
    sequence_digest: str
    position: int
    behavior_step: int
    train_step: int
    log_ratio: Fraction           # exact
    mixed_sequence: bool

    @property
    def lag(self) -> int:
        return self.train_step - self.behavior_step


def scored_tokens(ledger: TokenLedger) -> Iterable[ScoredToken]:
    """Join tokens with their scores. Tokens without a score are not yielded."""
    steps: dict[str, int] = {}

    def step_of(digest: str) -> int:
        if digest not in steps:
            steps[digest] = ledger.get_revision(digest).step
        return steps[digest]

    for seq in ledger.all_sequences():
        scores = ledger.scores_for(seq.digest)
        if not scores:
            continue
        mixed = seq.is_mixed_revision
        for s in scores:
            tok = seq.tokens[s.position]
            yield ScoredToken(
                actor_id=seq.actor_id, sequence_digest=seq.digest, position=s.position,
                behavior_step=step_of(tok.revision_digest), train_step=step_of(s.train_revision_digest),
                log_ratio=bits_to_fraction(s.logprob_bits) - bits_to_fraction(tok.logprob_bits),
                mixed_sequence=mixed,
            )


MAX_EXP = 700.0   # math.exp overflows above ~709.78; ratios beyond this are reported as overflow, not crashed on


@dataclass
class Bucket:
    lag: int
    n_tokens: int = 0
    n_overflow: int = 0
    n_sequences: int = 0
    n_mixed_sequences: int = 0
    sum_log_ratio: Fraction = Fraction(0)
    sum_abs_log_ratio: Fraction = Fraction(0)
    max_abs_log_ratio: Fraction = Fraction(0)
    log_ratios_float: list[float] = field(default_factory=list)   # declared float boundary
    _seqs: set = field(default_factory=set, repr=False)

    def add(self, t: ScoredToken) -> None:
        self.n_tokens += 1
        self.sum_log_ratio += t.log_ratio
        self.sum_abs_log_ratio += abs(t.log_ratio)
        self.max_abs_log_ratio = max(self.max_abs_log_ratio, abs(t.log_ratio))
        self.log_ratios_float.append(float(t.log_ratio))
        if t.sequence_digest not in self._seqs:
            self._seqs.add(t.sequence_digest)
            self.n_sequences += 1
            self.n_mixed_sequences += int(t.mixed_sequence)

    def summary(self, eps: tuple[float, ...] = DEFAULT_EPS) -> dict:
        n = self.n_tokens
        mean = self.sum_log_ratio / n if n else Fraction(0)
        mean_abs = self.sum_abs_log_ratio / n if n else Fraction(0)
        finite = [x for x in self.log_ratios_float if abs(x) <= MAX_EXP]
        n_overflow = n - len(finite)
        ratios = [math.exp(x) for x in finite]
        s1 = sum(ratios)
        s2 = sum(r * r for r in ratios)
        ess = (s1 * s1 / s2) if s2 > 0 and math.isfinite(s2) else 0.0
        srt = sorted(ratios)
        out = {
            "lag": self.lag, "n_tokens": n, "n_sequences": self.n_sequences,
            "n_mixed_sequences": self.n_mixed_sequences, "n_overflow_tokens": n_overflow,
            "mean_log_ratio": _fs(mean), "mean_abs_log_ratio": _fs(mean_abs),
            "max_abs_log_ratio": _fs(self.max_abs_log_ratio),
            "float_informational": {
                "mean_log_ratio": float(mean), "mean_abs_log_ratio": float(mean_abs),
                "max_abs_log_ratio": float(self.max_abs_log_ratio),
                "ratio_mean": s1 / n if n else 0.0,
                "ratio_p50": _pct(srt, 0.5), "ratio_p95": _pct(srt, 0.95), "ratio_max": srt[-1] if srt else 0.0,
                "ess": ess, "ess_fraction": ess / n if n else 0.0,
                "clipped_fraction": {str(e): ((sum(1 for r in ratios if r < 1 - e or r > 1 + e) + n_overflow) / n if n else 0.0)
                                     for e in eps},
            },
        }
        return out


def _fs(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = min(len(sorted_vals) - 1, max(0, math.ceil(q * len(sorted_vals)) - 1))   # nearest-rank
    return sorted_vals[k]


def decompose(ledger: TokenLedger, eps: tuple[float, ...] = DEFAULT_EPS) -> dict:
    """Bucket every scored token by lag and summarise; lag 0 is the engine-mismatch floor."""
    buckets: dict[int, Bucket] = {}
    n_unscored_sequences = 0
    by_train_step: dict[int, Bucket] = {}
    n_negative_lag = 0
    for t in scored_tokens(ledger):
        if t.lag < 0:
            n_negative_lag += 1
        buckets.setdefault(t.lag, Bucket(t.lag)).add(t)
        by_train_step.setdefault(t.train_step, Bucket(t.train_step)).add(t)
    n_sequences = ledger.count_sequences()
    scored_seqs = {d for b in buckets.values() for d in b._seqs}
    n_unscored_sequences = n_sequences - len(scored_seqs)
    lags = sorted(buckets)
    floor = buckets[0].summary(eps) if 0 in buckets else None
    return {
        "n_sequences": n_sequences,
        "n_unscored_sequences": n_unscored_sequences,
        "n_scored_tokens": sum(b.n_tokens for b in buckets.values()),
        "n_mixed_sequences": sum(1 for s in ledger.all_sequences() if s.is_mixed_revision),
        "lags": lags,
        "staleness_histogram": {str(lag): buckets[lag].n_tokens for lag in lags},
        "by_lag": [buckets[lag].summary(eps) for lag in lags],
        "by_train_step": [{**by_train_step[s].summary(eps), "train_step": s, "lag": None} for s in sorted(by_train_step)],
        "lag0_floor": floor,
        "n_negative_lag_tokens": n_negative_lag,
        "eps": list(eps),
        "notes": [
            "log-ratios are exact Fractions of recorded bits; fields under float_informational use float64 exp()",
            "lag 0 tokens were scored by the weights that generated them: their log-ratio is engine mismatch, not staleness",
            "a negative lag means a token was scored by an older step than generated it (resume or mislabel); inspect",
            "memory is O(scored tokens): one float per token is kept for percentiles",
        ],
    }


def attribute(d: dict) -> dict:
    """Split total |log-ratio| mass into the lag-0 engine floor and the staleness excess above it.

    Exact: uses the Fraction mean-abs per bucket. floor = lag-0 mean |log r|; for every lag > 0
    bucket, excess = max(0, mean_abs - floor) * n_tokens. Shares are of the total mean-abs mass
    over lag > 0 tokens. If there are no lag-0 tokens the floor is unknown and shares are None.
    """
    floor = None if d["lag0_floor"] is None else Fraction(d["lag0_floor"]["mean_abs_log_ratio"])
    total = Fraction(0)
    excess = Fraction(0)
    n_stale = 0
    for b in d["by_lag"]:
        if b["lag"] <= 0:
            continue
        mean_abs = Fraction(b["mean_abs_log_ratio"])
        total += mean_abs * b["n_tokens"]
        n_stale += b["n_tokens"]
        if floor is not None:
            excess += max(Fraction(0), mean_abs - floor) * b["n_tokens"]
    if floor is None or total == 0:
        share_stale = None
    else:
        share_stale = excess / total
    return {
        "floor_mean_abs_log_ratio": None if floor is None else _fs(floor),
        "stale_tokens": n_stale,
        "staleness_share": None if share_stale is None else _fs(share_stale),
        "engine_share": None if share_stale is None else _fs(1 - share_stale),
        "float_informational": {
            "staleness_share": None if share_stale is None else float(share_stale),
            "engine_share": None if share_stale is None else float(1 - share_stale),
            "floor": None if floor is None else float(floor),
        },
    }


def diagnosis(d: dict, eps: float = 0.2) -> list[str]:
    """Plain-language lines a researcher can act on. Thresholds are heuristics, stated inline."""
    a = attribute(d)
    lines: list[str] = []
    f = a["float_informational"]
    if f["staleness_share"] is None:
        lines.append("No lag-0 tokens: score at least one generation batch with the weights that produced it "
                     "(num_iterations=1 or steps_per_generation=1 for one step) to measure the engine floor.")
    else:
        lines.append(f"Of the off-policy signal in stale tokens, {100 * f['staleness_share']:.0f}% is staleness and "
                     f"{100 * f['engine_share']:.0f}% is engine-vs-trainer mismatch (lag-0 floor: mean |log r| = {f['floor']:.2e}).")
        if f["floor"] > 1e-2:
            lines.append("The engine floor is large (>1e-2 mean |log r|): the inference engine and the trainer disagree "
                         "even on fresh tokens. Check dtype (bf16 vs fp32 lm_head), sampler settings, and whether the "
                         "server's weights were actually updated; FP16 or a bit-exact engine shrinks this.")
        if f["staleness_share"] > 0.5:
            lines.append("Staleness dominates: lower num_iterations / the async level, or use a correction that is "
                         "unbiased at the sequence level (see martingale.exact).")
    for b in d["by_lag"]:
        fi = b["float_informational"]
        if b["lag"] > 0 and fi["ess_fraction"] < 0.5:
            lines.append(f"lag {b['lag']}: effective sample size is {100 * fi['ess_fraction']:.0f}% of the tokens; "
                         f"{100 * fi['clipped_fraction'][str(eps)]:.0f}% would be clipped at eps={eps}. Rollouts this "
                         "stale are mostly wasted.")
    if d["n_negative_lag_tokens"]:
        lines.append(f"{d['n_negative_lag_tokens']} tokens were scored by an OLDER step than generated them: "
                     "a resume, a mislabelled batch, or the server serving newer weights than the trainer thinks.")
    if d["n_mixed_sequences"]:
        lines.append(f"{d['n_mixed_sequences']} sequences span a weight update (in-flight sync); their tokens carry "
                     "two revisions and are bucketed by their own lag.")
    return lines
