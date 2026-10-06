"""
diagnostics/alarms.py — pure functions from a StepDiagnosis (plus history) to alarms.

Every threshold is a heuristic and lives in AlarmConfig with its rationale.
An Alarm carries the numbers and digests behind it so the message can be
checked. Two alarms are deliberately distinct:

  weights_unchanged  the TRAINER's weights digest did not change across an
                     optimizer step (broken optimizer, lr 0, frozen model) —
                     exact, from the record.
  stale_server       a lag-0 token the engine was >= 50% sure of that the
                     trainer scores > 2 nats lower. Numerics cannot do that;
                     different weights (sync not applied) or inputs can. Found
                     on the first real run: TRL 1.13 colocate served the initial
                     weights for three generations while the trainer trained,
                     and the median ratio stayed at 1 the whole time (a
                     near-deterministic task agrees on most tokens whatever
                     the weights), so a median-based proxy was wrong.
  floor_tail         the floor jumped with no confident disagreement: a few
                     low-probability tokens disagree by nats (bf16 tails).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from statistics import median

from martingale.diagnostics.incremental import StepDiagnosis


@dataclass(frozen=True)
class AlarmConfig:
    floor_window: int = 5           # trailing steps of lag-0 floor kept for the median
    floor_jump_factor: float = 3.0  # floor > factor x trailing median -> floor_jump (engine config changed?)
    stale_server_factor: float = 10.0  # floor > factor x trailing median right after a new generation -> stale_server proxy
    ess_warn: float = 0.3           # ESS/n below this in a bucket holding >= ess_min_token_share of the step's tokens
    ess_error: float = 0.1
    ess_min_token_share: float = 0.1
    span_max_fraction: float = 0.05  # fraction of new sequences spanning > 2 revisions
    unmatched_max_fraction: float = 0.01
    min_floor_tokens: int = 32      # ignore floor estimates from fewer lag-0 tokens
    median_band: float = 0.02       # informational: |ratio_p50 - 1| beyond this means the distribution shifted
    confident_min_tokens: int = 1   # confident disagreements at lag 0 that make the stale-engine verdict (exact criterion)


@dataclass(frozen=True)
class Alarm:
    kind: str
    level: str          # "warn" | "error"
    step: int
    message: str
    evidence: dict = field(default_factory=dict)


@dataclass
class AlarmHistory:
    floors: list[float] = field(default_factory=list)
    last_generation_weights: str | None = None
    last_generation_step: int | None = None

    def trailing_floor_median(self) -> float | None:
        return median(self.floors) if self.floors else None


def evaluate(diag: StepDiagnosis, history: AlarmHistory, cfg: AlarmConfig = AlarmConfig(),
             unmatched_rows: int = 0, matched_rows: int = 0) -> list[Alarm]:
    """Evaluate every alarm for one step and advance the history. Returns alarms, possibly empty."""
    alarms: list[Alarm] = []
    lag0 = next((b for b in diag.by_lag if b["lag"] == 0), None)
    floor: float = float(diag.floor_mean_abs) if diag.floor_mean_abs is not None else -1.0
    floor_ok = floor >= 0 and lag0 is not None and lag0["n_tokens"] >= cfg.min_floor_tokens
    med = history.trailing_floor_median()
    new_generation = bool(diag.new_weights_digests)

    # negative lag: exact, always an error
    if diag.n_negative_lag:
        alarms.append(Alarm("negative_lag", "error", diag.step,
                            f"{diag.n_negative_lag} tokens were scored by an older step than generated them "
                            "(resume mislabel, or the engine is ahead of the trainer)",
                            {"n_negative_lag": diag.n_negative_lag}))

    # trainer weights unchanged across an optimizer step: exact
    if new_generation and history.last_generation_weights is not None and history.last_generation_step is not None \
            and diag.step > history.last_generation_step and diag.new_weights_digests[0] == history.last_generation_weights:
        alarms.append(Alarm("weights_unchanged", "error", diag.step,
                            f"generation at step {diag.step} ran under the same trainer weights as the generation at "
                            f"step {history.last_generation_step} although optimizer steps happened in between "
                            "(frozen model, lr 0, or a broken optimizer)",
                            {"weights_digest": diag.new_weights_digests[0], "previous_step": history.last_generation_step}))

    # stale engine: a token the engine was >= 50% sure of that the trainer scores > 2 nats lower.
    # bf16 rounding cannot do that; different weights (sync did not apply) or different inputs can.
    # Found on the first real run (2026-10-06): TRL 1.13 colocate served the initial weights for
    # three generations while the trainer trained; the median ratio stayed at 1 the whole time.
    n_conf = lag0.get("n_confident_disagreements", 0) if lag0 is not None else 0
    if n_conf >= cfg.confident_min_tokens:
        alarms.append(Alarm("stale_server", "error", diag.step,
                            f"{n_conf} lag-0 tokens the engine was >=50% sure of are scored >2 nats lower by the "
                            "trainer: the engine generated from different weights than the trainer scored with "
                            "(weight sync not applied?) or from a different input. Verify with `martingale recompute` "
                            "against the trainer's checkpoint.",
                            {"n_confident_disagreements": n_conf, "floor": floor}))

    # floor jump: tail numerics or an engine config change (never alone a stale-server verdict)
    if floor_ok and med is not None and med > 0:
        ratio = floor / med
        p50 = lag0["float_informational"]["ratio_p50"] if lag0 is not None else 1.0
        if new_generation and ratio > cfg.stale_server_factor and n_conf < cfg.confident_min_tokens:
            alarms.append(Alarm("floor_tail", "warn", diag.step,
                                f"lag-0 floor jumped {ratio:.1f}x after a new generation with the median ratio at "
                                f"{p50:.4f} and no confident disagreements: low-probability tokens disagree by nats "
                                "(bf16-vs-fp32 tail effect, worse at temperature 1 and as the policy sharpens). "
                                "Consider FP16, a bit-exact engine, or masking tail ratios.",
                                {"floor": floor, "trailing_median": med, "factor": ratio, "ratio_p50": p50}))
        elif ratio > cfg.floor_jump_factor and n_conf < cfg.confident_min_tokens:
            alarms.append(Alarm("floor_jump", "warn", diag.step,
                                f"lag-0 floor is {ratio:.1f}x its trailing median ({floor:.2e} vs {med:.2e}): engine "
                                "config, dtype or server settings may have changed",
                                {"floor": floor, "trailing_median": med, "factor": ratio}))

    # ESS collapse per lag bucket
    total = sum(b["n_tokens"] for b in diag.by_lag) or 1
    for b in diag.by_lag:
        if b["lag"] <= 0 or b["n_tokens"] / total < cfg.ess_min_token_share:
            continue
        ess = b["float_informational"]["ess_fraction"]
        if ess < cfg.ess_error:
            level = "error"
        elif ess < cfg.ess_warn:
            level = "warn"
        else:
            continue
        alarms.append(Alarm("ess_collapse", level, diag.step,
                            f"lag {b['lag']}: effective sample size is {100 * ess:.0f}% of {b['n_tokens']} tokens; "
                            "rollouts this stale are mostly wasted (lower the async level / num_iterations)",
                            {"lag": b["lag"], "ess_fraction": ess, "n_tokens": b["n_tokens"]}))

    # sequences spanning > 2 revisions
    if diag.n_new_sequences and diag.span_fraction > cfg.span_max_fraction:
        alarms.append(Alarm("span", "warn", diag.step,
                            f"{100 * diag.span_fraction:.0f}% of new sequences span more than 2 weight revisions; "
                            "in-flight sync is aggressive for this completion length",
                            {"span_fraction": diag.span_fraction}))

    # unmatched loss rows
    seen = unmatched_rows + matched_rows
    if seen and unmatched_rows / seen > cfg.unmatched_max_fraction:
        alarms.append(Alarm("unmatched", "error", diag.step,
                            f"{unmatched_rows}/{seen} loss rows were not matched to a recorded sequence; the record "
                            "is incomplete (adapter lost row ids)", {"unmatched": unmatched_rows, "seen": seen}))

    # advance history
    if floor_ok:
        history.floors.append(floor)
        del history.floors[:-cfg.floor_window]
    if new_generation:
        history.last_generation_weights = diag.new_weights_digests[0]
        history.last_generation_step = diag.step
    return alarms


class MartingaleAlarmError(RuntimeError):
    def __init__(self, alarms: list[Alarm]) -> None:
        self.alarms = alarms
        super().__init__("; ".join(f"[{a.kind}@{a.step}] {a.message}" for a in alarms))


def _frac(x: Fraction | None) -> str | None:
    return None if x is None else f"{x.numerator}/{x.denominator}"
