"""
diagnostics/advise.py — `martingale doctor --advise`: what the per-lag table says about the staleness knobs.

Two suggestions, each with the numbers behind it, never applied to a run:

  sync interval   S = the number of optimizer updates one generation feeds, so its tokens
                  are scored at lags 0..S-1. The suggestion is the largest S whose lags all
                  keep ESS/n >= `ess_min` in every bucket holding over `min_token_share` of
                  the tokens. It is bounded by what ran: a record whose lags all pass says
                  nothing about lags it never saw.
  lr ceiling      Song et al. find the largest stable learning rate scales roughly as 1/S
                  and bound the staleness-induced update bias by O(S·η) (SCALING_LAW_CITATION).
                  Holding S·η at the budget that passed here, η_ceiling(S) = η·S_pass/S.
                  That is the paper's empirical law applied to this record, not a measurement,
                  and it assumes a failing lag fails from staleness bias (which a lower lr
                  shrinks), not from the engine floor or sampling noise (which it does not).
                  With no failing lag the record supports the run's own lr and nothing higher.

Exact where the inputs are: token shares, intervals and learning rates are ints and
Fractions (an lr passed as a decimal string stays exact); ESS/n is the record's declared
float (float_informational) and is compared to the float threshold as such.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction

from martingale.diagnostics.incremental import StepDiagnosis

SCALING_LAW_CITATION = ("Song et al., Scaling Laws for Collapse in Asynchronous GRPO, arXiv:2607.01083 "
                        "(largest stable lr ~ 1/S; staleness-induced update bias O(S·eta))")


def _fs(x: Fraction | None) -> str | None:
    return None if x is None else f"{x.numerator}/{x.denominator}"


def _sci(x: Fraction) -> str:
    return f"{float(x):.3g}"


def _nats(exact: str | None) -> str:
    return "unknown" if exact is None else f"{float(Fraction(exact)):.3g} nats"


@dataclass(frozen=True)
class AdviseConfig:
    # Calibrated on the five committed records (docs/advise.md): healthy buckets sit at
    # ESS/n >= 0.90, the two pathological ones at 0.35 and 0.40; 0.5 separates them with
    # margin on both sides. Five records, one model: a default, not a finding.
    ess_min: float = 0.5
    # A bucket counts when it holds strictly more than this share of the table's tokens:
    # thinner buckets are too few tokens for an ESS to mean anything (same 10% as ess_collapse).
    min_token_share: Fraction = Fraction(1, 10)

    def __post_init__(self) -> None:
        if not 0 <= self.ess_min <= 1:
            raise ValueError(f"ess_min must be in [0, 1], got {self.ess_min}")
        if not 0 <= self.min_token_share < 1:
            raise ValueError(f"min_token_share must be in [0, 1), got {self.min_token_share}")


@dataclass(frozen=True)
class LagVerdict:
    lag: int
    n_tokens: int
    token_share: Fraction
    ess_fraction: float
    max_abs_log_ratio: str | None
    qualifies: bool
    passes: bool | None          # None when the bucket does not qualify


@dataclass(frozen=True)
class Advice:
    status: str                  # no_data | withheld | engine_floor | reduce | within_observed
    config: AdviseConfig
    rows: tuple[LagVerdict, ...]
    rows_source: str             # "fresh" (fresh rows only) or "all"
    n_negative_lag_tokens: int = 0
    withheld_reason: str | None = None
    observed_interval: int | None = None    # largest lag in the table + 1
    passing_interval: int | None = None
    first_failing_lag: int | None = None
    lr: Fraction | None = None
    lr_ceiling: Fraction | None = None      # at the observed interval
    lr_failing: Fraction | None = None      # at the observed interval, from the interval that failed
    target_interval: int | None = None
    lr_ceiling_at_target: Fraction | None = None
    target_beyond_observed: bool = False
    updates_per_pass: int | None = None
    knobs: dict | None = None

    def lines(self) -> list[str]:
        whose = "the fresh tokens" if self.rows_source == "fresh" else "all tokens"
        out = [f"Suggestion, not a setting: ESS/n >= {self.config.ess_min:.3f} in every lag bucket holding over "
               f"{float(self.config.min_token_share):.0%} of {whose}."]
        if self.n_negative_lag_tokens:
            out.append(f"{self.n_negative_lag_tokens} tokens at negative lag are excluded (see negative_lag).")
        if self.status == "no_data":
            return out + ["No qualifying scored tokens at lag >= 0: nothing to advise from."]
        if self.status == "withheld":
            return out + [f"No advice: {self.withheld_reason}. The ratios at every lag compare two different "
                          "functions of the logits, so their ESS says nothing about staleness."]
        failing = next((r for r in self.rows if r.lag == self.first_failing_lag), None)
        if self.status == "engine_floor" and failing is not None:
            return out + [f"lag 0 already has ESS/n {failing.ess_fraction:.3f} over {failing.n_tokens} tokens "
                          f"(max |log r| {_nats(failing.max_abs_log_ratio)}): engine and trainer disagree before any "
                          "staleness. No async level or num_iterations fixes that; fix the floor first "
                          "(precision, sampler settings, weight sync)."]
        passing = self.passing_interval
        if passing is None:
            return out
        if self.status == "reduce" and failing is not None:
            out.append(f"lag {failing.lag} has ESS/n {failing.ess_fraction:.3f} over {failing.n_tokens} tokens "
                       f"(max |log r| {_nats(failing.max_abs_log_ratio)}): largest sync interval that passes is "
                       f"S = {passing} (lags 0..{passing - 1}); this run used S = {self.observed_interval}.")
        else:
            out.append(f"every qualifying lag passes up to lag {passing - 1}: S = {passing} is supported, and "
                       f"nothing is measured beyond the lags this run produced (S = {self.observed_interval}).")
            observed = self.observed_interval or passing
            if passing < observed:
                out.append(f"lags {passing}..{observed - 1} hold too few tokens to judge: "
                           "neither supported nor failed.")
        if self.knobs is not None:
            k = self.knobs
            if k["max_iterations"] >= 1:
                out.append(f"with {self.updates_per_pass} updates per pass over a generation: num_iterations / "
                           f"ppo_epochs <= {k['max_iterations']} (synchronous), or one pass with async level <= "
                           f"{k['max_async_level']} extra steps of lag.")
            else:
                out.append(f"S = {self.passing_interval} is shorter than one pass of {self.updates_per_pass} updates: "
                           f"use <= {k['max_updates_per_pass']} updates per generation, one iteration, no async.")
        if self.lr is not None and self.lr_ceiling is not None:
            if self.status == "reduce" and failing is not None and self.lr_failing is not None:
                out.append(f"learning rate {_sci(self.lr)}: a scaling-law heuristic, not a measured ceiling. If the "
                           f"lag-{failing.lag} failure is staleness bias, holding S·eta at the budget that passed "
                           f"({passing} x {_sci(self.lr)}) keeps S = {self.observed_interval} at lr <= "
                           f"{_sci(self.lr_ceiling)}; the budget that failed ({failing.lag + 1} x {_sci(self.lr)}) "
                           f"would give {_sci(self.lr_failing)}. A failure from the engine floor or from sampling "
                           "noise does not respond to a lower lr.")
            else:
                out.append(f"learning rate {_sci(self.lr)}: no lag failed, so the record shows no reason to change it "
                           "and no evidence that a higher one is safe.")
            if self.target_interval is not None and self.lr_ceiling_at_target is not None:
                out.append(f"at S = {self.target_interval} the scaling law at the passing budget gives lr <= "
                           f"{_sci(self.lr_ceiling_at_target)}"
                           + (": not measured in this record." if self.target_interval != passing else "."))
        out.append(f"Scaling law: {SCALING_LAW_CITATION}. An empirical law from GRPO sweeps, not a "
                   "guarantee for this run.")
        return out

    def to_dict(self) -> dict:
        return {
            "status": self.status, "rows_source": self.rows_source,
            "ess_min": self.config.ess_min, "min_token_share": _fs(self.config.min_token_share),
            "rows": [{"lag": r.lag, "n_tokens": r.n_tokens, "token_share": _fs(r.token_share),
                      "ess_fraction": r.ess_fraction, "max_abs_log_ratio": r.max_abs_log_ratio,
                      "qualifies": r.qualifies, "passes": r.passes} for r in self.rows],
            "n_negative_lag_tokens": self.n_negative_lag_tokens, "withheld_reason": self.withheld_reason,
            "observed_interval": self.observed_interval, "passing_interval": self.passing_interval,
            "first_failing_lag": self.first_failing_lag,
            "lr": _fs(self.lr), "lr_ceiling": _fs(self.lr_ceiling), "lr_failing": _fs(self.lr_failing),
            "target_interval": self.target_interval, "lr_ceiling_at_target": _fs(self.lr_ceiling_at_target),
            "target_beyond_observed": self.target_beyond_observed,
            "updates_per_pass": self.updates_per_pass, "knobs": None if self.knobs is None else dict(self.knobs),
            "citation": SCALING_LAW_CITATION, "lines": self.lines(),
            "float_informational": {k: None if v is None else float(v) for k, v in (
                ("lr", self.lr), ("lr_ceiling", self.lr_ceiling), ("lr_failing", self.lr_failing),
                ("lr_ceiling_at_target", self.lr_ceiling_at_target))},
        }


def knob_translation(interval: int, updates_per_pass: int) -> dict:
    """Largest knobs that keep every lag below `interval`, for a recipe with `updates_per_pass` optimizer
    updates per pass over one generation (TRL steps_per_generation; verl train_batch / ppo_mini_batch).

    Synchronous, K passes: lags 0..U*K-1, so K <= interval // U. One pass with async level a (a extra
    optimizer steps between generation and the first update on it): lags 0..U+a-1, so a <= interval - U.
    """
    if interval < 1 or updates_per_pass < 1:
        raise ValueError(f"interval and updates_per_pass must be >= 1, got {interval}, {updates_per_pass}")
    return {"interval": interval, "updates_per_pass": updates_per_pass,
            "max_iterations": interval // updates_per_pass,
            "max_async_level": interval - updates_per_pass if interval >= updates_per_pass else None,
            "max_updates_per_pass": min(interval, updates_per_pass)}


def _as_lr(lr: Fraction | str | int | None) -> Fraction | None:
    if lr is None:
        return None
    x = Fraction(lr)
    if x <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
    return x


def _verdicts(by_lag: Sequence[dict], cfg: AdviseConfig) -> tuple[tuple[LagVerdict, ...], int]:
    table = sorted((b for b in by_lag if b["lag"] >= 0), key=lambda b: b["lag"])
    n_negative = sum(b["n_tokens"] for b in by_lag if b["lag"] < 0)
    total = sum(b["n_tokens"] for b in table)
    rows = []
    for b in table:
        share = Fraction(b["n_tokens"], total) if total else Fraction(0)
        ess = float(b["float_informational"]["ess_fraction"])
        qualifies = share > cfg.min_token_share
        rows.append(LagVerdict(b["lag"], b["n_tokens"], share, ess, b.get("max_abs_log_ratio"), qualifies,
                               (ess >= cfg.ess_min) if qualifies else None))
    return tuple(rows), n_negative


def advise_table(by_lag: Sequence[dict], cfg: AdviseConfig = AdviseConfig(), *, lr: Fraction | str | int | None = None,
                 target_interval: int | None = None, updates_per_pass: int | None = None,
                 withheld: str | None = None, rows_source: str = "all") -> Advice:
    """Advice from one per-lag table (bucket dicts as `decompose` / `StepDiagnosis` produce them). Pure."""
    eta = _as_lr(lr)
    if target_interval is not None and target_interval < 1:
        raise ValueError(f"target_interval must be >= 1, got {target_interval}")
    rows, n_negative = _verdicts(by_lag, cfg)
    base = Advice(status="no_data", config=cfg, rows=rows, rows_source=rows_source, n_negative_lag_tokens=n_negative,
                  lr=eta, target_interval=target_interval, updates_per_pass=updates_per_pass)
    if not rows or sum(r.n_tokens for r in rows) == 0:
        return base
    if withheld is not None:
        return replace(base, status="withheld", withheld_reason=withheld)
    observed = rows[-1].lag + 1
    failing = next((r for r in rows if r.passes is False), None)
    if failing is not None and failing.lag == 0:
        return replace(base, status="engine_floor", observed_interval=observed, first_failing_lag=0)
    if failing is not None:
        status, passing = "reduce", failing.lag
    else:
        qualifying = [r.lag for r in rows if r.qualifies]
        if not qualifying:
            return replace(base, observed_interval=observed)
        status, passing = "within_observed", qualifying[-1] + 1
    return replace(
        base, status=status, observed_interval=observed, passing_interval=passing,
        first_failing_lag=None if failing is None else failing.lag,
        # nothing failed: the run's own lr is what the record supports at the interval it used
        lr_ceiling=None if eta is None else (eta * passing / observed if failing is not None else eta),
        lr_failing=None if eta is None or failing is None else eta * (failing.lag + 1) / observed,
        lr_ceiling_at_target=None if eta is None or target_interval is None else eta * passing / target_interval,
        target_beyond_observed=target_interval is not None and target_interval > observed,
        knobs=None if updates_per_pass is None else knob_translation(passing, updates_per_pass))


def advise_decomposition(d: dict, cfg: AdviseConfig = AdviseConfig(), **kwargs) -> Advice:
    """Advice from a whole-record `decompose()` dict: fresh rows when provenance is recorded (the sync knobs
    govern them; a replayed row's lag is the buffer's age), every row otherwise."""
    prov = d.get("by_provenance") or {}
    table, source = (prov["fresh"]["by_lag"] if "fresh" in prov else [], "fresh") if prov else (d["by_lag"], "all")
    withheld = (d.get("lag0_floor_withheld") or {}).get("reason")
    return advise_table(table, cfg, withheld=withheld, rows_source=source, **kwargs)


@dataclass(frozen=True)
class StepAdvice:
    step: int
    status: str
    passing_interval: int | None


@dataclass(frozen=True)
class HistoryAdvice:
    overall: Advice                          # from the last diagnosis's cumulative table
    per_step: tuple[StepAdvice, ...] = field(default_factory=tuple)
    tightest_step: int | None = None         # the step whose own table supports the smallest interval

    def lines(self) -> list[str]:
        out = self.overall.lines()
        tight = next((s for s in self.per_step if s.step == self.tightest_step), None)
        if tight is not None:
            what = "lag 0 failed" if tight.status == "engine_floor" else f"S = {tight.passing_interval}"
            out.insert(2, f"per step: the tightest step was {tight.step} ({what}); the cumulative table averages "
                          "it with the rest.")
        return out

    def to_dict(self) -> dict:
        return {**self.overall.to_dict(), "lines": self.lines(), "tightest_step": self.tightest_step,
                "per_step": [{"step": s.step, "status": s.status, "passing_interval": s.passing_interval}
                             for s in self.per_step]}


def _table(by_lag: list[dict], by_prov: dict[str, list[dict]]) -> tuple[list[dict], str]:
    """Fresh rows when provenance is recorded (an all-replayed table is then empty: no advice), else every row."""
    return (by_prov.get("fresh", []), "fresh") if by_prov else (by_lag, "all")


def advise(history: Sequence[StepDiagnosis], cfg: AdviseConfig = AdviseConfig(), **kwargs) -> HistoryAdvice:
    """Advice over a run so far: the last cumulative table, plus each step's own table so a bad step is not
    averaged away. Pure; `kwargs` are advise_table's (lr, target_interval, updates_per_pass)."""
    if not history:
        raise ValueError("advise() needs at least one StepDiagnosis")
    last = history[-1]
    table, source = _table(last.cumulative_by_lag, last.cumulative_by_provenance)
    withheld = next((d.floor_withheld for d in history if d.floor_withheld is not None), None)
    overall = advise_table(table, cfg, withheld=withheld, rows_source=source, **kwargs)
    per_step = []
    for d in history:
        t, src = _table(d.by_lag, d.by_provenance)
        a = advise_table(t, cfg, withheld=d.floor_withheld, rows_source=src)
        per_step.append(StepAdvice(d.step, a.status, a.passing_interval))
    ranked = [(0 if s.status == "engine_floor" else s.passing_interval, s.step) for s in per_step
              if s.status == "engine_floor" or s.passing_interval is not None]
    return HistoryAdvice(overall, tuple(per_step), min(ranked)[1] if ranked else None)
