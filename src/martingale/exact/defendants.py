"""
exact/defendants.py — built-in per-trajectory estimators for the test bench.

Each defendant is an `Estimator`: given one enumerated trajectory (a dict with
"prob", "steps", "return"), the target policy π and the behavior policy b, it
returns the gradient contribution it would assign to that sample, as a dict
(state, action) -> Fraction. The bench takes the exact expectation under b.

Two families, because practice and theory differ:

  seq_*    weight every step's score term by f(w) with w = ∏_t π(a_t|s_t)/b(a_t|s_t),
           the sequence-level ratio. `seq_is` is the unbiased IS-REINFORCE of T2.
  token_*  weight step t's score term by f(r_t) with r_t = π(a_t|s_t)/b(a_t|s_t)
           alone, which is what per-token PPO / TIS / MIS / CISPO implementations
           do. With a trajectory-level advantage this is biased even unclipped.

All defendants use the total return G(τ) as the advantage, matching
martingale.estimators.on_policy_gradient. All arithmetic is Fraction.

Not representable exactly, therefore absent: GSPO-style length-normalised
sequence ratios (an H-th root is irrational in general).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Callable

from martingale.policy import RationalPolicy, log_prob_gradient

GradientDict = dict[tuple[int, int], Fraction]
Estimator = Callable[[dict, RationalPolicy, RationalPolicy], GradientDict]
WeightFn = Callable[[Fraction, Fraction], Fraction]   # (ratio, G) -> weight on the score term


class SupportError(ValueError):
    """Behavior policy assigns zero probability to an action the trajectory took."""


def step_ratio(target: RationalPolicy, behavior: RationalPolicy, s: int, a: int) -> Fraction:
    b = behavior.prob(s, a)
    if b == 0:
        raise SupportError(f"behavior policy has b({a}|{s}) = 0; ratio undefined")
    return target.prob(s, a) / b


def trajectory_ratio(traj: dict, target: RationalPolicy, behavior: RationalPolicy) -> Fraction:
    """Exact sequence-level IS ratio w(τ) = π(τ)/b(τ); transition terms cancel."""
    w = Fraction(1)
    for (s, a, _sp, _r) in traj["steps"]:
        w *= step_ratio(target, behavior, s, a)
    return w


def _step_score(target: RationalPolicy, s: int, a: int, G: Fraction) -> dict[int, Fraction]:
    """G · ∇ log π(a|s). Zero (not undefined) when the weight on this step is zero."""
    return {a_i: G * g for a_i, g in log_prob_gradient(target, s, a).items()}


def _accumulate(out: GradientDict, s: int, scores: dict[int, Fraction], k: Fraction) -> None:
    if k == 0:
        return
    for a_i, g in scores.items():
        out[(s, a_i)] = out.get((s, a_i), Fraction(0)) + k * g


def _seq_level(weight: WeightFn) -> Estimator:
    def est(traj, target, behavior):
        G = traj["return"]
        k = weight(trajectory_ratio(traj, target, behavior), G)
        out: GradientDict = {}
        if k == 0:
            return out
        for (s, a, _sp, _r) in traj["steps"]:
            _accumulate(out, s, _step_score(target, s, a, G), k)
        return out
    return est


def _token_level(weight: WeightFn) -> Estimator:
    def est(traj, target, behavior):
        G = traj["return"]
        out: GradientDict = {}
        for (s, a, _sp, _r) in traj["steps"]:
            k = weight(step_ratio(target, behavior, s, a), G)
            if k == 0:
                continue
            _accumulate(out, s, _step_score(target, s, a, G), k)
        return out
    return est


# ---- weight functions -------------------------------------------------------

def _identity(r: Fraction, G: Fraction) -> Fraction:
    return r


def _truncate(cap: Fraction) -> WeightFn:
    cap = Fraction(cap)
    if cap <= 0:
        raise ValueError("cap must be > 0")
    return lambda r, G: min(r, cap)


def _mask(lo: Fraction, hi: Fraction) -> WeightFn:
    lo, hi = Fraction(lo), Fraction(hi)
    if not (0 <= lo <= hi):
        raise ValueError("need 0 <= lo <= hi")
    return lambda r, G: r if lo <= r <= hi else Fraction(0)


def _ppo(eps: Fraction) -> WeightFn:
    """∇ min(r·A, clip(r,1-ε,1+ε)·A) with gradient through r and A = G:
    r when the unclipped branch is active, 0 when the clipped (constant) branch is.
    At an exact tie the unclipped branch is taken (torch.minimum would split it)."""
    eps = Fraction(eps)
    if not (0 <= eps < 1):
        raise ValueError("need 0 <= eps < 1")
    lo, hi = 1 - eps, 1 + eps

    def w(r: Fraction, G: Fraction) -> Fraction:
        clipped_active = (G > 0 and r > hi) or (G < 0 and r < lo)
        return Fraction(0) if clipped_active else r
    return w


def _cispo(eps_lo: Fraction, eps_hi: Fraction) -> WeightFn:
    """sg(clip(r, 1-ε_lo, 1+ε_hi)): every token keeps a gradient (MiniMax-M1)."""
    eps_lo, eps_hi = Fraction(eps_lo), Fraction(eps_hi)
    if not (0 <= eps_lo < 1) or eps_hi < 0:
        raise ValueError("need 0 <= eps_lo < 1 and eps_hi >= 0")
    lo, hi = 1 - eps_lo, 1 + eps_hi
    return lambda r, G: lo if r < lo else hi if r > hi else r


def _named(est: Estimator, name: str) -> Estimator:
    est.__name__ = name
    return est


# ---- public defendants -------------------------------------------------------

unweighted = _named(_seq_level(lambda r, G: Fraction(1)), "unweighted")
"""Ignore staleness entirely: G · ∇log π. Biased whenever π ≠ b."""

seq_is = _named(_seq_level(_identity), "seq_is")
"""Sequence-level IS-REINFORCE: w(τ) · G · ∇log π. Unbiased (T2)."""

token_is = _named(_token_level(_identity), "token_is")
"""Per-token ratio only, r_t · G · ∇log π(a_t|s_t). The practical approximation; biased."""


def seq_truncated_is(cap: Fraction) -> Estimator:
    """Sequence-level TIS: min(w, cap) · G · ∇log π."""
    return _named(_seq_level(_truncate(cap)), f"seq_truncated_is(cap={Fraction(cap)})")


def token_truncated_is(cap: Fraction) -> Estimator:
    """Per-token TIS (Yao et al. 2025): min(r_t, cap) on each token's score term."""
    return _named(_token_level(_truncate(cap)), f"token_truncated_is(cap={Fraction(cap)})")


def seq_masked_is(lo: Fraction, hi: Fraction) -> Estimator:
    """Sequence-level MIS: w if lo ≤ w ≤ hi else 0."""
    return _named(_seq_level(_mask(lo, hi)), f"seq_masked_is(lo={Fraction(lo)},hi={Fraction(hi)})")


def token_masked_is(lo: Fraction, hi: Fraction) -> Estimator:
    """Per-token MIS / IcePop-style mask: r_t if lo ≤ r_t ≤ hi else 0, per token."""
    return _named(_token_level(_mask(lo, hi)), f"token_masked_is(lo={Fraction(lo)},hi={Fraction(hi)})")


def seq_ppo_clip(eps: Fraction) -> Estimator:
    """Sequence-level PPO clipped surrogate gradient on w(τ)."""
    return _named(_seq_level(_ppo(eps)), f"seq_ppo_clip(eps={Fraction(eps)})")


def token_ppo_clip(eps: Fraction) -> Estimator:
    """Per-token PPO clipped surrogate gradient on r_t, the standard implementation."""
    return _named(_token_level(_ppo(eps)), f"token_ppo_clip(eps={Fraction(eps)})")


def seq_cispo(eps_lo: Fraction, eps_hi: Fraction) -> Estimator:
    """Sequence-level CISPO on w(τ)."""
    return _named(_seq_level(_cispo(eps_lo, eps_hi)),
                  f"seq_cispo(eps_lo={Fraction(eps_lo)},eps_hi={Fraction(eps_hi)})")


def token_cispo(eps_lo: Fraction, eps_hi: Fraction) -> Estimator:
    """Per-token CISPO (MiniMax-M1): sg(clip(r_t)) · G · ∇log π(a_t|s_t)."""
    return _named(_token_level(_cispo(eps_lo, eps_hi)),
                  f"token_cispo(eps_lo={Fraction(eps_lo)},eps_hi={Fraction(eps_hi)})")


# Backwards-compatible aliases for the first draft of this module.
is_reinforce = seq_is
truncated_is = seq_truncated_is
masked_is = seq_masked_is
ppo_clip = seq_ppo_clip
cispo = seq_cispo

BUILTIN: dict[str, Callable] = {
    "unweighted": unweighted,
    "seq_is": seq_is, "token_is": token_is,
    "seq_truncated_is": seq_truncated_is, "token_truncated_is": token_truncated_is,
    "seq_masked_is": seq_masked_is, "token_masked_is": token_masked_is,
    "seq_ppo_clip": seq_ppo_clip, "token_ppo_clip": token_ppo_clip,
    "seq_cispo": seq_cispo, "token_cispo": token_cispo,
}
