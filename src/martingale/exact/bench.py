"""
exact/bench.py — machine-check whether an off-policy estimator is unbiased.

    from martingale.exact import check_unbiased, defendants
    report = check_unbiased(defendants.ppo_clip(Fraction(1, 5)), n_configs=20, lags=(1, 4, 8))
    report.all_unbiased          # False
    report.certificates[0].bias  # exact rational bias vector, (state, action) -> Fraction

The expectation is an exhaustive enumeration over all trajectories in exact
rational arithmetic, so a certificate is an identity or an exact inequality,
never a statistical estimate. Scope: finite tabular rational MDPs only (see
docs/nonclaims.md).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

from martingale.estimators import on_policy_gradient
from martingale.exact.defendants import Estimator, GradientDict
from martingale.exact.random_mdp import make_mdp, make_policy, make_stale_policy
from martingale.mdp import MDP, enumerate_trajectories, feasibility_count
from martingale.policy import RationalPolicy
from martingale.rational import fraction_to_str

DEFAULT_SEED = b"martingale-exact-bench-v1"
DEFAULT_SIZES: tuple[tuple[int, int, int], ...] = ((3, 2, 3), (4, 2, 3), (3, 3, 3), (4, 2, 4))
DEFAULT_LAGS: tuple[int, ...] = (1, 2, 4)
DEFAULT_ALPHA = Fraction(1, 10)
MAX_TRAJECTORIES = 10**7


def expected_gradient(
    mdp: MDP,
    estimator: Estimator,
    target: RationalPolicy,
    behavior: RationalPolicy,
) -> GradientDict:
    """E_{τ~b}[estimator(τ, π, b)] by exhaustive enumeration, averaged over uniform start states."""
    grad: GradientDict = {(s, a): Fraction(0) for s in mdp.states for a in mdp.actions}
    for s0 in mdp.states:
        for traj in enumerate_trajectories(mdp, behavior, start_state=s0):
            contrib = estimator(traj, target, behavior)
            for key, g in contrib.items():
                grad[key] += traj["prob"] * g
    n = len(mdp.states)
    return {key: v / n for key, v in grad.items()}


@dataclass(frozen=True)
class BiasCertificate:
    config_id: int
    n_states: int
    n_actions: int
    horizon: int
    lag: int
    bias: GradientDict           # estimator expectation minus true gradient
    sq_bias: Fraction            # ||bias||² (exact, no square roots)
    sq_true_norm: Fraction       # ||∇J||²
    rel_sq_bias: Fraction | None # sq_bias / sq_true_norm; None when the true gradient is zero

    @property
    def is_unbiased(self) -> bool:
        return all(v == 0 for v in self.bias.values())

    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "n_states": self.n_states, "n_actions": self.n_actions, "horizon": self.horizon,
            "lag": self.lag,
            "is_unbiased": self.is_unbiased,
            "bias": {f"{s},{a}": fraction_to_str(v) for (s, a), v in sorted(self.bias.items())},
            "sq_bias": fraction_to_str(self.sq_bias),
            "sq_true_norm": fraction_to_str(self.sq_true_norm),
            "rel_sq_bias": None if self.rel_sq_bias is None else fraction_to_str(self.rel_sq_bias),
            "rel_sq_bias_float_informational": _informational_float(self.rel_sq_bias),
        }


@dataclass(frozen=True)
class BenchReport:
    estimator_name: str
    seed: bytes
    n_configs: int                               # MDP/policy configs drawn
    certificates: tuple[BiasCertificate, ...]    # one per (config, lag)

    @property
    def n_certificates(self) -> int:
        return len(self.certificates)

    @property
    def n_unbiased(self) -> int:
        return sum(1 for c in self.certificates if c.is_unbiased)

    @property
    def all_unbiased(self) -> bool:
        return self.n_unbiased == self.n_certificates

    def to_dict(self) -> dict:
        return {
            "estimator": self.estimator_name,
            "seed": self.seed.decode("utf-8", "replace"),
            "n_configs": self.n_configs,
            "n_certificates": self.n_certificates,
            "n_unbiased": self.n_unbiased,
            "all_unbiased": self.all_unbiased,
            "certificates": [c.to_dict() for c in self.certificates],
        }


def _informational_float(x: Fraction | None) -> float | None:
    """Float rendering for humans only; never used on a decision path."""
    if x is None:
        return None
    try:
        return float(x)
    except OverflowError:
        return None


def _config_seed(seed: bytes, i: int) -> bytes:
    return hashlib.blake2b(seed + i.to_bytes(4, "big"), digest_size=16).digest()


def _assert_feasible(sizes: Iterable[tuple[int, int, int]]) -> None:
    for n_states, n_actions, horizon in sizes:
        count = feasibility_count(n_states, n_actions, horizon)
        if count > MAX_TRAJECTORIES:
            raise ValueError(
                f"Enumeration infeasible for |S|={n_states}, |A|={n_actions}, H={horizon}: "
                f"{count} trajectories exceeds the feasibility bound {MAX_TRAJECTORIES}."
            )


def check_unbiased(
    estimator: Estimator,
    *,
    n_configs: int = 20,
    lags: Sequence[int] = DEFAULT_LAGS,
    seed: bytes = DEFAULT_SEED,
    sizes: Sequence[tuple[int, int, int]] = DEFAULT_SIZES,
    alpha: Fraction = DEFAULT_ALPHA,
    name: str | None = None,
) -> BenchReport:
    """
    Certify an estimator against the exact on-policy gradient on seeded random MDPs.

    Each config i picks sizes[i % len(sizes)], a random MDP and target policy,
    and one behavior policy per lag (lag = number of exact gradient-ascent steps
    of rate `alpha` separating behavior from target). Raises ValueError before
    doing any work if any size exceeds the enumeration feasibility bound.
    """
    if n_configs < 1:
        raise ValueError("n_configs must be >= 1")
    if any(lag < 0 for lag in lags):
        raise ValueError("lags must be non-negative")
    _assert_feasible(sizes)

    certs: list[BiasCertificate] = []
    for i in range(n_configs):
        n_states, n_actions, horizon = sizes[i % len(sizes)]
        cseed = _config_seed(seed, i)
        mdp = make_mdp(cseed, n_states, n_actions, horizon)
        target = make_policy(cseed + b"pi", mdp.states, mdp.actions)
        true_grad = on_policy_gradient(mdp, target)
        sq_true = sum((v * v for v in true_grad.values()), Fraction(0))
        for lag in lags:
            behavior = make_stale_policy(target, mdp, lag, alpha, cseed + b"lag")
            est_grad = expected_gradient(mdp, estimator, target, behavior)
            bias = {key: est_grad[key] - true_grad[key] for key in true_grad}
            sq_bias = sum((v * v for v in bias.values()), Fraction(0))
            rel = sq_bias / sq_true if sq_true != 0 else None
            certs.append(BiasCertificate(
                config_id=i, n_states=n_states, n_actions=n_actions, horizon=horizon,
                lag=lag, bias=bias, sq_bias=sq_bias, sq_true_norm=sq_true, rel_sq_bias=rel,
            ))

    label = str(name or getattr(estimator, "__name__", type(estimator).__name__))
    return BenchReport(estimator_name=label, seed=seed, n_configs=n_configs, certificates=tuple(certs))
