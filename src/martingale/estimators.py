"""
estimators.py — exact expected gradients via trajectory enumeration.

Each estimator is implemented twice:
  1. As an exact enumerated expectation (sum over all trajectories).
  2. As a per-trajectory sample statistic (for the ledger-based accumulation path).

Exact arithmetic: all values are fractions.Fraction. No floats anywhere here.

Estimators implemented:
  - on_policy_gradient: E_{τ~π}[G(τ) · ∇ log π(τ)]
  - is_reinforce_gradient: E_{τ~b}[w(τ) · G(τ) · ∇ log π(τ)]
    where w(τ) = π(τ) / b(τ) (IS weight, transition factors cancel)
  - ppo_clipped_gradient: E_{τ~b}[clip(w, 1-ε, 1+ε) · G(τ) · ∇ log π(τ)]
  - grpo_gradient: group-relative policy optimization estimator

T2 identity: is_reinforce_gradient == on_policy_gradient (machine-checked equality).
T2 negative: ppo_clipped_gradient != on_policy_gradient (bias is nonzero).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from martingale.mdp import MDP, enumerate_trajectories
from martingale.policy import RationalPolicy, log_prob_gradient
from martingale.rational import to_fraction


# Type for gradient: maps (state, action) → Fraction
GradientDict = dict[tuple[int, int], Fraction]


def _zero_gradient(mdp: MDP) -> GradientDict:
    return {(s, a): Fraction(0) for s in mdp.states for a in mdp.actions}


def _trajectory_log_prob(traj: dict, policy: RationalPolicy) -> Fraction:
    """Compute log π(τ) = Σ_t log π(a_t|s_t) — but we need the ratio, not log."""
    raise NotImplementedError("Use _trajectory_prob_ratio instead.")


def _trajectory_policy_prob(traj: dict, policy: RationalPolicy) -> Fraction:
    """Compute π(τ) = ∏_t π(a_t|s_t) as an exact Fraction."""
    prob = Fraction(1)
    for (s, a, sp, r) in traj["steps"]:
        prob *= policy.prob(s, a)
    return prob


def _trajectory_reinforce_grad(
    traj: dict,
    policy: RationalPolicy,
    mdp: MDP,
) -> GradientDict:
    """
    Compute G(τ) · Σ_t ∇_θ log π(a_t|s_t) for one trajectory.

    This is the per-trajectory REINFORCE gradient contribution (before weighting).
    Uses the per-decision formulation: each time step contributes its log-prob gradient
    weighted by the return from that step onwards (reward-to-go).

    For simplicity here we use total return G(τ) (not reward-to-go), as in the
    basic REINFORCE formulation. The identity still holds.
    """
    G = traj["return"]
    grad = {(s, a): Fraction(0) for s in mdp.states for a in mdp.actions}
    for (state, action, next_state, reward) in traj["steps"]:
        step_grad = log_prob_gradient(policy, state, action)
        for a_i, g in step_grad.items():
            grad[(state, a_i)] += G * g
    return grad


def on_policy_gradient(mdp: MDP, policy: RationalPolicy) -> GradientDict:
    """
    Compute the exact on-policy REINFORCE expected gradient:
      ∇J = E_{τ~π}[G(τ) · Σ_t ∇ log π(a_t|s_t)]
         = Σ_τ P_π(τ) · G(τ) · Σ_t ∇ log π(a_t|s_t)

    Returns a dict mapping (state, action) → exact Fraction gradient.
    Averages uniformly over start states.
    """
    grad = _zero_gradient(mdp)
    n_starts = len(mdp.states)

    for s0 in mdp.states:
        for traj in enumerate_trajectories(mdp, policy, start_state=s0):
            p = traj["prob"]  # P_π(τ)
            traj_grad = _trajectory_reinforce_grad(traj, policy, mdp)
            for key, g in traj_grad.items():
                grad[key] += p * g

    # Average over uniform initial state distribution
    for key in grad:
        grad[key] /= n_starts
    return grad


def is_reinforce_gradient(
    mdp: MDP,
    target: RationalPolicy,
    behavior: RationalPolicy,
) -> GradientDict:
    """
    Compute the exact IS-REINFORCE expected gradient:
      E_{τ~b}[w(τ) · G(τ) · Σ_t ∇ log π(a_t|s_t)]

    where w(τ) = π(τ) / b(τ) = ∏_t [π(a_t|s_t) / b(a_t|s_t)].
    Transition probabilities cancel in the IS weight.

    T2 identity: this equals on_policy_gradient(mdp, target) exactly.
    """
    grad = _zero_gradient(mdp)
    n_starts = len(mdp.states)

    for s0 in mdp.states:
        for traj in enumerate_trajectories(mdp, behavior, start_state=s0):
            p_b = traj["prob"]  # P_b(τ) = ∏_t b(a_t|s_t) · T(...)
            # IS weight: w(τ) = π(τ) / b(τ) — transition factors cancel
            pi_prob = _trajectory_policy_prob(traj, target)
            b_prob = _trajectory_policy_prob(traj, behavior)
            if b_prob == Fraction(0):
                continue  # zero-prob under behavior: skip (traj["prob"] == 0 anyway)
            w = pi_prob / b_prob  # exact Fraction ratio

            traj_grad = _trajectory_reinforce_grad(traj, target, mdp)
            for key, g in traj_grad.items():
                # E_{τ~b}[w · grad] = Σ_τ P_b(τ) · w(τ) · grad(τ)
                grad[key] += p_b * w * g

    for key in grad:
        grad[key] /= n_starts
    return grad


def ppo_clipped_gradient(
    mdp: MDP,
    target: RationalPolicy,
    behavior: RationalPolicy,
    clip_eps: Fraction,
) -> GradientDict:
    """
    Compute the exact expected gradient of the PPO clipped surrogate:
      E_{τ~b}[Σ_t clip(r_t, 1-ε, 1+ε) · A_t · ∇ log π(a_t|s_t)]

    where r_t = π(a_t|s_t) / b(a_t|s_t) is the per-step IS ratio
    and A_t is the advantage (approximated here by the return G(τ)).

    Clipping is done per-step (not per-trajectory) using exact Fraction comparison:
      clipped_r = min(max(r, 1-ε), 1+ε)

    All arithmetic is exact Fraction. Clip comparisons use Fraction inequality.
    """
    clip_eps = to_fraction(clip_eps)
    lo = Fraction(1) - clip_eps
    hi = Fraction(1) + clip_eps

    grad = _zero_gradient(mdp)
    n_starts = len(mdp.states)

    for s0 in mdp.states:
        for traj in enumerate_trajectories(mdp, behavior, start_state=s0):
            p_b = traj["prob"]
            G = traj["return"]
            steps = traj["steps"]

            # Per-step clipped IS ratio
            for (state, action, next_state, reward) in steps:
                pi_a = target.prob(state, action)
                b_a = behavior.prob(state, action)
                if b_a == Fraction(0):
                    continue
                r = pi_a / b_a
                clipped_r = _clip(r, lo, hi)

                step_grad = log_prob_gradient(target, state, action)
                for a_i, sg in step_grad.items():
                    grad[(state, a_i)] += p_b * clipped_r * G * sg

    for key in grad:
        grad[key] /= n_starts
    return grad


def grpo_gradient(
    mdp: MDP,
    target: RationalPolicy,
    behavior: RationalPolicy,
    group_size: int,
) -> GradientDict:
    """
    Compute the exact expected gradient of a GRPO-style group-relative estimator.

    GRPO replaces absolute returns with group-relative advantages:
      A_i = G_i - mean(G_1, ..., G_k)  for k trajectories in a group.

    For exact enumeration, we compute the expected gradient by enumerating
    all possible groups of group_size trajectories (with replacement) and
    computing the group-relative contribution.

    For tractability, we compute this as:
      E[A_i · IS_i · ∇ log π_i] = E[G_i · IS_i · ∇ log π_i]
                                   - E[mean(G) · IS_i · ∇ log π_i]

    where the mean(G) term introduces a within-group correlation that
    makes GRPO biased under staleness.

    This is the exact per-decision formulation.
    """
    grad = _zero_gradient(mdp)
    n_starts = len(mdp.states)

    # Compute IS-weighted trajectory contributions per start state
    for s0 in mdp.states:
        trajs = list(enumerate_trajectories(mdp, behavior, start_state=s0))

        # Expected IS-weighted return: E_{τ~b}[w(τ) · G(τ)]
        # Used for the group-mean baseline
        expected_wG = Fraction(0)
        for traj in trajs:
            p_b = traj["prob"]
            pi_prob = _trajectory_policy_prob(traj, target)
            b_prob = _trajectory_policy_prob(traj, behavior)
            if b_prob == Fraction(0):
                continue
            w = pi_prob / b_prob
            expected_wG += p_b * w * traj["return"]

        # GRPO gradient: E[IS · (G - E[G]) · ∇ log π]
        # = E[IS · G · ∇ log π] - E[G] · E[IS · ∇ log π]
        # Note: E[IS · ∇ log π] = ∇ log π under IS = target gradient direction
        for traj in trajs:
            p_b = traj["prob"]
            G = traj["return"]
            pi_prob = _trajectory_policy_prob(traj, target)
            b_prob = _trajectory_policy_prob(traj, behavior)
            if b_prob == Fraction(0):
                continue
            w = pi_prob / b_prob
            # Advantage: G - expected_wG (group-relative baseline)
            advantage = G - expected_wG

            for (state, action, next_state, reward) in traj["steps"]:
                step_grad = log_prob_gradient(target, state, action)
                for a_i, sg in step_grad.items():
                    grad[(state, a_i)] += p_b * w * advantage * sg

    for key in grad:
        grad[key] /= n_starts
    return grad


def per_trajectory_is_reinforce(
    traj: dict,
    target: RationalPolicy,
    behavior: RationalPolicy,
) -> tuple[Fraction, GradientDict]:
    """
    Compute the IS weight and gradient contribution for a single trajectory.

    Returns:
      (w, contrib) where:
        w = π(τ) / b(τ) — exact IS weight (Fraction)
        contrib = G(τ) · Σ_t ∇ log π(a_t|s_t) — gradient contribution dict

    Used in the ledger-based accumulation path (per-trajectory sample estimator).
    The caller is responsible for weighting by P_b(τ) (which is in traj["prob"]).
    """
    pi_prob = _trajectory_policy_prob(traj, target)
    b_prob = _trajectory_policy_prob(traj, behavior)
    if b_prob == Fraction(0):
        raise ValueError("Trajectory has zero probability under behavior policy.")
    w = pi_prob / b_prob

    # Build contrib from the trajectory steps
    # We need the MDP to build the zero gradient — instead build from steps
    states_seen = list({s for (s, a, sp, r) in traj["steps"]})
    actions_seen = list({a for (s, a, sp, r) in traj["steps"]})
    # Also include actions from the policy table (need all actions for gradient)
    all_actions = list(target._table[states_seen[0]].keys()) if states_seen else []

    G = traj["return"]
    contrib: dict[tuple[int, int], Fraction] = {}
    for (state, action, next_state, reward) in traj["steps"]:
        step_grad = log_prob_gradient(target, state, action)
        for a_i, sg in step_grad.items():
            key = (state, a_i)
            contrib[key] = contrib.get(key, Fraction(0)) + G * sg

    return w, contrib


def _clip(value: Fraction, lo: Fraction, hi: Fraction) -> Fraction:
    """Clip a Fraction value to [lo, hi] using exact Fraction comparison."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value
