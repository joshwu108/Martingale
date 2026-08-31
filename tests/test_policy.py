"""Tests for policy.py — rational simplex parameterization and exact gradients."""
import pytest
from fractions import Fraction
from martingale.policy import (
    RationalPolicy,
    log_prob_gradient,
    gradient_ascent_step,
)


class TestRationalPolicy:
    def test_construction_from_dict(self):
        probs = {0: Fraction(1, 3), 1: Fraction(1, 3), 2: Fraction(1, 3)}
        p = RationalPolicy({0: probs})
        assert p.prob(state=0, action=0) == Fraction(1, 3)

    def test_construction_uniform(self):
        p = RationalPolicy.uniform(states=[0, 1], actions=[0, 1, 2])
        for s in [0, 1]:
            total = sum(p.prob(s, a) for a in [0, 1, 2])
            assert total == Fraction(1)
            for a in [0, 1, 2]:
                assert p.prob(s, a) == Fraction(1, 3)

    def test_sum_exactly_one(self):
        probs = {0: Fraction(1, 4), 1: Fraction(1, 2), 2: Fraction(1, 4)}
        p = RationalPolicy({0: probs})
        total = sum(p.prob(0, a) for a in [0, 1, 2])
        assert total == Fraction(1)

    def test_nonexistent_state_raises(self):
        p = RationalPolicy({0: {0: Fraction(1)}})
        with pytest.raises(KeyError):
            p.prob(99, 0)

    def test_invalid_prob_rejected(self):
        """Probability table not summing to 1 is rejected at construction."""
        with pytest.raises(ValueError, match="sum"):
            RationalPolicy({0: {0: Fraction(1, 2), 1: Fraction(1, 3)}})


class TestLogProbGradient:
    """
    For rational simplex policy, the gradient of log π(a|s) w.r.t. table entry p_i:
      ∂ log π(a|s) / ∂ p_i = 1/p_a  if i == a,  0 otherwise
    (treating other entries as fixed; simplex constraint handled in gradient_ascent_step).
    """

    def test_gradient_at_chosen_action(self):
        probs = {0: Fraction(1, 3), 1: Fraction(1, 3), 2: Fraction(1, 3)}
        p = RationalPolicy({0: probs})
        grad = log_prob_gradient(policy=p, state=0, action=1)
        # ∂ log(1/3) / ∂ p_1 = 1 / (1/3) = 3, w.r.t. p_1
        # ∂ log(1/3) / ∂ p_0 = 0 (action 0 ≠ chosen action 1)
        assert grad[1] == Fraction(3)
        assert grad[0] == Fraction(0)
        assert grad[2] == Fraction(0)

    def test_gradient_nonzero_only_at_chosen(self):
        probs = {0: Fraction(2, 5), 1: Fraction(3, 5)}
        p = RationalPolicy({0: probs})
        grad = log_prob_gradient(policy=p, state=0, action=0)
        assert grad[0] == Fraction(5, 2)  # 1 / (2/5) = 5/2
        assert grad[1] == Fraction(0)

    def test_gradient_formula_consistency(self):
        """
        Verify the log-prob gradient formula algebraically.

        For simplex policy: ∂ log π(a|s) / ∂ p_a = 1/p_a.
        We verify this via the chain rule applied to an exact polynomial function:
          f(p_a) = p_a  →  d/dp_a [p_a] = 1  (trivially exact)
          and log_prob_gradient returns 1/p_a.

        True exact FD (rational arithmetic) applies to J(p) — the expected return —
        which IS a polynomial in p, tested in test_estimators.py. Here we verify
        the gradient formula is algebraically consistent.
        """
        probs = {0: Fraction(2, 5), 1: Fraction(3, 5)}
        p = RationalPolicy({0: probs})
        # The gradient formula: ∂ log p_a / ∂ p_i = 1/p_a (i=a) else 0
        for chosen_action in [0, 1]:
            grad = log_prob_gradient(policy=p, state=0, action=chosen_action)
            # Analytic: 1/p_chosen_action for i=chosen_action, 0 otherwise
            expected_at_chosen = Fraction(1) / p.prob(0, chosen_action)
            assert grad[chosen_action] == expected_at_chosen
            for other in [0, 1]:
                if other != chosen_action:
                    assert grad[other] == Fraction(0)

    def test_finite_difference_via_expected_return(self):
        """
        Exact rational FD check of the gradient of J(p) for H=1 (where J is linear in p).

        For a 1-step MDP with deterministic transitions and H=1:
          J(p_a) = Σ_a p_a * r_a  (linear polynomial in p)
        So (J(p_a+ε) - J(p_a-ε)) / (2ε) = r_a  exactly (no approximation error).
        This equals ∂J/∂p_a analytically as well.
        All arithmetic is exact Fraction — no logs, no transcendentals.
        """
        from martingale.mdp import MDP, enumerate_trajectories

        # Single-state MDP, 2 actions, H=1, deterministic transitions
        mdp = MDP(
            states=[0],
            actions=[0, 1],
            transitions={0: {0: {0: Fraction(1)}, 1: {0: Fraction(1)}}},
            rewards={0: {0: {0: Fraction(3)}, 1: {0: Fraction(7)}}},
            horizon=1,
        )

        def compute_J(p_0: Fraction, p_1: Fraction) -> Fraction:
            """Exact expected return J = p_0 * 3 + p_1 * 7."""
            policy = {0: {0: p_0, 1: p_1}}
            trajs = list(enumerate_trajectories(mdp, policy, start_state=0))
            return sum(t["prob"] * t["return"] for t in trajs)

        p0 = Fraction(2, 5)
        p1 = Fraction(3, 5)
        eps = Fraction(1, 1000)

        # FD of J w.r.t. p_0 (adjust p_1 to maintain sum=1)
        J_fwd = compute_J(p0 + eps, p1 - eps)
        J_bwd = compute_J(p0 - eps, p1 + eps)
        fd_grad_p0 = (J_fwd - J_bwd) / (2 * eps)

        # Analytic: ∂J/∂p_0 = r_0 - r_1 (since p_1 = 1 - p_0, chain rule)
        # Actually ∂J/∂p_0 when p_1 = 1-p_0: dJ/dp_0 = r_0 - r_1 = 3 - 7 = -4
        expected_fd = Fraction(-4)
        assert fd_grad_p0 == expected_fd, f"FD={fd_grad_p0} != expected {expected_fd}"


class TestGradientAscentStep:
    def test_step_preserves_simplex_sum(self):
        """After one gradient ascent step, probabilities still sum to 1."""
        probs = {0: Fraction(1, 3), 1: Fraction(1, 3), 2: Fraction(1, 3)}
        p = RationalPolicy({0: probs})
        # Gradient: log π(a=0|s=0) gradient, return G=1
        alpha = Fraction(1, 10)
        new_p = gradient_ascent_step(policy=p, state=0, action=0, G=Fraction(1), alpha=alpha)
        total = sum(new_p.prob(0, a) for a in [0, 1, 2])
        assert total == Fraction(1), f"Simplex sum after step: {total}"

    def test_step_increases_chosen_action_prob(self):
        """Gradient ascent increases probability of the chosen action."""
        probs = {0: Fraction(1, 3), 1: Fraction(1, 3), 2: Fraction(1, 3)}
        p = RationalPolicy({0: probs})
        alpha = Fraction(1, 10)
        new_p = gradient_ascent_step(policy=p, state=0, action=0, G=Fraction(1), alpha=alpha)
        assert new_p.prob(0, 0) > p.prob(0, 0)

    def test_step_rationals_throughout(self):
        """All probabilities after the step are exact Fractions."""
        probs = {0: Fraction(2, 5), 1: Fraction(3, 5)}
        p = RationalPolicy({0: probs})
        alpha = Fraction(1, 10)
        new_p = gradient_ascent_step(policy=p, state=0, action=0, G=Fraction(2), alpha=alpha)
        for a in [0, 1]:
            prob = new_p.prob(0, a)
            assert isinstance(prob, Fraction), f"Prob is not Fraction: {type(prob)}"

    def test_multiple_steps_remain_valid(self):
        """After 10 gradient steps, probabilities remain valid and sum to 1."""
        probs = {0: Fraction(1, 3), 1: Fraction(1, 3), 2: Fraction(1, 3)}
        p = RationalPolicy({0: probs})
        alpha = Fraction(1, 20)
        for _ in range(10):
            p = gradient_ascent_step(policy=p, state=0, action=0, G=Fraction(1), alpha=alpha)
        total = sum(p.prob(0, a) for a in [0, 1, 2])
        assert total == Fraction(1)
        for a in [0, 1, 2]:
            assert p.prob(0, a) >= Fraction(0), "Negative probability after steps"
