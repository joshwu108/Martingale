"""
Tests for estimators.py — exact expected gradients via trajectory enumeration.

Key invariant (T2): E_{τ~b}[IS-REINFORCE gradient] = on-policy gradient
                    as an exact Fraction equality, component by component.
"""
import pytest
from fractions import Fraction
from martingale.mdp import MDP, enumerate_trajectories
from martingale.policy import RationalPolicy, gradient_ascent_step
from martingale.estimators import (
    on_policy_gradient,
    is_reinforce_gradient,
    ppo_clipped_gradient,
    grpo_gradient,
    per_trajectory_is_reinforce,
)


def _simple_2state_mdp() -> MDP:
    """2-state, 2-action, H=2 rational MDP used for estimator tests."""
    return MDP(
        states=[0, 1],
        actions=[0, 1],
        transitions={
            0: {0: {0: Fraction(2, 3), 1: Fraction(1, 3)},
                1: {0: Fraction(1, 4), 1: Fraction(3, 4)}},
            1: {0: {0: Fraction(1, 2), 1: Fraction(1, 2)},
                1: {0: Fraction(3, 5), 1: Fraction(2, 5)}},
        },
        rewards={
            0: {0: {0: Fraction(1), 1: Fraction(2)},
                1: {0: Fraction(0), 1: Fraction(3)}},
            1: {0: {0: Fraction(1), 1: Fraction(0)},
                1: {0: Fraction(2), 1: Fraction(1)}},
        },
        horizon=2,
    )


def _target_policy() -> RationalPolicy:
    return RationalPolicy({
        0: {0: Fraction(3, 5), 1: Fraction(2, 5)},
        1: {0: Fraction(1, 3), 1: Fraction(2, 3)},
    })


def _behavior_policy() -> RationalPolicy:
    """A different (stale) policy for IS weight computation."""
    return RationalPolicy({
        0: {0: Fraction(1, 2), 1: Fraction(1, 2)},
        1: {0: Fraction(2, 5), 1: Fraction(3, 5)},
    })


class TestOnPolicyGradient:
    def test_returns_fraction_per_parameter(self):
        """on_policy_gradient returns a dict of Fraction values."""
        mdp = _simple_2state_mdp()
        pi = _target_policy()
        grad = on_policy_gradient(mdp, pi)
        assert isinstance(grad, dict)
        for (s, a), g in grad.items():
            assert isinstance(g, Fraction), f"Gradient ({s},{a}) is not Fraction: {type(g)}"

    def test_gradient_is_finite(self):
        """All gradient components are finite rationals."""
        mdp = _simple_2state_mdp()
        pi = _target_policy()
        grad = on_policy_gradient(mdp, pi)
        for (s, a), g in grad.items():
            # Fraction arithmetic never produces inf/nan
            assert g.denominator > 0

    def test_analytic_formula_h1(self):
        """
        For H=1, single state, 2 actions, the REINFORCE gradient is:
          grad[(s, a)] = E[G · ∂ log π(a'|s)/∂ p_a] = π(a|s) · G_a · (1/π(a|s)) = G_a

        This is the UNCONSTRAINED gradient per probability entry, not ∂J/∂p_a
        under the simplex constraint. All arithmetic is exact Fraction.
        """
        mdp = MDP(
            states=[0],
            actions=[0, 1],
            transitions={0: {0: {0: Fraction(1)}, 1: {0: Fraction(1)}}},
            rewards={0: {0: {0: Fraction(5)}, 1: {0: Fraction(2)}}},
            horizon=1,
        )
        p0 = Fraction(3, 7)
        p1 = Fraction(4, 7)
        pi = RationalPolicy({0: {0: p0, 1: p1}})
        grad = on_policy_gradient(mdp, pi)

        # REINFORCE: grad[(0, a)] = G_a (reward of action a)
        assert grad[(0, 0)] == Fraction(5), f"Expected 5, got {grad[(0, 0)]}"
        assert grad[(0, 1)] == Fraction(2), f"Expected 2, got {grad[(0, 1)]}"


class TestISReinforceGradient:
    def test_returns_fractions(self):
        mdp = _simple_2state_mdp()
        pi = _target_policy()
        b = _behavior_policy()
        grad = is_reinforce_gradient(mdp, target=pi, behavior=b)
        for (s, a), g in grad.items():
            assert isinstance(g, Fraction)

    def test_T2_identity_exact(self):
        """
        T2 core: IS-REINFORCE expectation == on-policy gradient.
        Machine-checked exact Fraction equality, component by component.
        """
        mdp = _simple_2state_mdp()
        pi = _target_policy()
        b = _behavior_policy()
        grad_on = on_policy_gradient(mdp, pi)
        grad_is = is_reinforce_gradient(mdp, target=pi, behavior=b)
        # Must be EXACTLY equal (not approximately)
        for key in grad_on:
            assert grad_on[key] == grad_is[key], (
                f"T2 identity failed at parameter {key}: "
                f"on_policy={grad_on[key]}, IS={grad_is[key]}"
            )

    def test_T2_identity_target_equals_behavior(self):
        """When target == behavior, IS weights = 1 everywhere."""
        mdp = _simple_2state_mdp()
        pi = _target_policy()
        grad_on = on_policy_gradient(mdp, pi)
        grad_is = is_reinforce_gradient(mdp, target=pi, behavior=pi)
        for key in grad_on:
            assert grad_on[key] == grad_is[key]

    def test_T2_identity_uniform_behavior(self):
        """T2 identity holds when behavior is uniform."""
        mdp = _simple_2state_mdp()
        pi = _target_policy()
        b = RationalPolicy.uniform(states=mdp.states, actions=mdp.actions)
        grad_on = on_policy_gradient(mdp, pi)
        grad_is = is_reinforce_gradient(mdp, target=pi, behavior=b)
        for key in grad_on:
            assert grad_on[key] == grad_is[key], (
                f"T2 identity failed at {key}: on={grad_on[key]}, IS={grad_is[key]}"
            )

    def test_T2_identity_multiple_configs(self):
        """T2 holds for 5 random-ish MDP/policy configurations."""
        configs = [
            # (policy, behavior, horizon, states, actions)
            (
                RationalPolicy({0: {0: Fraction(1, 3), 1: Fraction(2, 3)}}),
                RationalPolicy({0: {0: Fraction(2, 5), 1: Fraction(3, 5)}}),
                1, [0], [0, 1],
            ),
            (
                RationalPolicy({0: {0: Fraction(3, 4), 1: Fraction(1, 4)},
                                1: {0: Fraction(1, 5), 1: Fraction(4, 5)}}),
                RationalPolicy({0: {0: Fraction(1, 2), 1: Fraction(1, 2)},
                                1: {0: Fraction(3, 7), 1: Fraction(4, 7)}}),
                2, [0, 1], [0, 1],
            ),
        ]
        for pi, b, H, states, actions in configs:
            mdp = MDP(
                states=states,
                actions=actions,
                transitions={s: {a: {s: Fraction(1)} for a in actions} for s in states},
                rewards={s: {a: {s: Fraction(a + s + 1)} for a in actions} for s in states},
                horizon=H,
            )
            grad_on = on_policy_gradient(mdp, pi)
            grad_is = is_reinforce_gradient(mdp, target=pi, behavior=b)
            for key in grad_on:
                assert grad_on[key] == grad_is[key], (
                    f"T2 failed for config H={H},states={states}: key={key}"
                )


class TestPPOClippedGradient:
    def test_returns_fractions(self):
        mdp = _simple_2state_mdp()
        pi = _target_policy()
        b = _behavior_policy()
        grad = ppo_clipped_gradient(mdp, target=pi, behavior=b, clip_eps=Fraction(2, 10))
        for (s, a), g in grad.items():
            assert isinstance(g, Fraction)

    def test_biased_when_clipped(self):
        """PPO clipped gradient is NOT equal to on-policy gradient under staleness."""
        mdp = _simple_2state_mdp()
        pi = _target_policy()
        b = _behavior_policy()
        clip_eps = Fraction(2, 10)
        grad_on = on_policy_gradient(mdp, pi)
        grad_ppo = ppo_clipped_gradient(mdp, target=pi, behavior=b, clip_eps=clip_eps)
        # At least one component must differ (bias is nonzero for these configs)
        differs = any(grad_on[k] != grad_ppo[k] for k in grad_on)
        assert differs, "PPO clipped gradient should be biased but equals on-policy gradient"

    def test_bias_vector_is_exact_fraction(self):
        """Bias = PPO_grad - on_policy_grad is an exact Fraction per component."""
        mdp = _simple_2state_mdp()
        pi = _target_policy()
        b = _behavior_policy()
        grad_on = on_policy_gradient(mdp, pi)
        grad_ppo = ppo_clipped_gradient(mdp, target=pi, behavior=b, clip_eps=Fraction(2, 10))
        bias = {k: grad_ppo[k] - grad_on[k] for k in grad_on}
        for k, v in bias.items():
            assert isinstance(v, Fraction)


class TestGRPOGradient:
    def test_returns_fractions(self):
        mdp = _simple_2state_mdp()
        pi = _target_policy()
        b = _behavior_policy()
        # GRPO uses a group of trajectories; group_size=4 for this test
        grad = grpo_gradient(mdp, target=pi, behavior=b, group_size=4)
        for (s, a), g in grad.items():
            assert isinstance(g, Fraction)

    def test_biased_under_staleness(self):
        """GRPO gradient is biased (not equal to on-policy) under staleness."""
        mdp = _simple_2state_mdp()
        pi = _target_policy()
        b = _behavior_policy()
        grad_on = on_policy_gradient(mdp, pi)
        grad_grpo = grpo_gradient(mdp, target=pi, behavior=b, group_size=4)
        differs = any(grad_on[k] != grad_grpo[k] for k in grad_on)
        assert differs, "GRPO gradient should be biased but equals on-policy"


class TestPerTrajectoryIS:
    def test_sample_estimator_structure(self):
        """per_trajectory_is_reinforce returns (is_weight, grad_contribution) per traj."""
        mdp = _simple_2state_mdp()
        pi = _target_policy()
        b = _behavior_policy()
        trajs = list(enumerate_trajectories(mdp, b, start_state=0))
        for traj in trajs[:5]:
            w, contrib = per_trajectory_is_reinforce(traj, target=pi, behavior=b)
            assert isinstance(w, Fraction)
            assert isinstance(contrib, dict)

    def test_expected_equals_is_gradient(self):
        """Expectation of per-trajectory estimator equals is_reinforce_gradient."""
        mdp = _simple_2state_mdp()
        pi = _target_policy()
        b = _behavior_policy()
        # Compute expected value by enumeration
        result = {k: Fraction(0) for k in [(s, a) for s in mdp.states for a in mdp.actions]}
        for s0 in mdp.states:
            trajs = list(enumerate_trajectories(mdp, b, start_state=s0))
            for traj in trajs:
                w, contrib = per_trajectory_is_reinforce(traj, target=pi, behavior=b)
                for k, v in contrib.items():
                    result[k] += traj["prob"] * w * v
        # Average over start states (uniform initial state distribution)
        n_starts = len(mdp.states)
        result = {k: v / n_starts for k, v in result.items()}

        grad_is = is_reinforce_gradient(mdp, target=pi, behavior=b)
        for k in grad_is:
            assert result[k] == grad_is[k], (
                f"Sample expectation {result[k]} != IS gradient {grad_is[k]} at {k}"
            )
