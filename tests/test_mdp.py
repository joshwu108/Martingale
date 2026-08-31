"""Tests for mdp.py — rational MDPs and exact enumeration engine."""
import pytest
from fractions import Fraction
from martingale.mdp import MDP, enumerate_trajectories, feasibility_count


class TestMDP:
    def _two_state_mdp(self):
        """Simple deterministic 2-state MDP."""
        states = [0, 1]
        actions = [0, 1]
        # T[s][a][s'] = prob
        transitions = {
            0: {0: {0: Fraction(1)}, 1: {1: Fraction(1)}},
            1: {0: {0: Fraction(1)}, 1: {1: Fraction(1)}},
        }
        rewards = {
            0: {0: {0: Fraction(1)}, 1: {1: Fraction(2)}},
            1: {0: {0: Fraction(0)}, 1: {1: Fraction(3)}},
        }
        return MDP(states=states, actions=actions, transitions=transitions, rewards=rewards, horizon=2)

    def test_construction(self):
        mdp = self._two_state_mdp()
        assert len(mdp.states) == 2
        assert len(mdp.actions) == 2
        assert mdp.horizon == 2

    def test_transition_sums_to_one(self):
        mdp = self._two_state_mdp()
        for s in mdp.states:
            for a in mdp.actions:
                total = sum(mdp.transitions[s][a].values())
                assert total == Fraction(1), f"Transition T[{s}][{a}] sums to {total}"

    def test_invalid_transition_rejected(self):
        """Transition rows that don't sum to 1 are rejected."""
        with pytest.raises(ValueError, match="transition"):
            MDP(
                states=[0, 1],
                actions=[0],
                transitions={0: {0: {0: Fraction(1, 2)}},  # sums to 1/2
                             1: {0: {1: Fraction(1)}}},
                rewards={0: {0: {0: Fraction(0)}}, 1: {0: {1: Fraction(0)}}},
                horizon=1,
            )

    def test_negative_transition_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            MDP(
                states=[0, 1],
                actions=[0],
                transitions={0: {0: {0: Fraction(-1), 1: Fraction(2)}},
                             1: {0: {1: Fraction(1)}}},
                rewards={0: {0: {0: Fraction(0), 1: Fraction(0)}},
                         1: {0: {1: Fraction(0)}}},
                horizon=1,
            )


class TestFeasibilityCount:
    def test_small_mdp(self):
        # |S|=2, |A|=2, H=3 → 2 * 2^3 = 16
        assert feasibility_count(n_states=2, n_actions=2, horizon=3) == 16

    def test_boundary(self):
        # 5 * 3^5 = 5 * 243 = 1215 — well under 10^7
        count = feasibility_count(n_states=5, n_actions=3, horizon=5)
        assert count == 5 * 3**5

    def test_over_limit_raises(self):
        with pytest.raises(ValueError, match=r"10\^7"):
            # |S|=5, |A|=3, H=15 → 5 * 3^15 = 5 * 14348907 >> 10^7
            feasibility_count(n_states=5, n_actions=3, horizon=15, check=True)


class TestEnumerateTrajectories:
    def _simple_mdp(self):
        """Stochastic 2-state MDP for enumeration tests."""
        return MDP(
            states=[0, 1],
            actions=[0, 1],
            transitions={
                0: {
                    0: {0: Fraction(3, 4), 1: Fraction(1, 4)},
                    1: {0: Fraction(1, 4), 1: Fraction(3, 4)},
                },
                1: {
                    0: {0: Fraction(1, 2), 1: Fraction(1, 2)},
                    1: {0: Fraction(1, 2), 1: Fraction(1, 2)},
                },
            },
            rewards={
                0: {0: {0: Fraction(1), 1: Fraction(0)},
                    1: {0: Fraction(0), 1: Fraction(1)}},
                1: {0: {0: Fraction(1), 1: Fraction(1)},
                    1: {0: Fraction(1), 1: Fraction(1)}},
            },
            horizon=2,
        )

    def test_probabilities_sum_to_one(self):
        """Sum of path probabilities over all trajectories = 1 (per start state)."""
        mdp = self._simple_mdp()
        # Uniform policy
        policy = {s: {a: Fraction(1, 2) for a in mdp.actions} for s in mdp.states}
        for s0 in mdp.states:
            trajectories = list(enumerate_trajectories(mdp, policy, start_state=s0))
            total_prob = sum(traj["prob"] for traj in trajectories)
            assert total_prob == Fraction(1), f"Probs from s0={s0} sum to {total_prob}"

    def test_trajectory_count(self):
        """Full trajectory enumeration for stochastic MDP: (|A|*|S|)^H = 4^2 = 16."""
        mdp = self._simple_mdp()
        policy = {s: {a: Fraction(1, 2) for a in mdp.actions} for s in mdp.states}
        for s0 in mdp.states:
            trajs = list(enumerate_trajectories(mdp, policy, start_state=s0))
            # Each step: 2 action choices × 2 next-state outcomes = 4 branches; 4^2 = 16 for H=2
            assert len(trajs) == 16, f"Expected 16 trajectories from s0={s0}, got {len(trajs)}"

    def test_trajectory_structure(self):
        """Each trajectory has 'prob', 'steps', 'return' keys."""
        mdp = self._simple_mdp()
        policy = {s: {a: Fraction(1, 2) for a in mdp.actions} for s in mdp.states}
        trajs = list(enumerate_trajectories(mdp, policy, start_state=0))
        for traj in trajs:
            assert "prob" in traj
            assert "steps" in traj
            assert "return" in traj
            assert isinstance(traj["prob"], Fraction)
            assert isinstance(traj["return"], Fraction)
            assert len(traj["steps"]) == mdp.horizon

    def test_deterministic_policy_single_trajectory(self):
        """A deterministic policy yields exactly one trajectory with prob=1."""
        mdp = MDP(
            states=[0],
            actions=[0, 1],
            transitions={0: {0: {0: Fraction(1)}, 1: {0: Fraction(1)}}},
            rewards={0: {0: {0: Fraction(1)}, 1: {0: Fraction(2)}}},
            horizon=2,
        )
        # Always pick action 0
        policy = {0: {0: Fraction(1), 1: Fraction(0)}}
        trajs = list(enumerate_trajectories(mdp, policy, start_state=0))
        nonzero = [t for t in trajs if t["prob"] > 0]
        assert len(nonzero) == 1
        assert nonzero[0]["prob"] == Fraction(1)
        assert nonzero[0]["return"] == Fraction(2)  # reward 1 at step 0 + reward 1 at step 1

    def test_mixed_revision_enumeration(self):
        """Per-step policy dicts allow mixed-revision trajectories."""
        mdp = self._simple_mdp()
        # Different policy at each time step
        policy_t0 = {s: {a: Fraction(1, 2) for a in mdp.actions} for s in mdp.states}
        policy_t1 = {s: {0: Fraction(3, 4), 1: Fraction(1, 4)} for s in mdp.states}
        per_step_policies = [policy_t0, policy_t1]
        trajs = list(enumerate_trajectories(mdp, per_step_policies, start_state=0))
        total = sum(t["prob"] for t in trajs)
        assert total == Fraction(1)
