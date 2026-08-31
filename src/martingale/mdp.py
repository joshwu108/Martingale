"""
mdp.py — rational MDPs and exact trajectory enumeration engine.

An MDP is represented with all probabilities and rewards as fractions.Fraction.
The enumeration engine walks all A^H action sequences via DFS, accumulating
exact path probabilities and trajectory functionals.

Mixed-revision support: the policy argument to enumerate_trajectories may be
  - a dict {state -> {action -> Fraction}}  (same policy at every step), or
  - a list of such dicts (per-step policy, enabling mixed-revision trajectories).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Iterator, Union

from martingale.rational import check_simplex, to_fraction


# Type aliases
TransitionTable = dict[int, dict[int, dict[int, Fraction]]]  # T[s][a][s'] = prob
RewardTable = dict[int, dict[int, dict[int, Fraction]]]       # R[s][a][s'] = reward
PolicyDict = dict[int, dict[int, Fraction]]                   # π[s][a] = prob
PerStepPolicy = Union[PolicyDict, list[PolicyDict]]


def feasibility_count(n_states: int, n_actions: int, horizon: int, check: bool = False) -> int:
    """
    Return the total trajectory count: n_states * n_actions^horizon.

    If check=True, raises ValueError if count > 10^7.
    """
    count = n_states * (n_actions ** horizon)
    if check and count > 10**7:
        raise ValueError(
            f"Trajectory count {count} exceeds 10^7 limit "
            f"(n_states={n_states}, n_actions={n_actions}, horizon={horizon}). "
            "Reduce MDP size or horizon."
        )
    return count


class MDP:
    """
    A finite-horizon rational MDP.

    All transition probabilities and rewards must be fractions.Fraction.
    Validation is performed at construction time.
    """

    def __init__(
        self,
        states: list[int],
        actions: list[int],
        transitions: TransitionTable,
        rewards: RewardTable,
        horizon: int,
    ) -> None:
        self.states = list(states)
        self.actions = list(actions)
        self.transitions = transitions
        self.rewards = rewards
        self.horizon = horizon
        self._validate()

    def _validate(self) -> None:
        for s in self.states:
            for a in self.actions:
                row = self.transitions.get(s, {}).get(a, {})
                # Check for negative probabilities
                for sp, prob in row.items():
                    prob = to_fraction(prob)
                    if prob < 0:
                        raise ValueError(
                            f"negative transition probability T[{s}][{a}][{sp}] = {prob}"
                        )
                # Check sum = 1
                total = sum(to_fraction(p) for p in row.values())
                if total != Fraction(1):
                    raise ValueError(
                        f"transition T[{s}][{a}] sums to {total} (expected 1); "
                        f"row = {dict(row)}"
                    )

    def transition_prob(self, s: int, a: int, sp: int) -> Fraction:
        return to_fraction(self.transitions[s][a][sp])

    def reward(self, s: int, a: int, sp: int) -> Fraction:
        return to_fraction(self.rewards[s][a][sp])


def _get_step_policy(policy: PerStepPolicy, t: int) -> PolicyDict:
    """Extract the policy dict for time step t."""
    # Support RationalPolicy objects (import lazily to avoid circular imports)
    from martingale.policy import RationalPolicy as _RationalPolicy
    if isinstance(policy, list):
        step = policy[t]
        if isinstance(step, _RationalPolicy):
            return step.as_dict()
        return step
    if isinstance(policy, _RationalPolicy):
        return policy.as_dict()
    return policy


def enumerate_trajectories(
    mdp: MDP,
    policy: PerStepPolicy,
    start_state: int,
) -> Iterator[dict]:
    """
    Enumerate all trajectories of length mdp.horizon starting from start_state.

    Yields dicts with:
      - "prob": Fraction — exact path probability under the given policy
      - "steps": list of (state, action, next_state, reward) tuples
      - "return": Fraction — sum of rewards along the trajectory

    policy may be:
      - a single PolicyDict (same policy at every step), or
      - a list of PolicyDicts (per-step, enabling mixed-revision analysis).

    Only trajectories with prob > 0 are yielded (zero-prob paths are pruned).
    """
    # DFS over action sequences
    # Stack entries: (current_state, current_step, cumulative_prob, steps_so_far, cumulative_return)
    stack = [(start_state, 0, Fraction(1), [], Fraction(0))]

    while stack:
        state, t, prob, steps, ret = stack.pop()

        if t == mdp.horizon:
            yield {"prob": prob, "steps": steps, "return": ret}
            continue

        step_policy = _get_step_policy(policy, t)
        action_probs = step_policy[state]

        for action in mdp.actions:
            pi_a = to_fraction(action_probs.get(action, Fraction(0)))
            if pi_a == Fraction(0):
                continue  # prune zero-prob branches

            next_state_dist = mdp.transitions[state][action]
            for next_state, trans_prob in next_state_dist.items():
                trans_prob = to_fraction(trans_prob)
                if trans_prob == Fraction(0):
                    continue

                r = mdp.reward(state, action, next_state)
                new_prob = prob * pi_a * trans_prob
                new_steps = steps + [(state, action, next_state, r)]
                new_ret = ret + r

                stack.append((next_state, t + 1, new_prob, new_steps, new_ret))
