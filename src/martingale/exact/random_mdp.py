"""
exact/random_mdp.py — seeded random rational MDPs and policies.

Extracted verbatim from campaigns/identity.py so the test bench and the T2
campaign draw from one implementation. Same seed bytes produce the same
objects as before the extraction (pinned by tests/test_exact_bench.py).
"""
from __future__ import annotations

import hashlib
from fractions import Fraction

from martingale.mdp import MDP
from martingale.policy import RationalPolicy, gradient_ascent_step


def seeded_fraction(seed: bytes, lo_num: int, lo_den: int,
                    hi_num: int, hi_den: int) -> Fraction:
    """Generate a deterministic rational in [lo, hi] from seed bytes."""
    h = int.from_bytes(hashlib.blake2b(seed, digest_size=8).digest(), "big")
    lo = Fraction(lo_num, lo_den)
    hi = Fraction(hi_num, hi_den)
    t = Fraction(h, 2**64)
    return lo + t * (hi - lo)


def make_rational_simplex(seed: bytes, n: int) -> dict[int, Fraction]:
    """Generate a random rational simplex of size n, summing to exactly 1."""
    raws = []
    for i in range(n):
        s = hashlib.blake2b(seed + i.to_bytes(4, "big"), digest_size=8).digest()
        v = Fraction(int.from_bytes(s, "big") + 1, 2**64)  # in (0, 1]
        raws.append(v)
    total = sum(raws)
    return {i: r / total for i, r in enumerate(raws)}


def make_mdp(seed: bytes, n_states: int, n_actions: int, horizon: int) -> MDP:
    """Generate a random rational MDP from a seed. Rewards are in {0,1,2,3}."""
    states = list(range(n_states))
    actions = list(range(n_actions))

    transitions: dict = {}
    rewards: dict = {}
    for s in states:
        transitions[s] = {}
        rewards[s] = {}
        for a in actions:
            s_seed = hashlib.blake2b(seed + f"T_{s}_{a}".encode(), digest_size=16).digest()
            transitions[s][a] = make_rational_simplex(s_seed, n_states)
            rewards[s][a] = {}
            for sp in states:
                r_seed = hashlib.blake2b(seed + f"R_{s}_{a}_{sp}".encode(), digest_size=8).digest()
                rewards[s][a][sp] = Fraction(int.from_bytes(r_seed[:1], "big") % 4)

    return MDP(states=states, actions=actions, transitions=transitions,
               rewards=rewards, horizon=horizon)


def make_policy(seed: bytes, states: list[int], actions: list[int]) -> RationalPolicy:
    """Generate a random rational simplex policy."""
    table: dict = {}
    for s in states:
        s_seed = hashlib.blake2b(seed + f"PI_{s}".encode(), digest_size=16).digest()
        simplex = make_rational_simplex(s_seed, len(actions))
        table[s] = {actions[i]: p for i, p in simplex.items()}
    return RationalPolicy(table)


def make_stale_policy(
    target: RationalPolicy,
    mdp: MDP,
    n_steps: int,
    alpha: Fraction,
    action_seq_seed: bytes,
) -> RationalPolicy:
    """Behavior policy = target after n_steps exact gradient-ascent steps (lag = n_steps)."""
    policy = target
    for step_i in range(n_steps):
        s_seed = hashlib.blake2b(action_seq_seed + step_i.to_bytes(4, "big"), digest_size=4).digest()
        s_idx = int.from_bytes(s_seed[:2], "big") % len(mdp.states)
        a_idx = int.from_bytes(s_seed[2:4], "big") % len(mdp.actions)
        policy = gradient_ascent_step(policy, mdp.states[s_idx], mdp.actions[a_idx],
                                      G=Fraction(1), alpha=alpha)
    return policy
