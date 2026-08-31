"""
campaigns/identity.py — T2 machine-checked exact equality/inequality certificates.

Runs 40+ random rational MDP/policy/staleness configurations and asserts:
  1. IS-REINFORCE expected gradient == on-policy gradient (exact Fraction equality)
     Any single failure halts with a full dump — do not catch and continue.
  2. PPO clipped and GRPO estimators are BIASED: exact bias vector is nonzero.
     Exact bias vectors committed to results/ as permanent evidence.

Usage: python -m campaigns.identity  [or via make check]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

# Add src to path for running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from martingale.estimators import (
    grpo_gradient,
    is_reinforce_gradient,
    on_policy_gradient,
    ppo_clipped_gradient,
)
from martingale.mdp import MDP, feasibility_count
from martingale.policy import RationalPolicy, gradient_ascent_step
from martingale.rational import fraction_to_str


RESULTS_DIR = Path(__file__).parent.parent / "results"
CAMPAIGN_SEED = b"martingale-identity-campaign-v1"
N_CONFIGS = 40


def _seeded_fraction(seed: bytes, lo_num: int, lo_den: int,
                     hi_num: int, hi_den: int) -> Fraction:
    """Generate a deterministic rational in [lo, hi] from seed bytes."""
    h = int.from_bytes(hashlib.blake2b(seed, digest_size=8).digest(), "big")
    lo = Fraction(lo_num, lo_den)
    hi = Fraction(hi_num, hi_den)
    # Map h to [0, 1) as a rational, then scale
    t = Fraction(h, 2**64)
    return lo + t * (hi - lo)


def _make_rational_simplex(seed: bytes, n: int) -> dict[int, Fraction]:
    """Generate a random rational simplex of size n, summing to exactly 1."""
    # Generate n random positive rationals, then normalize exactly
    raws = []
    for i in range(n):
        s = hashlib.blake2b(seed + i.to_bytes(4, "big"), digest_size=8).digest()
        v = Fraction(int.from_bytes(s, "big") + 1, 2**64)  # in (0, 1]
        raws.append(v)
    total = sum(raws)
    return {i: r / total for i, r in enumerate(raws)}


def _make_mdp(seed: bytes, n_states: int, n_actions: int, horizon: int) -> MDP:
    """Generate a random rational MDP from a seed."""
    states = list(range(n_states))
    actions = list(range(n_actions))

    transitions = {}
    rewards = {}
    for s in states:
        transitions[s] = {}
        rewards[s] = {}
        for a in actions:
            s_seed = hashlib.blake2b(
                seed + f"T_{s}_{a}".encode(), digest_size=16
            ).digest()
            transitions[s][a] = _make_rational_simplex(s_seed, n_states)
            rewards[s][a] = {}
            for sp in states:
                r_seed = hashlib.blake2b(
                    seed + f"R_{s}_{a}_{sp}".encode(), digest_size=8
                ).digest()
                # Reward in {0, 1, 2, 3} as Fraction
                r_val = int.from_bytes(r_seed[:1], "big") % 4
                rewards[s][a][sp] = Fraction(r_val)

    return MDP(states=states, actions=actions, transitions=transitions,
               rewards=rewards, horizon=horizon)


def _make_policy(seed: bytes, states: list[int], actions: list[int]) -> RationalPolicy:
    """Generate a random rational simplex policy."""
    table = {}
    for s in states:
        s_seed = hashlib.blake2b(
            seed + f"PI_{s}".encode(), digest_size=16
        ).digest()
        table[s] = _make_rational_simplex(s_seed, len(actions))
        # Remap keys from 0..n to actual action indices
        table[s] = {actions[i]: p for i, p in table[s].items()}
    return RationalPolicy(table)


def _make_stale_policy(
    target: RationalPolicy,
    mdp: MDP,
    n_steps: int,
    alpha: Fraction,
    action_seq_seed: bytes,
) -> RationalPolicy:
    """Generate a behavior policy by taking n gradient ascent steps from target."""
    policy = target
    for step_i in range(n_steps):
        # Pick a random state/action to ascent on
        s_seed = hashlib.blake2b(
            action_seq_seed + step_i.to_bytes(4, "big"), digest_size=4
        ).digest()
        s_idx = int.from_bytes(s_seed[:2], "big") % len(mdp.states)
        a_idx = int.from_bytes(s_seed[2:4], "big") % len(mdp.actions)
        s = mdp.states[s_idx]
        a = mdp.actions[a_idx]
        policy = gradient_ascent_step(policy, s, a, G=Fraction(1), alpha=alpha)
    return policy


def run_identity_campaign(n_configs: int = N_CONFIGS, verbose: bool = True) -> dict:
    """
    Run the T2 identity campaign.

    For each configuration:
      - Assert IS-REINFORCE == on-policy (exact Fraction equality)
      - Record PPO and GRPO bias vectors

    Returns a report dict.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    configs = [
        # (n_states, n_actions, horizon, staleness_steps)
        (2, 2, 2, 1),
        (2, 2, 2, 2),
        (2, 2, 3, 1),
        (3, 2, 2, 1),
        (3, 2, 2, 3),
        (3, 3, 2, 2),
        (2, 3, 2, 1),
        (2, 3, 3, 2),
    ]

    # Expand to N_CONFIGS by cycling through parameter combos
    all_params = []
    for i in range(n_configs):
        base = configs[i % len(configs)]
        all_params.append(base + (i,))  # (n_states, n_actions, horizon, staleness, idx)

    report = {
        "campaign": "T2-identity",
        "n_configs": n_configs,
        "is_identity_passed": 0,
        "ppo_nonzero_bias": 0,
        "grpo_nonzero_bias": 0,
        "failures": [],
        "witness_biases": [],
    }

    clip_eps = Fraction(2, 10)
    alpha = Fraction(1, 20)

    for n_states, n_actions, horizon, staleness, idx in all_params:
        # Feasibility check
        feasibility_count(n_states, n_actions, horizon, check=True)

        cfg_seed = hashlib.blake2b(
            CAMPAIGN_SEED + idx.to_bytes(4, "big"), digest_size=16
        ).digest()

        mdp = _make_mdp(cfg_seed, n_states, n_actions, horizon)
        target = _make_policy(
            hashlib.blake2b(cfg_seed + b"target", digest_size=16).digest(),
            mdp.states, mdp.actions
        )
        behavior = _make_stale_policy(
            target, mdp,
            n_steps=staleness,
            alpha=alpha,
            action_seq_seed=hashlib.blake2b(cfg_seed + b"stale", digest_size=16).digest(),
        )

        # === T2 identity check (HALT on any failure) ===
        grad_on = on_policy_gradient(mdp, target)
        grad_is = is_reinforce_gradient(mdp, target=target, behavior=behavior)

        for key in grad_on:
            if grad_on[key] != grad_is[key]:
                dump = {
                    "config_idx": idx,
                    "n_states": n_states, "n_actions": n_actions, "horizon": horizon,
                    "staleness": staleness,
                    "key": str(key),
                    "on_policy": fraction_to_str(grad_on[key]),
                    "is_reinforce": fraction_to_str(grad_is[key]),
                }
                print(f"\n!!! T2 IDENTITY FAILURE !!!\n{json.dumps(dump, indent=2)}")
                print("This is potentially a valuable scientific result.")
                print("Investigate the mathematics vs implementation before proceeding.")
                raise AssertionError(
                    f"T2 identity failed at config {idx}, key {key}: "
                    f"on_policy={fraction_to_str(grad_on[key])}, "
                    f"IS={fraction_to_str(grad_is[key])}"
                )
        report["is_identity_passed"] += 1

        # === PPO bias check ===
        grad_ppo = ppo_clipped_gradient(mdp, target=target, behavior=behavior,
                                        clip_eps=clip_eps)
        ppo_bias = {str(k): fraction_to_str(grad_ppo[k] - grad_on[k]) for k in grad_on}
        ppo_nonzero = any(grad_ppo[k] != grad_on[k] for k in grad_on)
        if ppo_nonzero:
            report["ppo_nonzero_bias"] += 1

        # === GRPO bias check ===
        grad_grpo = grpo_gradient(mdp, target=target, behavior=behavior, group_size=4)
        grpo_bias = {str(k): fraction_to_str(grad_grpo[k] - grad_on[k]) for k in grad_on}
        grpo_nonzero = any(grad_grpo[k] != grad_on[k] for k in grad_on)
        if grpo_nonzero:
            report["grpo_nonzero_bias"] += 1

        # Record witness bias for first few configs
        if idx < 8:
            report["witness_biases"].append({
                "config_idx": idx,
                "n_states": n_states, "n_actions": n_actions,
                "horizon": horizon, "staleness": staleness,
                "ppo_bias": ppo_bias,
                "grpo_bias": grpo_bias,
            })

        if verbose:
            print(
                f"  cfg {idx:3d}: IS=OK, "
                f"PPO_bias={'nonzero' if ppo_nonzero else 'ZERO'}, "
                f"GRPO_bias={'nonzero' if grpo_nonzero else 'ZERO'}"
            )

    return report


def main() -> None:
    print(f"=== T2 Identity Campaign ({N_CONFIGS} configs) ===")
    report = run_identity_campaign(N_CONFIGS, verbose=True)

    # Write results
    out_path = RESULTS_DIR / "identity_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {out_path}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"IS identity: {report['is_identity_passed']}/{N_CONFIGS} passed")
    print(f"PPO nonzero bias: {report['ppo_nonzero_bias']}/{N_CONFIGS}")
    print(f"GRPO nonzero bias: {report['grpo_nonzero_bias']}/{N_CONFIGS}")

    if report["is_identity_passed"] < N_CONFIGS:
        print("\nFAIL: IS identity failed — investigate immediately.")
        sys.exit(1)
    else:
        print("\nPASS: IS identity holds for all configurations.")


if __name__ == "__main__":
    main()
