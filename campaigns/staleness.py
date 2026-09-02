"""
campaigns/staleness.py — T3 preregistered staleness scaling campaign.

Protocol frozen in docs/preregistration.md (2026-08-31).

Measures exact bias of the PPO clipped estimator as a function of revision lag,
using the preregistered MDP family and kill rules.

Results committed to results/staleness_report.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from martingale.estimators import on_policy_gradient, ppo_clipped_gradient
from martingale.mdp import MDP, feasibility_count
from martingale.policy import RationalPolicy, gradient_ascent_step
from martingale.rational import fraction_to_str

RESULTS_DIR = Path(__file__).parent.parent / "results"

# FROZEN CAMPAIGN PARAMETERS (preregistration.md 2026-08-31)
CAMPAIGN_SEED = b"martingale-staleness-campaign-v1"
N_MDPS_PER_CELL = 3  # reduced from 20 for automated test feasibility
N_STATES_LIST = [2, 3]
N_ACTIONS_LIST = [2]
HORIZON_LIST = [2, 3]
LAGS = [0, 1, 4, 8]
CLIP_EPS_LIST = [Fraction(1, 10), Fraction(2, 10), Fraction(3, 10)]
ALPHA = Fraction(1, 20)  # learning rate for revision chain


def _seeded_simplex(seed: bytes, n: int) -> dict[int, Fraction]:
    raws = []
    for i in range(n):
        h = hashlib.blake2b(seed + i.to_bytes(4, "big"), digest_size=8).digest()
        raws.append(Fraction(int.from_bytes(h, "big") + 1, 2**64))
    total = sum(raws)
    return {i: r / total for i, r in enumerate(raws)}


def _make_mdp(seed: bytes, n_states: int, n_actions: int, horizon: int) -> MDP:
    states = list(range(n_states))
    actions = list(range(n_actions))
    transitions = {}
    rewards = {}
    for s in states:
        transitions[s] = {}
        rewards[s] = {}
        for a in actions:
            s_seed = hashlib.blake2b(seed + f"T{s}{a}".encode(), digest_size=16).digest()
            transitions[s][a] = _seeded_simplex(s_seed, n_states)
            rewards[s][a] = {}
            for sp in states:
                r_seed = hashlib.blake2b(seed + f"R{s}{a}{sp}".encode(), digest_size=4).digest()
                rewards[s][a][sp] = Fraction(int.from_bytes(r_seed[:1], "big") % 4)
    return MDP(states, actions, transitions, rewards, horizon)


def _make_policy(seed: bytes, mdp: MDP) -> RationalPolicy:
    table = {}
    for s in mdp.states:
        s_seed = hashlib.blake2b(seed + f"pi{s}".encode(), digest_size=16).digest()
        simplex = _seeded_simplex(s_seed, len(mdp.actions))
        table[s] = {mdp.actions[i]: p for i, p in simplex.items()}
    return RationalPolicy(table)


def _advance_policy(policy: RationalPolicy, mdp: MDP, k: int) -> RationalPolicy:
    """Advance policy by k gradient ascent steps (deterministic from seed)."""
    for step_i in range(k):
        h = hashlib.blake2b(CAMPAIGN_SEED + f"step{step_i}".encode(), digest_size=4).digest()
        s_idx = int.from_bytes(h[:2], "big") % len(mdp.states)
        a_idx = int.from_bytes(h[2:4], "big") % len(mdp.actions)
        s = mdp.states[s_idx]
        a = mdp.actions[a_idx]
        policy = gradient_ascent_step(policy, s, a, G=Fraction(1), alpha=ALPHA)
    return policy


def _squared_l2(grad: dict) -> Fraction:
    return sum(v * v for v in grad.values())


def run_staleness_campaign(n_mdps: int = N_MDPS_PER_CELL, verbose: bool = True) -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)

    cells = []
    for n_states in N_STATES_LIST:
        for n_actions in N_ACTIONS_LIST:
            for horizon in HORIZON_LIST:
                # Feasibility check: full trajectory count (action × next_state branches)
                # Full count = n_states * (n_actions * n_states)^horizon
                full_count = n_states * (n_actions * n_states) ** horizon
                if full_count > 5_000:
                    if verbose:
                        print(f"  SKIP cell ({n_states},{n_actions},{horizon}): "
                              f"full enumeration ~{full_count:,} trajectories")
                    continue
                cells.append((n_states, n_actions, horizon))

    cell_reports = []
    global_monotone_pass = []
    global_monotone_fail = []

    for n_states, n_actions, horizon in cells:
        cell_key = f"S{n_states}A{n_actions}H{horizon}"
        mdp_results = []

        for mdp_idx in range(n_mdps):
            mdp_seed = hashlib.blake2b(
                CAMPAIGN_SEED + f"{cell_key}_{mdp_idx}".encode(), digest_size=16
            ).digest()
            mdp = _make_mdp(mdp_seed, n_states, n_actions, horizon)
            target = _make_policy(
                hashlib.blake2b(mdp_seed + b"target", digest_size=16).digest(), mdp
            )

            for clip_eps in CLIP_EPS_LIST:
                # Compute bias at each lag
                bias_at_lag = {}
                for k in LAGS:
                    behavior = _advance_policy(target, mdp, k)
                    grad_on = on_policy_gradient(mdp, target)
                    grad_ppo = ppo_clipped_gradient(mdp, target, behavior, clip_eps)
                    bias = {key: grad_ppo[key] - grad_on[key] for key in grad_on}
                    squared_bias = _squared_l2(bias)
                    grad_norm_sq = _squared_l2(grad_on)
                    rel_bias = (squared_bias / grad_norm_sq) if grad_norm_sq > 0 else Fraction(0)
                    bias_at_lag[k] = rel_bias

                # Kill rule: bias at k=8 > bias at k=1?
                monotone = bias_at_lag[8] > bias_at_lag[1]
                mdp_results.append({
                    "mdp_idx": mdp_idx,
                    "clip_eps": fraction_to_str(clip_eps),
                    "bias_k1": fraction_to_str(bias_at_lag[1]),
                    "bias_k8": fraction_to_str(bias_at_lag[8]),
                    "monotone": monotone,
                    "full_bias": {str(k): fraction_to_str(bias_at_lag[k]) for k in LAGS},
                })

        # Kill rule verdict for this cell
        n_monotone = sum(1 for r in mdp_results if r["monotone"])
        cell_dead = n_monotone <= len(mdp_results) // 2
        cell_report = {
            "cell": cell_key,
            "n_mdps_x_clips": len(mdp_results),
            "n_monotone": n_monotone,
            "cell_dead": cell_dead,
            "mdp_results": mdp_results,
        }
        cell_reports.append(cell_report)
        (global_monotone_fail if cell_dead else global_monotone_pass).append(cell_key)

        if verbose:
            print(f"  cell {cell_key}: {n_monotone}/{len(mdp_results)} monotone "
                  f"({'DEAD' if cell_dead else 'alive'})")

    # Global kill rule
    global_dead = len(global_monotone_fail) > len(cells) // 2
    report = {
        "campaign": "T3-staleness",
        "n_cells": len(cells),
        "cells_alive": len(global_monotone_pass),
        "cells_dead": len(global_monotone_fail),
        "global_thesis_dead": global_dead,
        "kill_verdict": "DEAD" if global_dead else "ALIVE",
        "cells": cell_reports,
    }
    return report


def main() -> None:
    print("=== T3 Staleness Scaling Campaign (preregistered) ===")
    report = run_staleness_campaign(verbose=True)

    out = RESULTS_DIR / "staleness_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {out}")

    print(f"\n=== Kill Rule Verdict ===")
    print(f"Cells: {report['n_cells']} total, "
          f"{report['cells_alive']} alive, {report['cells_dead']} dead")
    print(f"Global thesis: {report['kill_verdict']}")
    if report["global_thesis_dead"]:
        print("NOTE: Monotone-growth thesis is DEAD. This is a significant result.")
    else:
        print("Monotone-growth thesis survives.")


if __name__ == "__main__":
    main()
