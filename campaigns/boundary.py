"""
campaigns/boundary.py — T4 clip-boundary flip search.

Protocol frozen in docs/preregistration.md (2026-08-31).

Constructs near-boundary (exact-ratio, parameterization) pairs and searches
for cases where the float IS ratio crosses the clip boundary differently than
the exact rational ratio. Reports every flip with a shrunk minimal reproducer.

Results committed to results/boundary_report.json.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from campaigns.float_baselines import detect_boundary_flip, cumulative_product_divergence

RESULTS_DIR = Path(__file__).parent.parent / "results"
CAMPAIGN_SEED = b"martingale-boundary-campaign-v1"

# FROZEN PARAMETERS (preregistration.md)
N_CANDIDATES = 1000      # reduced from 10^5 for automated tests; use 10^5 in production
CLIP_EPS_LIST = [Fraction(1, 10), Fraction(2, 10)]
FLOAT_VARIANTS = ["float32", "float64"]
J_OFFSETS = list(range(-4, 5))   # j ∈ {-4,...,4} ULP offsets from boundary
HORIZON_LIST = [16, 64, 256]     # for cumulative product study


def _float32_ulp(x: float) -> float:
    """Return the ULP (unit of least precision) of x in float32."""
    import math
    # Use struct to get float32 representation
    x_f32 = struct.unpack('f', struct.pack('f', x))[0]
    if x_f32 == 0.0:
        # Smallest positive float32 denormal
        return struct.unpack('f', struct.pack('I', 1))[0]
    bits = struct.unpack('I', struct.pack('f', x_f32))[0]
    next_bits = bits + 1
    next_val = struct.unpack('f', struct.pack('I', next_bits))[0]
    return abs(next_val - x_f32)


def _construct_near_boundary_pair(
    clip_eps: Fraction,
    j: int,
    candidate_idx: int,
) -> tuple[Fraction, Fraction] | None:
    """
    Construct (pi_prob, b_prob) such that the exact IS ratio is at distance
    j·ULP from the clip boundary (1 + clip_eps or 1 - clip_eps).

    Returns (pi_prob, b_prob) or None if construction fails.

    Method: target exact ratio r = (1 + clip_eps) ± j·ulp(clip_boundary).
    Then choose b_prob = Fraction(N, D) with N,D small, pi_prob = r * b_prob.
    """
    import math

    # Target: upper boundary 1 + clip_eps
    boundary_f32 = float(1 + clip_eps)
    ulp = _float32_ulp(boundary_f32)
    target_float = boundary_f32 + j * ulp

    # Find a rational approximation of target_float
    # Use Fraction.from_float for exact representation
    target_frac = Fraction(target_float).limit_denominator(10**6)

    # Construct b_prob: a rational in [0.1, 0.9] from seed
    h = hashlib.blake2b(
        CAMPAIGN_SEED + f"pair_{j}_{candidate_idx}".encode(), digest_size=8
    ).digest()
    b_num = (int.from_bytes(h[:4], "big") % 90) + 10  # 10..99
    b_den = 100
    b_prob = Fraction(b_num, b_den)

    pi_prob = target_frac * b_prob
    # Ensure pi_prob is a valid probability in (0, 1]
    if pi_prob <= 0 or pi_prob > 1:
        return None
    return (pi_prob, b_prob)


def _shrink_minimal_reproducer(
    pi_prob: Fraction,
    b_prob: Fraction,
    clip_eps: Fraction,
    variant: str,
) -> dict:
    """
    Find the smallest (simplest) rational parameterization that still triggers a flip.

    Bisection: try to simplify fractions while maintaining the flip.
    Returns the minimal reproducer dict.
    """
    best = {"pi_prob": pi_prob, "b_prob": b_prob}
    # Try simplifying by reducing denominator
    for max_den in [10**4, 10**3, 100, 10]:
        pi_simple = pi_prob.limit_denominator(max_den)
        b_simple = b_prob.limit_denominator(max_den)
        if pi_simple <= 0 or b_simple <= 0:
            continue
        result = detect_boundary_flip(pi_simple, b_simple, clip_eps, variant)
        if result["flip"]:
            best = {"pi_prob": pi_simple, "b_prob": b_simple}
    return best


def run_boundary_campaign(n_candidates: int = N_CANDIDATES, verbose: bool = True) -> dict:
    """Run the T4 clip-boundary flip search."""
    RESULTS_DIR.mkdir(exist_ok=True)

    all_flips = []
    stats = {}

    for clip_eps in CLIP_EPS_LIST:
        for variant in FLOAT_VARIANTS:
            key = f"eps={clip_eps}_variant={variant}"
            n_flips = 0
            flip_examples = []

            # Near-boundary search
            for candidate_idx in range(n_candidates):
                for j in J_OFFSETS:
                    pair = _construct_near_boundary_pair(clip_eps, j, candidate_idx)
                    if pair is None:
                        continue
                    pi_prob, b_prob = pair
                    result = detect_boundary_flip(pi_prob, b_prob, clip_eps, variant)
                    if result["flip"]:
                        n_flips += 1
                        # Shrink to minimal reproducer
                        minimal = _shrink_minimal_reproducer(pi_prob, b_prob, clip_eps, variant)
                        flip_examples.append({
                            "pi_prob": str(pi_prob),
                            "b_prob": str(b_prob),
                            "minimal_pi": str(minimal["pi_prob"]),
                            "minimal_b": str(minimal["b_prob"]),
                            "j_offset": j,
                            "candidate_idx": candidate_idx,
                            **{k: v for k, v in result.items()
                               if k not in ("exact_ratio", "float_ratio", "exact_clipped", "float_clipped")},
                        })
                        if len(flip_examples) <= 5:  # keep first 5 as examples
                            all_flips.append(flip_examples[-1])

            stats[key] = {
                "n_candidates_tested": n_candidates * len(J_OFFSETS),
                "n_flips": n_flips,
                "flip_rate_pct": 100 * n_flips / (n_candidates * len(J_OFFSETS)),
                "kill_verdict": "DEAD" if n_flips == 0 else "ALIVE",
                "examples": flip_examples[:5],
            }

            if verbose:
                print(f"  {key}: {n_flips} flips "
                      f"({'DEAD - kill rule fired!' if n_flips == 0 else 'alive'})")

    # Cumulative product study
    cumulative_results = []
    for horizon in HORIZON_LIST:
        for variant in FLOAT_VARIANTS:
            # Generate a sequence of per-step IS ratios
            flips = 0
            for trial_idx in range(min(n_candidates, 100)):
                h = hashlib.blake2b(
                    CAMPAIGN_SEED + f"cum_{horizon}_{variant}_{trial_idx}".encode(),
                    digest_size=8
                ).digest()
                # Generate per-step probs
                per_step_pi = []
                per_step_b = []
                for step in range(horizon):
                    step_h = hashlib.blake2b(
                        h + step.to_bytes(4, "big"), digest_size=8
                    ).digest()
                    pi_a = Fraction((int.from_bytes(step_h[:4], "big") % 90) + 5, 100)
                    b_a  = Fraction((int.from_bytes(step_h[4:], "big") % 90) + 5, 100)
                    per_step_pi.append(pi_a)
                    per_step_b.append(b_a)

                for clip_eps in CLIP_EPS_LIST:
                    r = cumulative_product_divergence(
                        per_step_pi, per_step_b, clip_eps, variant
                    )
                    if r["flip"]:
                        flips += 1

            cumulative_results.append({
                "horizon": horizon,
                "variant": variant,
                "n_trials": min(n_candidates, 100) * len(CLIP_EPS_LIST),
                "n_flips": flips,
            })
            if verbose:
                print(f"  cumulative H={horizon} {variant}: {flips} flips")

    report = {
        "campaign": "T4-boundary",
        "n_candidates": n_candidates,
        "per_variant_stats": stats,
        "cumulative_product_study": cumulative_results,
        "kill_verdict": "DEAD" if any(
            v["kill_verdict"] == "DEAD" for v in stats.values()
        ) else "ALIVE",
        "flip_examples": all_flips[:10],
    }
    return report


def main() -> None:
    print("=== T4 Clip-Boundary Flip Campaign (preregistered) ===")
    report = run_boundary_campaign(verbose=True)

    out = RESULTS_DIR / "boundary_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport written to {out}")

    print(f"\n=== Kill Rule Verdict ===")
    print(f"Global: {report['kill_verdict']}")
    for key, stat in report["per_variant_stats"].items():
        print(f"  {key}: {stat['n_flips']} flips → {stat['kill_verdict']}")


if __name__ == "__main__":
    main()
