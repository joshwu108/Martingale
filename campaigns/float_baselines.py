"""
campaigns/float_baselines.py — T4 float defendant implementations.

Float defendants:
  - float32 log-prob subtraction + exp + clip (numpy path)
  - float64 log-prob subtraction + exp + clip (numpy path)

Each defendant takes an exact rational IS ratio and a clip epsilon,
and computes the float approximation for comparison.

Note: PyTorch defendant is included if torch is available.
"""
from __future__ import annotations

import struct
from fractions import Fraction
from typing import Literal


FloatVariant = Literal["float32", "float64"]


def compute_float_is_ratio(
    pi_prob: Fraction,
    b_prob: Fraction,
    variant: FloatVariant,
) -> float:
    """
    Compute IS ratio π(a)/b(a) in floating point.

    Uses the log-prob subtraction + exp path:
      float_ratio = exp(log(π(a)) - log(b(a)))

    This is the production-style computation path.
    """
    import math
    pi_f = float(pi_prob)
    b_f = float(b_prob)
    if variant == "float32":
        import struct
        # Simulate float32 precision by round-tripping through float32
        pi_f32 = struct.unpack('f', struct.pack('f', pi_f))[0]
        b_f32 = struct.unpack('f', struct.pack('f', b_f))[0]
        log_ratio = math.log(pi_f32) - math.log(b_f32)
        return struct.unpack('f', struct.pack('f', math.exp(log_ratio)))[0]
    else:  # float64
        return math.exp(math.log(pi_f) - math.log(b_f))


def clip_ratio(ratio: float, clip_eps: float, variant: FloatVariant) -> float:
    """Clip IS ratio to [1-eps, 1+eps] in floating point."""
    lo = 1.0 - clip_eps
    hi = 1.0 + clip_eps
    if variant == "float32":
        lo = struct.unpack('f', struct.pack('f', lo))[0]
        hi = struct.unpack('f', struct.pack('f', hi))[0]
        clipped = max(lo, min(hi, ratio))
        return struct.unpack('f', struct.pack('f', clipped))[0]
    return max(lo, min(hi, ratio))


def exact_is_ratio(pi_prob: Fraction, b_prob: Fraction) -> Fraction:
    """Exact IS ratio as a Fraction."""
    return pi_prob / b_prob


def clip_ratio_exact(ratio: Fraction, clip_eps: Fraction) -> Fraction:
    """Clip IS ratio to [1-eps, 1+eps] as exact Fraction."""
    lo = Fraction(1) - clip_eps
    hi = Fraction(1) + clip_eps
    if ratio < lo:
        return lo
    if ratio > hi:
        return hi
    return ratio


def detect_boundary_flip(
    pi_prob: Fraction,
    b_prob: Fraction,
    clip_eps: Fraction,
    variant: FloatVariant,
) -> dict:
    """
    Detect if float clip-boundary classification disagrees with exact arithmetic.

    Returns a dict with:
      - exact_ratio: Fraction IS ratio
      - float_ratio: float IS ratio (variant precision)
      - exact_clipped: Fraction clipped ratio
      - float_clipped: float clipped ratio
      - flip: bool — True if float and exact disagree about clip activation
      - flip_type: "exact_clipped_float_not" | "float_clipped_exact_not" | None
    """
    exact_r = exact_is_ratio(pi_prob, b_prob)
    float_r = compute_float_is_ratio(pi_prob, b_prob, variant)

    lo = Fraction(1) - clip_eps
    hi = Fraction(1) + clip_eps
    lo_f = float(lo)
    hi_f = float(hi)
    if variant == "float32":
        lo_f = struct.unpack('f', struct.pack('f', lo_f))[0]
        hi_f = struct.unpack('f', struct.pack('f', hi_f))[0]

    exact_active = exact_r < lo or exact_r > hi  # clipping is active
    float_active = float_r < lo_f or float_r > hi_f

    flip = exact_active != float_active
    if flip:
        if exact_active and not float_active:
            flip_type = "exact_clipped_float_not"
        else:
            flip_type = "float_clipped_exact_not"
    else:
        flip_type = None

    return {
        "exact_ratio": str(exact_r),
        "float_ratio": float_r,
        "exact_clipped": str(clip_ratio_exact(exact_r, clip_eps)),
        "float_clipped": clip_ratio(float_r, float(clip_eps), variant),
        "flip": flip,
        "flip_type": flip_type,
        "variant": variant,
        "clip_eps": str(clip_eps),
    }


def cumulative_product_divergence(
    per_step_pi: list[Fraction],
    per_step_b: list[Fraction],
    clip_eps: Fraction,
    variant: FloatVariant,
) -> dict:
    """
    Compute exact vs float IS ratio for a multi-step trajectory.

    For long trajectories (H=16,64,256), float cumulative products can
    underflow/overflow, causing clip-boundary disagreements.
    """
    import math

    # Exact cumulative ratio
    exact_prod = Fraction(1)
    for pi_a, b_a in zip(per_step_pi, per_step_b):
        exact_prod *= pi_a / b_a

    # Float cumulative ratio via log-sum
    log_ratio = 0.0
    for pi_a, b_a in zip(per_step_pi, per_step_b):
        pi_f = float(pi_a)
        b_f = float(b_a)
        if variant == "float32":
            pi_f = struct.unpack('f', struct.pack('f', pi_f))[0]
            b_f = struct.unpack('f', struct.pack('f', b_f))[0]
        log_ratio += math.log(pi_f) - math.log(b_f)

    float_ratio = math.exp(log_ratio)
    if variant == "float32":
        float_ratio = struct.unpack('f', struct.pack('f', float_ratio))[0]

    lo = Fraction(1) - clip_eps
    hi = Fraction(1) + clip_eps
    lo_f = float(lo)
    hi_f = float(hi)

    exact_active = exact_prod < lo or exact_prod > hi
    float_active = float_ratio < lo_f or float_ratio > hi_f
    flip = exact_active != float_active

    return {
        "horizon": len(per_step_pi),
        "exact_ratio": str(exact_prod),
        "float_ratio": float_ratio,
        "flip": flip,
        "variant": variant,
        "clip_eps": str(clip_eps),
    }
