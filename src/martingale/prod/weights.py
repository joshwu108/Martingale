"""
prod/weights.py — Float32/float64 IS weight computation for production training loops.

Provides:
  - compute_is_weights: log-prob subtraction + exp (standard production path)
  - clip_is_weights: PPO-style clipping with gradient flow
  - compute_is_weights_exact_shadow: optional shadow validation against exact Fractions

All tensor ops are differentiable; gradients flow through for policy gradient updates.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

import torch


@dataclass
class IsWeightResult:
    weights: torch.Tensor      # clipped or raw IS weights
    log_ratios: torch.Tensor   # log(pi/b) before exp


def compute_is_weights(
    log_pi: torch.Tensor,
    log_b: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    return_named: bool = False,
) -> torch.Tensor | IsWeightResult:
    """
    Compute IS weights w = π(a)/b(a) via log-prob subtraction + exp.

    Args:
        log_pi: log-probabilities under target policy, shape (...)
        log_b:  log-probabilities under behavior policy, same shape
        dtype:  output dtype (float32 or float64)
        return_named: if True, return IsWeightResult with weights + log_ratios

    Returns:
        Tensor of IS weights, or IsWeightResult if return_named=True.
    """
    log_pi = log_pi.to(dtype)
    log_b = log_b.to(dtype)
    log_ratios = log_pi - log_b
    weights = torch.exp(log_ratios)
    if return_named:
        return IsWeightResult(weights=weights, log_ratios=log_ratios)
    return weights


def clip_is_weights(
    weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Clip IS weights to [1-eps, 1+eps] — PPO-style.

    Differentiable: gradients flow through the unclipped region.
    """
    return torch.clamp(weights, min=1.0 - eps, max=1.0 + eps)


def compute_is_weights_exact_shadow(
    log_pi: torch.Tensor,
    log_b: torch.Tensor,
    exact_pi: list[Fraction],
    exact_b: list[Fraction],
    clip_eps: float,
) -> dict:
    """
    Shadow validation: compute IS weights in float AND in exact Fraction arithmetic,
    then report any cases where the two disagree about clip boundary activation.

    This is the production analog of the T4 boundary flip study — can be run
    periodically on a sample of trajectories to detect silent precision loss.

    Returns a dict with:
      - n_checked: number of (pi, b) pairs checked
      - n_boundary_flips: cases where float and exact disagree on clip side
      - flip_indices: indices where flips occurred
    """
    weights_float = compute_is_weights(log_pi, log_b)
    lo_f = 1.0 - clip_eps
    hi_f = 1.0 + clip_eps
    lo_e = Fraction(1) - Fraction(clip_eps).limit_denominator(10**9)
    hi_e = Fraction(1) + Fraction(clip_eps).limit_denominator(10**9)

    flip_indices = []
    for i, (pi_e, b_e) in enumerate(zip(exact_pi, exact_b)):
        exact_r = pi_e / b_e
        float_r = weights_float[i].item()
        exact_clipped = exact_r < lo_e or exact_r > hi_e
        float_clipped = float_r < lo_f or float_r > hi_f
        if exact_clipped != float_clipped:
            flip_indices.append(i)

    return {
        "n_checked": len(exact_pi),
        "n_boundary_flips": len(flip_indices),
        "flip_indices": flip_indices,
    }
