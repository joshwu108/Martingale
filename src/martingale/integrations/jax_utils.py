"""
integrations/jax_utils.py — Pure-function IS weight utilities (JAX/NumPy compatible).

These functions operate on plain Python lists/floats, making them usable from
JAX, NumPy, or any framework without PyTorch dependency.

For JAX JIT compilation, wrap with jax.jit after importing.
"""
from __future__ import annotations

import math
from typing import List


def compute_is_weights_numpy(
    log_pi: List[float],
    log_b: List[float],
) -> List[float]:
    """Compute IS weights w_i = exp(log_pi_i - log_b_i)."""
    return [math.exp(lp - lb) for lp, lb in zip(log_pi, log_b)]


def clip_is_weights_numpy(
    weights: List[float],
    eps: float,
) -> List[float]:
    """Clip IS weights to [1-eps, 1+eps]."""
    lo, hi = 1.0 - eps, 1.0 + eps
    return [max(lo, min(hi, w)) for w in weights]


def ppo_loss_numpy(
    log_pi: List[float],
    log_b: List[float],
    advantages: List[float],
    clip_eps: float,
) -> float:
    """
    Compute the PPO clipped surrogate loss (scalar).

    loss = -mean(min(r * A, clip(r, 1-e, 1+e) * A))
    where r = exp(log_pi - log_b).
    """
    weights = compute_is_weights_numpy(log_pi, log_b)
    clipped = clip_is_weights_numpy(weights, clip_eps)
    terms = [
        min(w * adv, cw * adv)
        for w, cw, adv in zip(weights, clipped, advantages)
    ]
    return -sum(terms) / len(terms) if terms else 0.0
