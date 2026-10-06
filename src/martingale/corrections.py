"""
corrections.py — torch reference implementations of per-token off-policy corrections.

Pure functions over tensors of log-probabilities. Every weight function here
mirrors an exact-arithmetic ground truth in `martingale.exact.defendants`
(`_truncate`, `_mask`, `_ppo`, `_cispo`) and is tested against it; see
tests/test_corrections.py. This is the tested reference, not the fast path.

Conventions:
  r = pi(a|s) / b(a|s) is the per-token ratio, computed as exp(log_pi - log_b).
  `advantage` plays the role of G in the exact bench (one scalar per token).
  `dtype` is the working precision; it defaults to float32, which is what
  production loops use and what `clip_flips` audits.

torch is imported lazily inside each function so that this module imports
without torch installed (the exact core has no torch dependency).
"""
from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # torch is optional at import time
    import torch
    from torch import Tensor

_EXACT_EPS_MAX_DENOMINATOR = 10**9   # matches prod/weights.py


# ---- parameter validation (mirrors exact/defendants.py) ----------------------

def _check_cap(cap: float) -> None:
    if not cap > 0:
        raise ValueError("cap must be > 0")


def _check_mask(lo: float, hi: float) -> None:
    if not (0 <= lo <= hi):
        raise ValueError("need 0 <= lo <= hi")


def _check_eps(eps: float) -> None:
    if not (0 <= eps < 1):
        raise ValueError("need 0 <= eps < 1")


def _check_cispo(eps_lo: float, eps_hi: float) -> None:
    if not (0 <= eps_lo < 1) or eps_hi < 0:
        raise ValueError("need 0 <= eps_lo < 1 and eps_hi >= 0")


def _resolve_dtype(dtype: torch.dtype | None) -> torch.dtype:
    import torch
    return torch.float32 if dtype is None else dtype


# ---- ratios -------------------------------------------------------------------

def log_ratio(log_pi: Tensor, log_b: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    """log r = log_pi - log_b, computed in `dtype` (default float32)."""
    dt = _resolve_dtype(dtype)
    return log_pi.to(dt) - log_b.to(dt)


def is_weight(log_pi: Tensor, log_b: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    """Plain importance weight r = exp(log_pi - log_b). Exact analog: `_identity`."""
    import torch
    return torch.exp(log_ratio(log_pi, log_b, dtype))


def truncated_is(log_pi: Tensor, log_b: Tensor, cap: float,
                 dtype: torch.dtype | None = None) -> Tensor:
    """Truncated IS, min(r, cap). TIS of Yao et al. 2025 ("Your Efficient RL
    Framework Secretly Brings You Off-Policy RL Training"). Exact analog: `_truncate`."""
    _check_cap(cap)
    w = is_weight(log_pi, log_b, dtype)
    return w.clamp(max=cap)


def masked_is(log_pi: Tensor, log_b: Tensor, lo: float, hi: float,
              dtype: torch.dtype | None = None) -> Tensor:
    """Masked IS: r if lo <= r <= hi else 0 (MIS of Yao et al. 2025; IcePop-style
    token masking, Ling team 2025). Exact analog: `_mask`."""
    import torch
    _check_mask(lo, hi)
    w = is_weight(log_pi, log_b, dtype)
    keep = (w >= lo) & (w <= hi)
    return torch.where(keep, w, torch.zeros_like(w))


def ppo_clip_weight(log_pi: Tensor, log_b: Tensor, advantage: Tensor, eps: float,
                    dtype: torch.dtype | None = None) -> Tensor:
    """Weight on the score term implied by the PPO clipped surrogate (Schulman et
    al. 2017): r when the unclipped branch of min(r A, clip(r) A) is active, 0 when
    the constant clipped branch is. Exact ties take the unclipped branch, matching
    `_ppo` (torch.minimum would split the gradient at a tie)."""
    import torch
    _check_eps(eps)
    w = is_weight(log_pi, log_b, dtype)
    clipped = _ppo_clipped_active(w, advantage.to(w.dtype), eps)
    return torch.where(clipped, torch.zeros_like(w), w)


def _ppo_clipped_active(w: Tensor, advantage: Tensor, eps: float) -> Tensor:
    """Boolean mask: the clipped (constant) branch of the PPO objective is active."""
    return ((advantage > 0) & (w > 1 + eps)) | ((advantage < 0) & (w < 1 - eps))


def cispo_weight(log_pi: Tensor, log_b: Tensor, eps_lo: float, eps_hi: float,
                 dtype: torch.dtype | None = None) -> Tensor:
    """CISPO weight sg(clip(r, 1-eps_lo, 1+eps_hi)) of MiniMax-M1 (2025): clipped
    but detached, so every token keeps a gradient. Exact analog: `_cispo`."""
    _check_cispo(eps_lo, eps_hi)
    w = is_weight(log_pi, log_b, dtype)
    return w.clamp(min=1 - eps_lo, max=1 + eps_hi).detach()


# ---- losses -------------------------------------------------------------------

def ppo_surrogate_loss(log_pi: Tensor, log_b: Tensor, advantage: Tensor,
                       eps: float) -> Tensor:
    """PPO clipped surrogate, -mean(min(r A, clip(r, 1-eps, 1+eps) A)), with gradient
    through log_pi only (Schulman et al. 2017). The branch is chosen by
    `_ppo_clipped_active` rather than torch.minimum so that an exact tie keeps the
    unclipped branch, as in the exact bench; the value is identical either way."""
    import torch
    _check_eps(eps)
    r = torch.exp(log_pi - log_b.detach())
    a = advantage.to(r.dtype).detach()
    clipped = _ppo_clipped_active(r.detach(), a, eps)
    clipped_term = r.detach().clamp(min=1 - eps, max=1 + eps) * a
    return -torch.where(clipped, clipped_term, r * a).mean()


def cispo_loss(log_pi: Tensor, log_b: Tensor, advantage: Tensor,
               eps_lo: float, eps_hi: float) -> Tensor:
    """CISPO loss -mean(sg(clip(r)) * A * log_pi) of MiniMax-M1 (2025)."""
    _check_cispo(eps_lo, eps_hi)
    w = cispo_weight(log_pi.detach(), log_b.detach(), eps_lo, eps_hi, dtype=log_pi.dtype)
    a = advantage.to(log_pi.dtype).detach()
    return -(w * a * log_pi).mean()


# ---- exact shadow ---------------------------------------------------------------

def _exact_eps(eps: float | Fraction) -> Fraction:
    if isinstance(eps, Fraction):
        return eps
    return Fraction(eps).limit_denominator(_EXACT_EPS_MAX_DENOMINATOR)


def _exact_side(r: Fraction, lo: Fraction, hi: Fraction) -> int:
    """-1 below the clip window, 0 inside (inclusive), +1 above."""
    return -1 if r < lo else 1 if r > hi else 0


def _float_sides(w: Tensor, eps: float) -> list[int]:
    """Same classification done the way a training loop does it: the float32 ratio
    compared against the scalar bounds in float32 (never promoted to float64)."""
    below, above = w < (1.0 - eps), w > (1.0 + eps)
    return (above.int() - below.int()).tolist()


def clip_flips(log_pi: Tensor, log_b: Tensor, exact_pi: list[Fraction],
               exact_b: list[Fraction], eps: float | Fraction) -> dict:
    """Production analog of the T4 boundary-flip study: indices where the float32
    ratio exp(log_pi - log_b) lands on a different side of the PPO clip window
    [1-eps, 1+eps] than the exact rational pi/b. A float `eps` is rationalised with
    limit_denominator(1e9), as in prod/weights.py; pass a Fraction to avoid that.

    Returns {"n_checked", "n_flips", "flip_indices", "sides"} where "sides" maps a
    flipped index to (exact_side, float_side) with sides in {-1, 0, +1}."""
    import torch
    _check_eps(float(eps))
    if len(exact_pi) != len(exact_b) or len(exact_pi) != log_pi.numel():
        raise ValueError("exact_pi, exact_b and log_pi must have the same length")
    e = _exact_eps(eps)
    w = is_weight(log_pi.reshape(-1), log_b.reshape(-1), dtype=torch.float32)
    float_sides = _float_sides(w, float(eps))
    sides: dict[int, tuple[int, int]] = {}
    for i, (pi_e, b_e) in enumerate(zip(exact_pi, exact_b)):
        if b_e == 0:
            raise ValueError(f"exact_b[{i}] is 0; ratio undefined")
        exact_side = _exact_side(pi_e / b_e, 1 - e, 1 + e)
        if exact_side != float_sides[i]:
            sides[i] = (exact_side, float_sides[i])
    return {"n_checked": len(exact_pi), "n_flips": len(sides),
            "flip_indices": sorted(sides), "sides": sides}
