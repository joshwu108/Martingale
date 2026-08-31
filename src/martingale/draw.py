"""
draw.py — keyed BLAKE2b uniform draws over exact rational simplexes.

Action selection uses rejection sampling with a keyed BLAKE2b stream.
Key: (seed, actor_id, episode, step) — fully deterministic and reproducible.
The draw integer and rejection count are recorded in the ledger to allow
independent re-derivation of the chosen action by the checker.

Explicitly NOT a security boundary (see docs/nonclaims.md).
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional


@dataclass(frozen=True)
class DrawResult:
    """Result of a single action draw."""
    action: int
    draw_integer: int       # First accepted draw integer (from rejection sampling)
    rejection_count: int    # Number of rejected draws before acceptance


def _make_key(seed: bytes, actor_id: int, episode: int, step: int) -> bytes:
    """
    Construct a 32-byte BLAKE2b key from the draw context.

    Combines seed, actor_id, episode, step deterministically.
    Uses struct packing to produce a fixed-length byte string.
    """
    ctx = struct.pack(">QQQ", actor_id & 0xFFFFFFFFFFFFFFFF,
                      episode & 0xFFFFFFFFFFFFFFFF,
                      step & 0xFFFFFFFFFFFFFFFF)
    # Derive a 32-byte key: BLAKE2b of seed + context (no-key mode → full output)
    return hashlib.blake2b(seed + ctx, digest_size=32).digest()


def _draw_uniform_integer(key: bytes, counter: int, n_bits: int = 256) -> int:
    """
    Draw a uniform integer in [0, 2^n_bits) using a keyed BLAKE2b stream.

    counter is incremented with each rejection to produce a fresh integer.
    Returns a non-negative integer.
    """
    ctr_bytes = struct.pack(">Q", counter & 0xFFFFFFFFFFFFFFFF)
    digest = hashlib.blake2b(ctr_bytes, digest_size=32, key=key).digest()
    return int.from_bytes(digest, byteorder="big")


def draw_action(
    probs: dict[int, Fraction],
    seed: bytes,
    actor_id: int,
    episode: int,
    step: int,
) -> DrawResult:
    """
    Draw an action from the simplex defined by probs using keyed BLAKE2b
    rejection sampling.

    Algorithm:
    1. Build cumulative probability boundaries over the simplex as exact Fractions.
    2. Generate a uniform integer u in [0, 2^256).
    3. Map u to a rational in [0, 1): u_rat = Fraction(u, 2^256).
    4. Find the action a such that CDF(a-1) <= u_rat < CDF(a).
    5. If no action matches (shouldn't happen with valid simplex), retry with
       incremented counter (rejection sampling).

    The draw_integer is u (the raw 256-bit integer).
    The rejection_count is the number of failed attempts before acceptance.

    Parameters
    ----------
    probs : dict mapping action → Fraction probability (must sum to 1).
    seed  : bytes, shared seed across all draws (not a security boundary).
    actor_id, episode, step : integers identifying the draw context.
    """
    key = _make_key(seed, actor_id, episode, step)
    denom = 2 ** 256  # fixed denominator for the uniform draw

    # Build sorted action list and cumulative boundaries
    actions_sorted = sorted(probs.keys())
    cumulative = Fraction(0)
    boundaries = []  # (action, lower_bound_exclusive, upper_bound_inclusive)
    for a in actions_sorted:
        lower = cumulative
        cumulative += probs[a]
        if probs[a] > 0:
            boundaries.append((a, lower, cumulative))

    # Handle degenerate case: only one action with non-zero probability
    if len(boundaries) == 1:
        return DrawResult(
            action=boundaries[0][0],
            draw_integer=0,
            rejection_count=0,
        )

    counter = 0
    while True:
        u = _draw_uniform_integer(key, counter)
        u_rat = Fraction(u, denom)

        for action, lower, upper in boundaries:
            if lower <= u_rat < upper:
                return DrawResult(
                    action=action,
                    draw_integer=u,
                    rejection_count=counter,
                )

        # u_rat == 1 exactly (u == denom, impossible for 256-bit draw) or
        # landed in a gap (shouldn't happen with valid simplex).
        # Increment counter and retry.
        counter += 1
        if counter > 1000:
            raise RuntimeError(
                f"draw_action: exceeded 1000 rejections. "
                f"This indicates a bug in the simplex or draw logic."
            )
