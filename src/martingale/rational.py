"""
rational.py — exact conversions and reduced-fraction serialization.

All values in martingale must be fractions.Fraction or int.
Floats are rejected loudly except at explicitly declared boundaries (T4 only).
"""
from fractions import Fraction
from typing import Union


ExactNumber = Union[Fraction, int]


def to_fraction(value) -> Fraction:
    """Convert an exact value to Fraction. Rejects floats."""
    if isinstance(value, float):
        raise TypeError(
            f"float values are not permitted in exact paths; got {value!r}. "
            "Use fractions.Fraction or int."
        )
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, tuple) and len(value) == 2:
        return Fraction(value[0], value[1])
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError(f"Cannot convert {type(value).__name__!r} to Fraction: {value!r}")


def fraction_to_str(f: Fraction) -> str:
    """Serialize a Fraction as 'numerator/denominator' in reduced form."""
    f = Fraction(f)  # ensure reduced
    return f"{f.numerator}/{f.denominator}"


def from_fraction_str(s: str) -> Fraction:
    """Deserialize a 'numerator/denominator' string to Fraction."""
    if "/" in s:
        num, den = s.split("/", 1)
        return Fraction(int(num), int(den))
    return Fraction(int(s))


def assert_exact(value) -> None:
    """Raise TypeError if value is a float (not permitted in exact paths)."""
    if isinstance(value, float):
        raise TypeError(f"float not permitted in exact paths: {value!r}")


def assert_no_inf_nan(value) -> None:
    """Raise ValueError if value is inf or nan."""
    if isinstance(value, float):
        import math
        if math.isinf(value) or math.isnan(value):
            raise ValueError(f"inf/nan not permitted: {value!r}")


def check_simplex(probs: dict) -> None:
    """
    Validate that probs is a valid probability simplex:
      - non-empty
      - all values are non-negative Fractions
      - values sum to exactly 1
    Raises ValueError with a descriptive message on failure.
    """
    if not probs:
        raise ValueError("Simplex must be non-empty.")
    for action, p in probs.items():
        p = to_fraction(p)
        if p < 0:
            raise ValueError(f"negative probability for action {action}: {p}")
    total = sum(to_fraction(p) for p in probs.values())
    if total != Fraction(1):
        raise ValueError(
            f"Simplex probabilities sum to {total} (expected 1). "
            f"Values: {dict(probs)}"
        )
