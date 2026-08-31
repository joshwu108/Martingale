"""Smoke test: project imports and basic Python stdlib availability."""
import fractions
import hashlib
import multiprocessing


def test_fractions_exact_arithmetic():
    """Exact rational arithmetic works as expected."""
    a = fractions.Fraction(1, 3)
    b = fractions.Fraction(1, 6)
    assert a + b == fractions.Fraction(1, 2)
    assert a * b == fractions.Fraction(1, 18)
    assert str(a + b) == "1/2"


def test_blake2b_available():
    """BLAKE2b is available in hashlib for keyed draws."""
    h = hashlib.blake2b(b"test", digest_size=32, key=b"martingale12345!")
    assert len(h.digest()) == 32


def test_multiprocessing_spawn():
    """spawn start method is available (required for macOS + async pipeline)."""
    ctx = multiprocessing.get_context("spawn")
    assert ctx.get_start_method() == "spawn"


def test_fraction_comparison_no_float():
    """Fraction comparisons work without any float involvement."""
    p = fractions.Fraction(3, 10)
    q = fractions.Fraction(1, 10)
    clip = fractions.Fraction(2, 10)
    assert p > clip
    assert q < clip
    assert fractions.Fraction(2, 10) == clip
