"""Tests for rational.py — exact conversions and serialization."""
import pytest
from fractions import Fraction
from martingale.rational import (
    to_fraction,
    from_fraction_str,
    fraction_to_str,
    assert_exact,
    assert_no_inf_nan,
    check_simplex,
)


class TestToFraction:
    def test_int_input(self):
        assert to_fraction(3) == Fraction(3)

    def test_fraction_passthrough(self):
        f = Fraction(2, 7)
        assert to_fraction(f) is f

    def test_tuple_input(self):
        assert to_fraction((3, 7)) == Fraction(3, 7)

    def test_string_fraction(self):
        assert to_fraction("3/7") == Fraction(3, 7)

    def test_string_integer(self):
        assert to_fraction("5") == Fraction(5)

    def test_float_rejected(self):
        with pytest.raises(TypeError, match="float"):
            to_fraction(0.5)

    def test_zero_denominator_rejected(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            to_fraction((1, 0))


class TestFractionSerialization:
    def test_round_trip(self):
        for num, den in [(1, 3), (7, 12), (0, 1), (100, 1), (5, 5)]:
            f = Fraction(num, den)
            s = fraction_to_str(f)
            assert from_fraction_str(s) == f

    def test_reduced_form(self):
        # 6/4 should serialize as "3/2"
        s = fraction_to_str(Fraction(6, 4))
        assert s == "3/2"

    def test_zero(self):
        assert fraction_to_str(Fraction(0)) == "0/1"

    def test_whole_number(self):
        assert fraction_to_str(Fraction(5)) == "5/1"

    def test_from_str_format(self):
        assert from_fraction_str("3/7") == Fraction(3, 7)
        assert from_fraction_str("0/1") == Fraction(0)


class TestAssertExact:
    def test_fraction_passes(self):
        assert_exact(Fraction(1, 3))  # no exception

    def test_int_passes(self):
        assert_exact(5)

    def test_float_raises(self):
        with pytest.raises(TypeError):
            assert_exact(0.5)

    def test_inf_raises(self):
        with pytest.raises(ValueError):
            assert_no_inf_nan(float("inf"))

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            assert_no_inf_nan(float("nan"))


class TestCheckSimplex:
    def test_valid_simplex(self):
        probs = {0: Fraction(1, 3), 1: Fraction(1, 3), 2: Fraction(1, 3)}
        check_simplex(probs)  # no exception

    def test_sum_not_one_raises(self):
        probs = {0: Fraction(1, 3), 1: Fraction(1, 4)}
        with pytest.raises(ValueError, match="sum"):
            check_simplex(probs)

    def test_negative_prob_raises(self):
        probs = {0: Fraction(-1, 3), 1: Fraction(4, 3)}
        with pytest.raises(ValueError, match="negative"):
            check_simplex(probs)

    def test_empty_simplex_raises(self):
        with pytest.raises(ValueError):
            check_simplex({})
