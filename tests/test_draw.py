"""Tests for draw.py — keyed BLAKE2b draws over rational simplexes."""
import pytest
from fractions import Fraction
from martingale.draw import draw_action, DrawResult


class TestDrawAction:
    def _uniform_probs(self, n: int) -> dict[int, Fraction]:
        return {i: Fraction(1, n) for i in range(n)}

    def test_result_structure(self):
        probs = self._uniform_probs(3)
        result = draw_action(probs, seed=b"test_seed", actor_id=0, episode=0, step=0)
        assert isinstance(result, DrawResult)
        assert result.action in probs
        assert isinstance(result.draw_integer, int)
        assert result.draw_integer >= 0
        assert isinstance(result.rejection_count, int)
        assert result.rejection_count >= 0

    def test_action_in_support(self):
        probs = {0: Fraction(1, 4), 1: Fraction(3, 4)}
        for i in range(20):
            result = draw_action(probs, seed=b"s", actor_id=0, episode=i, step=0)
            assert result.action in {0, 1}

    def test_deterministic_with_same_key(self):
        """Same key always produces same result."""
        probs = self._uniform_probs(4)
        r1 = draw_action(probs, seed=b"fixed", actor_id=1, episode=5, step=2)
        r2 = draw_action(probs, seed=b"fixed", actor_id=1, episode=5, step=2)
        assert r1.action == r2.action
        assert r1.draw_integer == r2.draw_integer
        assert r1.rejection_count == r2.rejection_count

    def test_different_steps_different_result(self):
        """Different (actor, episode, step) keys produce independent draws."""
        probs = self._uniform_probs(2)
        results = [
            draw_action(probs, seed=b"s", actor_id=0, episode=0, step=t).action
            for t in range(100)
        ]
        # Not all the same (would be astronomically unlikely)
        assert len(set(results)) > 1

    def test_deterministic_action_draws_correctly(self):
        """A deterministic policy (prob=1 for one action) always picks that action."""
        probs = {0: Fraction(0), 1: Fraction(1), 2: Fraction(0)}
        for i in range(10):
            result = draw_action(probs, seed=b"s", actor_id=0, episode=i, step=0)
            assert result.action == 1

    def test_draw_integer_nonnegative(self):
        probs = self._uniform_probs(3)
        for i in range(20):
            result = draw_action(probs, seed=b"test", actor_id=0, episode=0, step=i)
            assert result.draw_integer >= 0

    def test_zero_prob_action_never_selected(self):
        """Actions with probability 0 are never selected."""
        probs = {0: Fraction(1, 2), 1: Fraction(0), 2: Fraction(1, 2)}
        for i in range(50):
            result = draw_action(probs, seed=b"s", actor_id=0, episode=i, step=0)
            assert result.action != 1

    def test_reproducibility_cross_call(self):
        """Draw is fully reproducible: given (seed, actor, episode, step), result is fixed."""
        probs = {0: Fraction(1, 3), 1: Fraction(1, 3), 2: Fraction(1, 3)}
        # Draw at many keys, record results
        records = {}
        for ep in range(5):
            for step in range(5):
                r = draw_action(probs, seed=b"repro", actor_id=0, episode=ep, step=step)
                records[(ep, step)] = (r.action, r.draw_integer, r.rejection_count)
        # Replay and verify
        for ep in range(5):
            for step in range(5):
                r = draw_action(probs, seed=b"repro", actor_id=0, episode=ep, step=step)
                assert (r.action, r.draw_integer, r.rejection_count) == records[(ep, step)]
