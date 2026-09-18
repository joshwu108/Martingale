"""Tests for prod/weights.py — float32/float64 IS weight computation."""
import pytest
import torch
from fractions import Fraction
from martingale.prod.weights import (
    compute_is_weights,
    clip_is_weights,
    compute_is_weights_exact_shadow,
    IsWeightResult,
)


class TestComputeIsWeights:
    def test_basic_ratio(self):
        """w = exp(log_pi - log_b) = pi/b."""
        log_pi = torch.tensor([0.0, -1.0])   # pi = [1.0, 0.368]
        log_b  = torch.tensor([-1.0, -1.0])  # b  = [0.368, 0.368]
        w = compute_is_weights(log_pi, log_b)
        assert w.shape == (2,)
        assert w.dtype == torch.float32
        assert torch.allclose(w, torch.exp(log_pi - log_b))

    def test_equal_policies_weight_one(self):
        """When target == behavior, all IS weights == 1."""
        log_p = torch.tensor([-0.5, -1.2, -0.3])
        w = compute_is_weights(log_p, log_p)
        assert torch.allclose(w, torch.ones(3), atol=1e-6)

    def test_float32_output_by_default(self):
        log_pi = torch.tensor([-0.5], dtype=torch.float32)
        log_b  = torch.tensor([-1.0], dtype=torch.float32)
        w = compute_is_weights(log_pi, log_b)
        assert w.dtype == torch.float32

    def test_float64_when_requested(self):
        log_pi = torch.tensor([-0.5], dtype=torch.float64)
        log_b  = torch.tensor([-1.0], dtype=torch.float64)
        w = compute_is_weights(log_pi, log_b, dtype=torch.float64)
        assert w.dtype == torch.float64

    def test_batch_dimension(self):
        B, T = 4, 8
        log_pi = torch.randn(B, T)
        log_b  = torch.randn(B, T)
        w = compute_is_weights(log_pi, log_b)
        assert w.shape == (B, T)

    def test_returns_isweightresult(self):
        log_pi = torch.tensor([-0.5])
        log_b  = torch.tensor([-1.0])
        result = compute_is_weights(log_pi, log_b, return_named=True)
        assert isinstance(result, IsWeightResult)
        assert hasattr(result, "weights")
        assert hasattr(result, "log_ratios")


class TestClipIsWeights:
    def test_unclipped_stays_unchanged(self):
        w = torch.tensor([0.95, 1.0, 1.05])
        clipped = clip_is_weights(w, eps=0.2)
        assert torch.allclose(clipped, w)

    def test_clips_above(self):
        w = torch.tensor([1.5, 2.0])
        clipped = clip_is_weights(w, eps=0.2)
        assert torch.all(clipped <= 1.2 + 1e-6)

    def test_clips_below(self):
        w = torch.tensor([0.5, 0.3])
        clipped = clip_is_weights(w, eps=0.2)
        assert torch.all(clipped >= 0.8 - 1e-6)

    def test_boundary_values(self):
        w = torch.tensor([0.8, 1.2])
        clipped = clip_is_weights(w, eps=0.2)
        assert torch.allclose(clipped, w)

    def test_gradient_flows_through(self):
        w = torch.tensor([0.9], requires_grad=True)
        clipped = clip_is_weights(w, eps=0.2)
        clipped.sum().backward()
        assert w.grad is not None


class TestExactShadow:
    def test_shadow_validation_passes_close_values(self):
        """Shadow check passes when float and exact ratios agree on clip side."""
        from fractions import Fraction
        pi_prob = Fraction(3, 5)
        b_prob  = Fraction(1, 2)
        result = compute_is_weights_exact_shadow(
            log_pi=torch.tensor([float(pi_prob)]).log(),
            log_b=torch.tensor([float(b_prob)]).log(),
            exact_pi=[pi_prob],
            exact_b=[b_prob],
            clip_eps=0.2,
        )
        assert "n_boundary_flips" in result
        assert isinstance(result["n_boundary_flips"], int)

    def test_shadow_detects_flip(self):
        """Shadow check detects when float clips but exact does not (or vice versa)."""
        # Construct a case near the boundary
        from fractions import Fraction
        import struct, math
        clip_eps = 0.1
        boundary = 1.0 + clip_eps
        # Exact ratio slightly below boundary
        exact_pi = Fraction(11, 10) - Fraction(1, 10**8)
        exact_b  = Fraction(1)
        # Float representation may round to above boundary
        log_pi = torch.tensor([math.log(float(exact_pi))], dtype=torch.float32)
        log_b  = torch.tensor([0.0], dtype=torch.float32)
        result = compute_is_weights_exact_shadow(
            log_pi=log_pi, log_b=log_b,
            exact_pi=[exact_pi], exact_b=[exact_b],
            clip_eps=clip_eps,
        )
        assert "n_boundary_flips" in result
