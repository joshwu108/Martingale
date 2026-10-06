"""Tests for martingale/corrections.py against the exact weight functions in
martingale/exact/defendants.py."""
from __future__ import annotations

import ast
import importlib
import inspect
import math
import sys
from fractions import Fraction

import pytest

torch = pytest.importorskip("torch")

from martingale import corrections as C  # noqa: E402
from martingale.exact import defendants as D  # noqa: E402
from martingale.exact.random_mdp import seeded_fraction  # noqa: E402

N_PAIRS = 200
REL_TOL = 1e-9
CAP, LO, HI, EPS, EPS_LO, EPS_HI = 1.5, 0.8, 1.2, 0.2, 0.2, 0.3


def _seeded_pairs(n: int = N_PAIRS) -> list[tuple[Fraction, Fraction]]:
    """n deterministic (pi, b) pairs of exact rationals in [1/8, 1]; ratios in [1/8, 8]."""
    return [
        (seeded_fraction(b"corr-pi-" + i.to_bytes(4, "big"), 1, 8, 1, 1),
         seeded_fraction(b"corr-b-" + i.to_bytes(4, "big"), 1, 8, 1, 1))
        for i in range(n)
    ]


def _logs(pairs):
    """float64 log-prob tensors for a list of exact (pi, b) pairs."""
    lp = torch.tensor([math.log(float(p)) for p, _ in pairs], dtype=torch.float64)
    lb = torch.tensor([math.log(float(b)) for _, b in pairs], dtype=torch.float64)
    return lp, lb


def _assert_matches_exact(got: torch.Tensor, pairs, exact_fn, G: Fraction) -> None:
    assert got.dtype == torch.float64
    for i, (p, b) in enumerate(pairs):
        want = float(exact_fn(p / b, G))
        assert math.isclose(got[i].item(), want, rel_tol=REL_TOL, abs_tol=0.0), (
            f"index {i}: ratio={p / b} torch={got[i].item()!r} exact={want!r}")


# ---- 1. exact-shadow agreement ------------------------------------------------

class TestExactShadowAgreement:
    pairs = _seeded_pairs()
    G = Fraction(1)

    def test_is_weight(self):
        lp, lb = _logs(self.pairs)
        _assert_matches_exact(C.is_weight(lp, lb, dtype=torch.float64),
                              self.pairs, D._identity, self.G)

    def test_truncated_is(self):
        lp, lb = _logs(self.pairs)
        _assert_matches_exact(C.truncated_is(lp, lb, CAP, dtype=torch.float64),
                              self.pairs, D._truncate(Fraction(3, 2)), self.G)

    def test_masked_is(self):
        lp, lb = _logs(self.pairs)
        _assert_matches_exact(C.masked_is(lp, lb, LO, HI, dtype=torch.float64),
                              self.pairs, D._mask(Fraction(4, 5), Fraction(6, 5)), self.G)

    @pytest.mark.parametrize("G", [Fraction(1), Fraction(-1)])
    def test_ppo_clip_weight_both_advantage_signs(self, G):
        lp, lb = _logs(self.pairs)
        adv = torch.full((len(self.pairs),), float(G), dtype=torch.float64)
        _assert_matches_exact(C.ppo_clip_weight(lp, lb, adv, EPS, dtype=torch.float64),
                              self.pairs, D._ppo(Fraction(1, 5)), G)

    def test_cispo_weight(self):
        lp, lb = _logs(self.pairs)
        _assert_matches_exact(C.cispo_weight(lp, lb, EPS_LO, EPS_HI, dtype=torch.float64),
                              self.pairs, D._cispo(Fraction(1, 5), Fraction(3, 10)), self.G)

    def test_every_branch_is_exercised(self):
        """The 200 pairs must hit both sides of every clip window, or the agreement
        tests above say nothing about the clipped branches."""
        ratios = [p / b for p, b in self.pairs]
        assert any(r > Fraction(3, 2) for r in ratios) and any(r < Fraction(3, 2) for r in ratios)
        assert any(r < Fraction(4, 5) for r in ratios) and any(r > Fraction(6, 5) for r in ratios)
        assert any(Fraction(4, 5) <= r <= Fraction(6, 5) for r in ratios)


# ---- 2. clip-boundary ties -----------------------------------------------------

class TestClipBoundaryTies:
    """log_b = 0 and log_pi = log(1 +/- eps) give ratios exactly on the boundary in
    float64 for the eps values used here (exp(log(x)) == x; asserted, not assumed)."""

    def _tie_inputs(self, eps):
        lp = torch.log(torch.tensor([1 + eps, 1 - eps], dtype=torch.float64))
        lb = torch.zeros(2, dtype=torch.float64)
        w = C.is_weight(lp, lb, dtype=torch.float64)
        assert w.tolist() == [1 + eps, 1 - eps], "precondition: ratio must sit exactly on the boundary"
        return lp, lb

    @pytest.mark.parametrize("eps", [0.2, 0.25, 0.5])
    @pytest.mark.parametrize("G", [Fraction(1), Fraction(-1)])
    def test_ppo_tie_takes_unclipped_branch(self, eps, G):
        lp, lb = self._tie_inputs(eps)
        adv = torch.full((2,), float(G), dtype=torch.float64)
        got = C.ppo_clip_weight(lp, lb, adv, eps, dtype=torch.float64)
        exact = D._ppo(Fraction(eps).limit_denominator(100))
        for i, r in enumerate([Fraction(1) + Fraction(eps).limit_denominator(100),
                               Fraction(1) - Fraction(eps).limit_denominator(100)]):
            assert got[i].item() == float(exact(r, G))
        assert torch.equal(got, torch.tensor([1 + eps, 1 - eps], dtype=torch.float64))

    def test_mask_and_cispo_ties_are_inclusive(self):
        lp, lb = self._tie_inputs(EPS)
        masked = C.masked_is(lp, lb, 1 - EPS, 1 + EPS, dtype=torch.float64)
        assert masked.tolist() == [1 + EPS, 1 - EPS]
        cispo = C.cispo_weight(lp, lb, EPS, EPS, dtype=torch.float64)
        assert cispo.tolist() == [1 + EPS, 1 - EPS]
        truncated = C.truncated_is(lp, lb, 1 + EPS, dtype=torch.float64)
        assert truncated.tolist() == [1 + EPS, 1 - EPS]


# ---- 3. clip_flips --------------------------------------------------------------

class TestClipFlips:
    def test_finds_flip_on_adversarial_float32_input(self):
        """Construction: exact ratios r_k = (1+eps) + k / 2**30 for k in [-4, 4] \\ {0},
        with b = 1/2 and pi = r_k * b. The offsets (~1e-9) are far below the float32
        ulp at 1.2 (~1.2e-7), so every r_k maps to the same float32 ratio. Exact
        sides differ (k < 0 inside, k > 0 above), so whichever side float32 picks,
        at least four of the eight must flip."""
        hi = Fraction(1) + Fraction(1, 5)
        b = Fraction(1, 2)
        ratios = [hi + Fraction(k, 2**30) for k in range(-4, 5) if k != 0]
        pairs = [(r * b, b) for r in ratios]
        lp, lb = _logs(pairs)
        out = C.clip_flips(lp, lb, [p for p, _ in pairs], [b for _, b in pairs], EPS)
        assert out["n_checked"] == 8
        assert out["n_flips"] >= 4
        assert out["flip_indices"] == sorted(out["sides"])
        for i in out["flip_indices"]:
            exact_side, float_side = out["sides"][i]
            assert exact_side != float_side

    def test_no_flips_far_from_boundary(self):
        pairs = [(Fraction(1, 2), Fraction(1, 2)), (Fraction(9, 10), Fraction(1, 2)),
                 (Fraction(1, 10), Fraction(1, 2))]
        lp, lb = _logs(pairs)
        out = C.clip_flips(lp, lb, [p for p, _ in pairs], [b for _, b in pairs], EPS)
        assert out == {"n_checked": 3, "n_flips": 0, "flip_indices": [], "sides": {}}

    def test_accepts_exact_fraction_eps(self):
        pairs = [(Fraction(3, 5), Fraction(1, 2))]   # ratio exactly 6/5 = 1 + eps
        lp, lb = _logs(pairs)
        out = C.clip_flips(lp, lb, [pairs[0][0]], [pairs[0][1]], Fraction(1, 5))
        assert out["n_flips"] == 0

    def test_length_mismatch_raises(self):
        lp, lb = _logs([(Fraction(1, 2), Fraction(1, 2))])
        with pytest.raises(ValueError):
            C.clip_flips(lp, lb, [Fraction(1, 2)], [], EPS)


# ---- 4. gradients ---------------------------------------------------------------

class TestGradients:
    def _inputs(self):
        # ratios: 0.5 (below lo), 1.0 (inside), 2.0 (above hi), 1.0 (inside)
        log_b = torch.zeros(4, dtype=torch.float64)
        log_pi = torch.log(torch.tensor([0.5, 1.0, 2.0, 1.0], dtype=torch.float64)).requires_grad_(True)
        adv = torch.tensor([-1.0, 1.0, 1.0, -1.0], dtype=torch.float64)
        return log_pi, log_b, adv

    def test_ppo_surrogate_gradient_zero_only_where_clipped(self):
        log_pi, log_b, adv = self._inputs()
        loss = C.ppo_surrogate_loss(log_pi, log_b, adv, EPS)
        loss.backward()
        clipped = [True, False, True, False]  # (A<0, r<lo), inside, (A>0, r>hi), inside
        for i, is_clipped in enumerate(clipped):
            if is_clipped:
                assert log_pi.grad[i].item() == 0.0
            else:
                assert log_pi.grad[i].item() != 0.0
        # the unclipped tokens get d/dlog_pi of -(r A)/n = -(r A)/n
        assert math.isclose(log_pi.grad[1].item(), -1.0 / 4)
        assert math.isclose(log_pi.grad[3].item(), 1.0 / 4)

    def test_ppo_surrogate_value_matches_min_formula(self):
        log_pi, log_b, adv = self._inputs()
        r = torch.exp(log_pi - log_b).detach()
        want = -torch.minimum(r * adv, r.clamp(1 - EPS, 1 + EPS) * adv).mean()
        assert torch.allclose(C.ppo_surrogate_loss(log_pi, log_b, adv, EPS).detach(), want)

    def test_cispo_gradient_nonzero_on_every_token_with_nonzero_advantage(self):
        log_pi, log_b, adv = self._inputs()
        loss = C.cispo_loss(log_pi, log_b, adv, EPS_LO, EPS_HI)
        loss.backward()
        assert (log_pi.grad != 0).all()
        # gradient is -sg(clip(r)) A / n: clipped tokens contribute the clip value
        clip_r = torch.tensor([0.5, 1.0, 2.0, 1.0]).clamp(1 - EPS_LO, 1 + EPS_HI).double()
        assert torch.allclose(log_pi.grad, -(clip_r * adv) / 4)

    def test_cispo_zero_advantage_gives_zero_gradient(self):
        log_pi, log_b, _ = self._inputs()
        C.cispo_loss(log_pi, log_b, torch.zeros(4, dtype=torch.float64), EPS_LO, EPS_HI).backward()
        assert (log_pi.grad == 0).all()

    def test_cispo_weight_is_detached(self):
        log_pi, log_b, _ = self._inputs()
        assert not C.cispo_weight(log_pi, log_b, EPS_LO, EPS_HI, dtype=torch.float64).requires_grad


# ---- 5. parameter validation -------------------------------------------------

class TestValidation:
    lp = torch.zeros(2)
    lb = torch.zeros(2)
    adv = torch.ones(2)

    @pytest.mark.parametrize("cap", [0.0, -1.0])
    def test_truncated_cap(self, cap):
        with pytest.raises(ValueError):
            C.truncated_is(self.lp, self.lb, cap)

    @pytest.mark.parametrize("lo,hi", [(1.2, 0.8), (-0.1, 1.0)])
    def test_masked_bounds(self, lo, hi):
        with pytest.raises(ValueError):
            C.masked_is(self.lp, self.lb, lo, hi)

    @pytest.mark.parametrize("eps", [-0.1, 1.0, 1.5])
    def test_ppo_eps(self, eps):
        with pytest.raises(ValueError):
            C.ppo_clip_weight(self.lp, self.lb, self.adv, eps)
        with pytest.raises(ValueError):
            C.ppo_surrogate_loss(self.lp, self.lb, self.adv, eps)
        with pytest.raises(ValueError):
            C.clip_flips(self.lp, self.lb, [Fraction(1)] * 2, [Fraction(1)] * 2, eps)

    @pytest.mark.parametrize("eps_lo,eps_hi", [(-0.1, 0.2), (1.0, 0.2), (0.2, -0.1)])
    def test_cispo_eps(self, eps_lo, eps_hi):
        with pytest.raises(ValueError):
            C.cispo_weight(self.lp, self.lb, eps_lo, eps_hi)
        with pytest.raises(ValueError):
            C.cispo_loss(self.lp, self.lb, self.adv, eps_lo, eps_hi)

    def test_valid_edges_accepted(self):
        C.ppo_clip_weight(self.lp, self.lb, self.adv, 0.0)
        C.masked_is(self.lp, self.lb, 0.0, 0.0)
        C.cispo_weight(self.lp, self.lb, 0.0, 0.0)


# ---- 6. importable without torch --------------------------------------------

class TestImportWithoutTorch:
    def test_no_top_level_torch_import_in_ast(self):
        tree = ast.parse(inspect.getsource(C))
        for node in tree.body:   # only module-level statements, TYPE_CHECKING block excluded
            if isinstance(node, ast.Import):
                assert all(a.name.split(".")[0] != "torch" for a in node.names)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "torch"

    def test_reload_with_torch_absent(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)   # `import torch` now raises ImportError
        mod = importlib.reload(C)
        assert hasattr(mod, "ppo_clip_weight")
        monkeypatch.undo()
        importlib.reload(C)
        assert C.is_weight(torch.zeros(1), torch.zeros(1)).item() == 1.0
