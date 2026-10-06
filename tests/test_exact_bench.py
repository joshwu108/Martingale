"""Tests for martingale.exact — the exact estimator test bench (Phase 4-C)."""
from fractions import Fraction

import pytest

from martingale.estimators import on_policy_gradient
from martingale.exact import (
    BenchReport,
    check_unbiased,
    defendants,
    expected_gradient,
)
from martingale.exact.random_mdp import make_mdp, make_policy, make_stale_policy

SEED = b"test-exact-bench"


@pytest.fixture
def small_case():
    mdp = make_mdp(SEED, n_states=3, n_actions=2, horizon=3)
    target = make_policy(SEED + b"pi", mdp.states, mdp.actions)
    behavior = make_stale_policy(target, mdp, n_steps=2, alpha=Fraction(1, 10),
                                 action_seq_seed=SEED + b"lag")
    return mdp, target, behavior


class TestExpectedGradient:
    def test_is_reinforce_matches_on_policy_exactly(self, small_case):
        mdp, target, behavior = small_case
        got = expected_gradient(mdp, defendants.seq_is, target, behavior)
        want = on_policy_gradient(mdp, target)
        assert got == want  # exact Fraction equality, every component

    def test_unweighted_stale_estimator_is_biased(self, small_case):
        mdp, target, behavior = small_case
        got = expected_gradient(mdp, defendants.unweighted, target, behavior)
        want = on_policy_gradient(mdp, target)
        assert got != want

    def test_zero_lag_makes_every_defendant_unbiased(self, small_case):
        mdp, target, _ = small_case
        want = on_policy_gradient(mdp, target)
        for est in (defendants.unweighted, defendants.seq_ppo_clip(Fraction(1, 10)),
                    defendants.seq_truncated_is(Fraction(2)), defendants.seq_masked_is(Fraction(1, 2), Fraction(2)),
                    defendants.seq_cispo(Fraction(1, 5), Fraction(1, 5))):
            assert expected_gradient(mdp, est, target, target) == want

    def test_all_values_are_fractions(self, small_case):
        mdp, target, behavior = small_case
        got = expected_gradient(mdp, defendants.seq_ppo_clip(Fraction(1, 10)), target, behavior)
        assert all(isinstance(v, Fraction) for v in got.values())


class TestCheckUnbiased:
    def test_is_reinforce_certified_unbiased(self):
        report = check_unbiased(defendants.seq_is, n_configs=4, lags=(1, 2), seed=SEED)
        assert isinstance(report, BenchReport)
        assert report.all_unbiased
        assert report.n_configs == 4
        assert all(c.rel_sq_bias == 0 or c.rel_sq_bias is None for c in report.certificates)

    def test_ppo_clip_certified_biased_with_exact_bias_vector(self):
        report = check_unbiased(defendants.seq_ppo_clip(Fraction(1, 10)), n_configs=4, lags=(2, 4), seed=SEED)
        assert not report.all_unbiased
        biased = [c for c in report.certificates if not c.is_unbiased]
        assert biased
        c = biased[0]
        assert isinstance(c.rel_sq_bias, Fraction) and c.rel_sq_bias > 0
        assert any(v != 0 for v in c.bias.values())

    def test_user_defined_estimator_is_accepted(self):
        def my_estimator(traj, target, behavior):
            # deliberately wrong: forgets the IS weight entirely
            return defendants.unweighted(traj, target, behavior)

        report = check_unbiased(my_estimator, n_configs=3, lags=(4,), seed=SEED, name="mine")
        assert report.estimator_name == "mine"
        assert not report.all_unbiased

    def test_report_is_json_serialisable_with_reduced_fractions(self):
        import json
        report = check_unbiased(defendants.seq_truncated_is(Fraction(3, 2)), n_configs=2, lags=(1,), seed=SEED)
        d = report.to_dict()
        json.dumps(d)
        for cert in d["certificates"]:
            assert cert["rel_sq_bias"] is None or "/" in cert["rel_sq_bias"] or cert["rel_sq_bias"] == "0"

    def test_deterministic_for_same_seed(self):
        a = check_unbiased(defendants.seq_cispo(Fraction(1, 5), Fraction(1, 5)), n_configs=3, lags=(1,), seed=SEED)
        b = check_unbiased(defendants.seq_cispo(Fraction(1, 5), Fraction(1, 5)), n_configs=3, lags=(1,), seed=SEED)
        assert a.to_dict() == b.to_dict()

    def test_refuses_infeasible_enumeration(self):
        with pytest.raises(ValueError, match="feasib"):
            check_unbiased(defendants.seq_is, n_configs=1, lags=(1,), seed=SEED,
                           sizes=((5, 3, 20),))


class TestRandomMdpHelpersStayIdenticalToCampaign:
    """The identity campaign must keep producing the same configs after the extraction."""

    def test_campaign_imports_shared_helpers(self):
        from campaigns import identity
        assert identity._make_mdp is make_mdp
        assert identity._make_policy is make_policy
        assert identity._make_stale_policy is make_stale_policy


class TestTokenLevelDefendants:
    def test_token_is_is_biased_with_trajectory_advantage(self, small_case):
        mdp, target, behavior = small_case
        assert expected_gradient(mdp, defendants.token_is, target, behavior) != on_policy_gradient(mdp, target)

    def test_token_and_seq_agree_at_horizon_one(self):
        mdp = make_mdp(SEED, n_states=3, n_actions=3, horizon=1)
        target = make_policy(SEED + b"pi", mdp.states, mdp.actions)
        behavior = make_stale_policy(target, mdp, 3, Fraction(1, 10), SEED + b"lag")
        for seq, tok in ((defendants.seq_is, defendants.token_is),
                         (defendants.seq_ppo_clip(Fraction(1, 10)), defendants.token_ppo_clip(Fraction(1, 10))),
                         (defendants.seq_cispo(Fraction(1, 5), Fraction(1, 5)), defendants.token_cispo(Fraction(1, 5), Fraction(1, 5)))):
            assert expected_gradient(mdp, seq, target, behavior) == expected_gradient(mdp, tok, target, behavior)

    def test_every_builtin_is_unbiased_at_zero_lag(self, small_case):
        mdp, target, _ = small_case
        want = on_policy_gradient(mdp, target)
        for name, item in defendants.BUILTIN.items():
            if name in ("unweighted", "seq_is", "token_is"):
                est = item
            elif "masked" in name or "cispo" in name:
                est = item(Fraction(1, 2), Fraction(2))
            elif "truncated" in name:
                est = item(Fraction(2))       # cap must admit r = 1
            else:
                est = item(Fraction(1, 5))
            assert expected_gradient(mdp, est, target, target) == want, name


class TestValidationAndSupport:
    @pytest.mark.parametrize("factory,args", [
        (defendants.seq_truncated_is, (Fraction(0),)),
        (defendants.token_masked_is, (Fraction(2), Fraction(1))),
        (defendants.seq_ppo_clip, (Fraction(-1, 10),)),
        (defendants.token_ppo_clip, (Fraction(1),)),
        (defendants.seq_cispo, (Fraction(1), Fraction(1, 5))),
    ])
    def test_factories_reject_bad_parameters(self, factory, args):
        with pytest.raises(ValueError):
            factory(*args)

    def test_zero_behavior_probability_raises_support_error(self):
        from martingale.policy import RationalPolicy
        target = RationalPolicy({0: {0: Fraction(1, 2), 1: Fraction(1, 2)}})
        behavior = RationalPolicy({0: {0: Fraction(1), 1: Fraction(0)}})
        traj = {"prob": Fraction(1), "steps": [(0, 1, 0, Fraction(1))], "return": Fraction(1)}
        with pytest.raises(defendants.SupportError):
            defendants.seq_is(traj, target, behavior)

    def test_zero_target_probability_gives_zero_contribution_not_error(self):
        from martingale.policy import RationalPolicy
        target = RationalPolicy({0: {0: Fraction(1), 1: Fraction(0)}})
        behavior = RationalPolicy({0: {0: Fraction(1, 2), 1: Fraction(1, 2)}})
        traj = {"prob": Fraction(1, 2), "steps": [(0, 1, 0, Fraction(1))], "return": Fraction(1)}
        assert defendants.seq_is(traj, target, behavior) == {}
        assert defendants.token_ppo_clip(Fraction(1, 10))(traj, target, behavior) == {}

    def test_rel_sq_bias_is_none_when_true_gradient_is_zero(self):
        from martingale.exact.bench import BiasCertificate
        c = BiasCertificate(0, 1, 1, 1, 1, {(0, 0): Fraction(1)}, Fraction(1), Fraction(0), None)
        assert c.to_dict()["rel_sq_bias"] is None
        assert c.to_dict()["rel_sq_bias_float_informational"] is None
