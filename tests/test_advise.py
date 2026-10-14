"""`martingale doctor --advise`: the largest sync interval the per-lag ESS supports, and the S·η learning-rate ceiling."""
import copy
import dataclasses
import json
from fractions import Fraction

import pytest

from martingale.diagnostics.advise import (
    SCALING_LAW_CITATION,
    AdviseConfig,
    advise,
    advise_decomposition,
    advise_table,
    knob_translation,
)
from martingale.diagnostics.incremental import StepDiagnosis

LR = Fraction("5e-6")


def _bucket(lag, n, ess, max_abs="1/10"):
    return {"lag": lag, "n_tokens": n, "max_abs_log_ratio": max_abs,
            "float_informational": {"ess_fraction": ess}}


def _diag(step, by_lag, cumulative, prov=None):
    return StepDiagnosis(step=step, n_new_scores=sum(b["n_tokens"] for b in by_lag), n_new_sequences=1,
                         by_lag=list(by_lag), cumulative_by_lag=list(cumulative),
                         attribution={"float_informational": {}}, floor_mean_abs=None, new_weights_digests=[],
                         span_fraction=0.0, n_negative_lag=0, unscored_new_sequences=0,
                         cumulative_by_provenance=prov or {})


HEALTHY = [_bucket(0, 100, 1.0), _bucket(1, 100, 0.95), _bucket(2, 100, 0.9), _bucket(3, 100, 0.85)]


class TestSyncInterval:
    def test_every_bucket_passing_is_bounded_by_what_was_observed(self):
        a = advise_table(HEALTHY)
        assert a.status == "within_observed"
        assert a.observed_interval == 4 and a.passing_interval == 4 and a.first_failing_lag is None

    def test_first_failing_lag_sets_the_interval(self):
        a = advise_table([_bucket(0, 100, 1.0), _bucket(1, 100, 0.9), _bucket(2, 100, 0.3), _bucket(3, 100, 0.9)])
        assert a.status == "reduce"
        assert a.first_failing_lag == 2 and a.passing_interval == 2 and a.observed_interval == 4

    def test_failing_lag_zero_is_the_engine_floor_not_staleness(self):
        a = advise_table([_bucket(0, 100, 0.4), _bucket(1, 100, 0.9)])
        assert a.status == "engine_floor"
        assert a.passing_interval is None and a.first_failing_lag == 0

    def test_bucket_at_exactly_ten_percent_does_not_qualify(self):
        # 10 of 100 tokens is not "over 10%"; 11 of 100 is
        low = [_bucket(0, 45, 1.0), _bucket(1, 45, 1.0), _bucket(2, 10, 0.01)]
        assert advise_table(low).status == "within_observed"
        low = [_bucket(0, 45, 1.0), _bucket(1, 44, 1.0), _bucket(2, 11, 0.01)]
        assert advise_table(low).first_failing_lag == 2

    def test_token_share_is_exact(self):
        a = advise_table([_bucket(0, 1, 1.0), _bucket(1, 2, 1.0)])
        assert [r.token_share for r in a.rows] == [Fraction(1, 3), Fraction(2, 3)]

    def test_threshold_is_inclusive(self):
        assert advise_table([_bucket(0, 50, 1.0), _bucket(1, 50, 0.5)], AdviseConfig(ess_min=0.5)).status == "within_observed"

    def test_non_qualifying_bucket_between_passing_ones_is_skipped_not_failed(self):
        a = advise_table([_bucket(0, 100, 1.0), _bucket(1, 100, 0.9), _bucket(2, 5, 0.01), _bucket(3, 100, 0.2)])
        assert a.first_failing_lag == 3 and a.passing_interval == 3
        assert [r.qualifies for r in a.rows] == [True, True, False, True]

    def test_negative_lags_are_excluded_and_counted(self):
        a = advise_table([_bucket(-1, 100, 0.01), _bucket(0, 100, 1.0), _bucket(1, 100, 1.0)])
        assert a.status == "within_observed" and a.n_negative_lag_tokens == 100
        assert [r.lag for r in a.rows] == [0, 1]
        assert any("100 tokens at negative lag are excluded" in line for line in a.lines())

    def test_thin_top_lag_is_not_counted_as_supported(self):
        a = advise_table(HEALTHY + [_bucket(4, 5, 1.0)], lr=LR)
        assert a.status == "within_observed" and a.observed_interval == 5 and a.passing_interval == 4
        assert a.lr_ceiling == LR                                      # an unjudged lag is not a reason to cut lr
        assert any("lags 4..4 hold too few tokens" in line for line in a.lines())

    def test_withheld_floor_gives_no_advice(self):
        a = advise_table(HEALTHY, lr=LR, withheld="behavior log-probs are unscaled, trainer's are scaled")
        assert a.status == "withheld" and a.passing_interval is None and a.lr_ceiling is None
        assert "unscaled" in " ".join(a.lines())

    def test_no_tokens_is_no_data(self):
        a = advise_table([])
        assert a.status == "no_data" and a.passing_interval is None and a.lr_ceiling is None
        assert "nothing to advise" in " ".join(a.lines())

    def test_engine_floor_text_says_no_knob_fixes_it(self):
        assert "fix the floor first" in " ".join(advise_table([_bucket(0, 100, 0.2), _bucket(1, 100, 0.9)]).lines())

    def test_rejects_threshold_outside_unit_interval(self):
        with pytest.raises(ValueError):
            AdviseConfig(ess_min=1.5)
        with pytest.raises(ValueError):
            AdviseConfig(min_token_share=Fraction(-1, 10))

    def test_input_is_not_mutated(self):
        table = copy.deepcopy(HEALTHY)
        advise_table(table, lr=LR, target_interval=8)
        assert table == HEALTHY


class TestLearningRateCeiling:
    def test_ceiling_at_the_observed_interval_keeps_s_eta_at_the_passing_budget(self):
        a = advise_table([_bucket(0, 100, 1.0), _bucket(1, 100, 0.9), _bucket(2, 100, 0.3), _bucket(3, 100, 0.9)], lr=LR)
        assert a.lr == LR
        assert a.lr_ceiling == LR * 2 / 4 == Fraction(1, 400000)       # passing interval 2, observed 4
        assert a.lr_failing == LR * 3 / 4                              # the interval that failed, 3, at the same S·η

    def test_target_interval_is_the_scaling_law_and_says_so(self):
        a = advise_table(HEALTHY, lr=LR, target_interval=8)
        assert a.lr_ceiling == LR                                      # nothing measured beyond what ran
        assert a.lr_ceiling_at_target == LR * 4 / 8
        assert a.target_beyond_observed is True
        assert any("not measured" in line for line in a.lines())

    def test_lr_accepts_decimal_strings_exactly(self):
        assert advise_table(HEALTHY, lr="1e-6").lr == Fraction(1, 10**6)

    def test_rejects_non_positive_lr_and_interval(self):
        with pytest.raises(ValueError):
            advise_table(HEALTHY, lr="0")
        with pytest.raises(ValueError):
            advise_table(HEALTHY, lr=LR, target_interval=0)

    def test_engine_floor_gives_no_ceiling(self):
        a = advise_table([_bucket(0, 100, 0.2), _bucket(1, 100, 0.9)], lr=LR, target_interval=2)
        assert a.lr_ceiling is None and a.lr_ceiling_at_target is None

    def test_without_lr_there_is_no_ceiling(self):
        a = advise_table(HEALTHY)
        assert a.lr is None and a.lr_ceiling is None


class TestKnobTranslation:
    # interval S, U updates per pass: synchronous recipes run lags 0..U*K-1, so K <= S // U;
    # one pass with async level a runs lags 0..U+a-1, so a <= S - U
    @pytest.mark.parametrize("interval, per_pass, iterations, async_level", [
        (4, 2, 2, 2), (2, 2, 1, 0), (5, 2, 2, 3), (8, 4, 2, 4), (7, 2, 3, 5)])
    def test_largest_iterations_and_async_level(self, interval, per_pass, iterations, async_level):
        k = knob_translation(interval, per_pass)
        assert (k["max_iterations"], k["max_async_level"]) == (iterations, async_level)

    def test_interval_shorter_than_one_pass_says_to_shrink_the_pass(self):
        k = knob_translation(1, 2)
        assert k["max_iterations"] == 0 and k["max_async_level"] is None and k["max_updates_per_pass"] == 1

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            knob_translation(4, 0)

    def test_text_for_an_interval_shorter_than_one_pass(self):
        a = advise_table([_bucket(0, 100, 1.0), _bucket(1, 100, 0.2)], updates_per_pass=2)
        assert any("shorter than one pass of 2 updates" in line for line in a.lines())

    def test_advice_carries_the_translation_when_given(self):
        a = advise_table([_bucket(0, 100, 1.0), _bucket(1, 100, 0.9), _bucket(2, 100, 0.3), _bucket(3, 100, 0.9)],
                         updates_per_pass=2)
        assert a.knobs == knob_translation(2, 2)
        assert a.knobs["max_iterations"] == 1


class TestText:
    def test_lines_cite_the_law_and_are_a_suggestion(self):
        a = advise_table([_bucket(0, 100, 1.0), _bucket(1, 100, 0.9), _bucket(2, 100, 0.3), _bucket(3, 100, 0.9)],
                         lr=LR, updates_per_pass=2)
        text = "\n".join(a.lines())
        assert "arXiv:2607.01083" in SCALING_LAW_CITATION and SCALING_LAW_CITATION in text
        assert "suggestion" in text.lower() and "heuristic, not a measured ceiling" in text
        assert "does not respond to a lower lr" in text
        assert "lag 2" in text and "0.300" in text and "2.5e-06" in text

    def test_to_dict_is_json_with_exact_strings(self):
        a = advise_table(HEALTHY, lr=LR, target_interval=8, updates_per_pass=2)
        d = json.loads(json.dumps(a.to_dict()))
        assert d["lr_ceiling"] == "1/200000" and d["lr_ceiling_at_target"] == "1/400000"
        assert d["rows"][0]["token_share"] == "1/4"
        assert d["float_informational"]["lr_ceiling"] == pytest.approx(5e-6)
        assert d["citation"] == SCALING_LAW_CITATION


class TestOverHistory:
    def test_uses_the_last_cumulative_table_and_reports_each_step(self):
        s1 = _diag(1, [_bucket(0, 50, 1.0), _bucket(1, 50, 0.9)], [_bucket(0, 50, 1.0), _bucket(1, 50, 0.9)])
        s2 = _diag(2, [_bucket(0, 50, 1.0), _bucket(1, 50, 0.2)], [_bucket(0, 100, 1.0), _bucket(1, 100, 0.6)])
        h = advise([s1, s2])
        assert h.overall.status == "within_observed" and h.overall.passing_interval == 2
        assert [(s.step, s.status, s.passing_interval) for s in h.per_step] == [
            (1, "within_observed", 2), (2, "reduce", 1)]
        assert h.tightest_step == 2

    def test_history_text_and_dict_name_the_tightest_step(self):
        s1 = _diag(1, [_bucket(0, 50, 1.0), _bucket(1, 50, 0.9)], [_bucket(0, 50, 1.0), _bucket(1, 50, 0.9)])
        s2 = _diag(2, [_bucket(0, 50, 0.1), _bucket(1, 50, 0.9)], [_bucket(0, 100, 0.8), _bucket(1, 100, 0.9)])
        h = advise([s1, s2], lr=LR)
        assert h.tightest_step == 2
        assert any("tightest step was 2 (lag 0 failed)" in line for line in h.lines())
        d = json.loads(json.dumps(h.to_dict()))
        assert d["tightest_step"] == 2 and d["per_step"][1]["status"] == "engine_floor"

    def test_prefers_fresh_rows_when_provenance_is_recorded(self):
        fresh = [_bucket(0, 50, 1.0), _bucket(1, 50, 0.9)]
        everything = fresh + [_bucket(7, 50, 0.01)]
        h = advise([_diag(1, everything, everything, prov={"fresh": fresh, "replayed": [_bucket(7, 50, 0.01)]})])
        assert h.overall.observed_interval == 2 and h.overall.rows_source == "fresh"

    def test_a_withheld_floor_anywhere_in_history_withholds(self):
        ok = _diag(1, HEALTHY, HEALTHY)
        bad = dataclasses.replace(_diag(2, HEALTHY, HEALTHY), floor_withheld="temperature scaling differs")
        assert advise([ok, bad]).overall.status == "withheld"

    def test_empty_history_is_rejected(self):
        with pytest.raises(ValueError):
            advise([])


class TestDecomposition:
    def test_reads_fresh_buckets_from_a_decomposition(self):
        d = {"by_lag": HEALTHY + [_bucket(9, 300, 0.01)],
             "by_provenance": {"fresh": {"by_lag": HEALTHY}, "replayed": {"by_lag": [_bucket(9, 300, 0.01)]}},
             "n_negative_lag_tokens": 0}
        a = advise_decomposition(d)
        assert a.status == "within_observed" and a.rows_source == "fresh" and a.observed_interval == 4

    def test_all_replayed_record_gives_no_advice(self):
        d = {"by_lag": HEALTHY, "by_provenance": {"replayed": {"by_lag": HEALTHY}}, "n_negative_lag_tokens": 0}
        a = advise_decomposition(d)
        assert a.status == "no_data" and a.rows_source == "fresh"

    def test_without_provenance_reads_every_row(self):
        a = advise_decomposition({"by_lag": HEALTHY, "by_provenance": {}, "n_negative_lag_tokens": 0})
        assert a.rows_source == "all"
