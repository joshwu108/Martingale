"""The doctor on a record with replayed rows: fresh vs replayed buckets (lag, floor, ESS), the buffer's declared
sequence weights against the ratios measured from the trainer's scores, the live doctor's split, and the report."""
import math
from fractions import Fraction

import pytest
from click.testing import CliRunner

from martingale.cli import cli
from martingale.diagnostics import IncrementalDiagnosis, decompose, diagnosis, render_markdown, scored_tokens
from martingale.diagnostics.replay import replay
from martingale.record import Recorder, SamplerConfig

S = SamplerConfig(temperature=1.0)
TOK, CD = "f" * 64, "c" * 64


def _ws(tmp_path):
    rec = Recorder(tmp_path / "ws")
    r0 = rec.publish_revision("1" * 64, TOK, S, step=0)
    r3 = rec.publish_revision("3" * 64, TOK, S, step=3)
    # fresh A: lag 0 with a tiny floor; fresh B: lag 3 with token ratios 2 and 1/2
    with rec.sequence(0, "a", [1]) as a:
        a.token(r0.digest, 5, -0.5); a.token(r0.digest, 6, -1.0)
    rec.score(a.record.digest, r0.digest, [-0.5, -1.0 + 1e-6])
    with rec.sequence(0, "b", [2]) as b:
        b.token(r0.digest, 5, -1.0); b.token(r0.digest, 6, -1.0)
    rec.score(b.record.digest, r3.digest, [-1.0 + math.log(2), -1.0 - math.log(2)])
    # replayed copy of B scored at step 3 (lag 3): the buffer declares w = 1 and a policy rescale of 1/2
    s = rec.replay_from(b.record.digest, actor_id=1, sequence_id="step3/replay0", draw_id="1", content_digest=CD,
                        is_weight=Fraction(1), rescale=Fraction(1, 2))
    rec.score(s.digest, r3.digest, [-1.0 + math.log(2), -1.0 - math.log(2)])
    # replayed copy of A scored at step 3 with declared weight 1/4 (the trainer measures a ratio of 1)
    s2 = rec.replay_from(a.record.digest, actor_id=1, sequence_id="step3/replay1", draw_id="2", content_digest=CD,
                         is_weight=Fraction(1, 4))
    rec.score(s2.digest, r3.digest, [-0.5, -1.0])
    # a third replayed copy that was never scored
    rec.replay_from(a.record.digest, actor_id=1, sequence_id="step3/replay2", draw_id="3", content_digest=CD, is_weight=1)
    return rec


class TestDecompose:
    def test_tokens_carry_provenance(self, tmp_path):
        toks = list(scored_tokens(_ws(tmp_path).ledger))
        assert sorted(t.provenance for t in toks) == ["fresh"] * 4 + ["replayed"] * 4
        assert {t.lag for t in toks if t.provenance == "replayed"} == {3}

    def test_buckets_per_provenance_and_counts(self, tmp_path):
        d = decompose(_ws(tmp_path).ledger)
        assert d["n_sequences"] == 5 and d["n_replayed_sequences"] == 3 and d["n_unscored_sequences"] == 1
        assert d["staleness_histogram"] == {"0": 2, "3": 6}             # the combined view keeps every token
        bp = d["by_provenance"]
        assert set(bp) == {"fresh", "replayed"}
        assert [b["lag"] for b in bp["fresh"]["by_lag"]] == [0, 3] and [b["lag"] for b in bp["replayed"]["by_lag"]] == [3]
        assert bp["replayed"]["by_lag"][0]["n_tokens"] == 4 and bp["replayed"]["n_sequences"] == 2
        assert bp["replayed"]["n_scored_tokens"] == 4 and bp["fresh"]["n_scored_tokens"] == 4
        assert bp["fresh"]["lag0_floor"]["n_tokens"] == 2 and bp["replayed"]["lag0_floor"] is None
        # the replayed bucket has no lag-0 tokens: its attribution borrows the fresh floor and says so
        assert bp["replayed"]["attribution"]["floor_source"] == "fresh lag-0"
        assert bp["replayed"]["attribution"]["floor_mean_abs_log_ratio"] == bp["fresh"]["attribution"]["floor_mean_abs_log_ratio"]
        assert bp["fresh"]["attribution"]["floor_source"] == "own lag-0"
        assert Fraction(bp["replayed"]["attribution"]["staleness_share"]) > Fraction(99, 100)

    def test_ess_per_bucket_and_declared_versus_measured_weights(self, tmp_path):
        d = decompose(_ws(tmp_path).ledger)
        rb = d["by_provenance"]["replayed"]["by_lag"][0]["float_informational"]
        s1, s2 = 2 + 0.5 + 1 + 1, 4 + 0.25 + 1 + 1                     # token ratios of the two scored replayed rows
        assert rb["ess"] == pytest.approx(s1 * s1 / s2, rel=1e-6)
        fb = next(b for b in d["by_provenance"]["fresh"]["by_lag"] if b["lag"] == 3)["float_informational"]
        assert fb["ess"] == pytest.approx((2 + 0.5) ** 2 / (4 + 0.25), rel=1e-6)
        r = d["replayed"]
        assert r["n_sequences"] == 3 and r["n_scored_sequences"] == 2 and r["n_without_origin"] == 0
        w = r["sequence_weights"]
        assert w["n"] == 2 and w["n_train_steps"] == 1
        # applied weights 1 * 1/2 and 1/4 * 1: ESS = (3/4)^2 / (1/4 + 1/16), over 2 rows
        assert w["declared_ess_fraction"] == pytest.approx(((0.75 ** 2) / (0.25 + 0.0625)) / 2, rel=1e-9)
        # the trainer's scores give a sequence ratio of exactly 1 on both rows
        assert w["measured_ess_fraction"] == pytest.approx(1.0, rel=1e-9)
        # log declared - log measured: 0 and log(1/4); centred per train step the dispersion is log 2
        assert w["mean_log_declared_minus_measured"] == pytest.approx(-math.log(4) / 2, rel=1e-9)
        assert w["log_weight_dispersion"] == pytest.approx(math.log(2), rel=1e-9)

    def test_zero_and_extreme_weights_do_not_crash_the_report(self, tmp_path):
        rec = Recorder(tmp_path / "ws")
        r0 = rec.publish_revision("1" * 64, TOK, S, step=0)
        r1 = rec.publish_revision("2" * 64, TOK, S, step=1)
        with rec.sequence(0, "a", [1]) as a:
            a.token(r0.digest, 5, -0.5)
        rec.score(a.record.digest, r0.digest, [-0.5])
        for k, w in enumerate((Fraction(0), Fraction(0), Fraction(1, 10 ** 400), Fraction(10 ** 400))):
            s = rec.replay_from(a.record.digest, actor_id=1, sequence_id=f"r{k}", draw_id=str(k), content_digest=CD, is_weight=w)
            rec.score(s.digest, r1.digest, [-0.5])
        d = decompose(rec.ledger)
        w = d["replayed"]["sequence_weights"]
        # 10**-400 underflows to 0.0 (a zero weight), 10**400 has no float image (overflow)
        assert w["n"] == 4 and w["n_overflow"] == 1 and w["n_zero_weight"] == 3 and w["n_dispersion_rows"] == 0
        assert w["declared_ess_fraction"] is None and w["measured_ess_fraction"] == pytest.approx(1.0)
        assert w["log_weight_dispersion"] is None
        lines = diagnosis(d)
        assert any("n/a" in line for line in lines) and not any("disagree" in line for line in lines)
        assert "n/a" in render_markdown(d)

    def test_dispersion_needs_two_rows_in_a_step(self, tmp_path):
        rec = Recorder(tmp_path / "ws")
        r0 = rec.publish_revision("1" * 64, TOK, S, step=0)
        r1 = rec.publish_revision("2" * 64, TOK, S, step=1)
        r2 = rec.publish_revision("3" * 64, TOK, S, step=2)
        with rec.sequence(0, "a", [1]) as a:
            a.token(r0.digest, 5, -0.5)
        for k, (rev, w) in enumerate(((r1, Fraction(1, 1000)), (r2, Fraction(1000)))):   # one row per step, nats apart
            s = rec.replay_from(a.record.digest, actor_id=1, sequence_id=f"r{k}", draw_id=str(k), content_digest=CD, is_weight=w)
            rec.score(s.digest, rev.digest, [-0.5])
        d = decompose(rec.ledger)
        w = d["replayed"]["sequence_weights"]
        assert w["n"] == 2 and w["n_train_steps"] == 2 and w["n_dispersion_rows"] == 0 and w["log_weight_dispersion"] is None
        assert w["mean_log_declared_minus_measured"] == pytest.approx(0.0, abs=1e-9)
        assert not any("disagree" in line for line in diagnosis(d))

    def test_rows_without_origin_are_counted(self, tmp_path):
        rec = _ws(tmp_path)
        r0 = next(r for r in rec.ledger.all_revisions() if r.step == 0)
        rec.replayed(actor_id=1, sequence_id="restored", prompt_ids=[7], steps=[(r0.digest, 1, -0.2)],
                     draw_id="9", content_digest=CD, is_weight=1)
        d = decompose(rec.ledger)
        assert d["replayed"]["n_without_origin"] == 1 and d["n_replayed_sequences"] == 4


class TestDiagnosisAndReport:
    def test_lines_and_markdown_mention_the_replayed_bucket(self, tmp_path):
        d = decompose(_ws(tmp_path).ledger)
        lines = diagnosis(d)
        assert any("Replayed rows" in line and "3 sequences" in line for line in lines), lines
        assert any("declared" in line for line in lines)
        md = render_markdown(d)
        assert "| replayed | 3 |" in md and "| fresh | 0 |" in md and "| fresh | 3 |" in md and "declared" in md
        r = CliRunner().invoke(cli, ["doctor", "--dir", str(tmp_path / "ws")])
        assert r.exit_code == 0, r.output
        assert "Replayed rows" in r.output

    def test_disagreeing_weights_get_a_line(self, tmp_path):
        rec = _ws(tmp_path)
        r3 = next(r for r in rec.ledger.all_revisions() if r.step == 3)
        b = next(s for s in rec.ledger.all_sequences() if s.sequence_id == "b")
        for k in range(3):   # three more copies whose declared weight is off by nats with no common normaliser
            s = rec.replay_from(b.digest, actor_id=1, sequence_id=f"off{k}", draw_id=f"off{k}", content_digest=CD,
                                is_weight=Fraction(1, 10 ** (k + 1)))
            rec.score(s.digest, r3.digest, [-1.0 + math.log(2), -1.0 - math.log(2)])
        lines = diagnosis(decompose(rec.ledger))
        assert any("disagree" in line for line in lines), lines

    def test_no_replayed_rows_keeps_the_report_as_before(self, tmp_path):
        rec = Recorder(tmp_path / "ws")
        r0 = rec.publish_revision("1" * 64, TOK, S, step=0)
        with rec.sequence(0, "a", [1]) as a:
            a.token(r0.digest, 5, -0.5)
        rec.score(a.record.digest, r0.digest, [-0.5])
        d = decompose(rec.ledger)
        assert d["replayed"] is None and set(d["by_provenance"]) == {"fresh"} and d["n_replayed_sequences"] == 0
        assert not any("Replayed" in line for line in diagnosis(d))
        assert "replayed" not in render_markdown(d)


class TestIncremental:
    def test_step_diagnosis_splits_provenance_and_logs_replayed_metrics(self, tmp_path):
        rec = _ws(tmp_path)
        inc = IncrementalDiagnosis(rec.ledger)
        d = inc.update(step=4)
        assert set(d.by_provenance) == {"fresh", "replayed"}
        assert [b["lag"] for b in d.by_provenance["replayed"]] == [3] and d.by_provenance["replayed"][0]["n_tokens"] == 4
        full = decompose(rec.ledger)
        for prov in ("fresh", "replayed"):
            assert [(b["lag"], b["n_tokens"], b["mean_abs_log_ratio"]) for b in d.cumulative_by_provenance[prov]] == \
                [(b["lag"], b["n_tokens"], b["mean_abs_log_ratio"]) for b in full["by_provenance"][prov]["by_lag"]]
        m = d.metrics()
        assert m["martingale/replayed_tokens"] == 4.0 and m["martingale/replayed_share"] == 0.5
        assert m["martingale/replayed_ess_fraction_lag3"] == pytest.approx(
            full["by_provenance"]["replayed"]["by_lag"][0]["float_informational"]["ess_fraction"])
        assert "martingale/ess_fraction_lag3" in m and "martingale/ess_fraction_lag0" in m   # combined keys unchanged
        d2 = inc.update(step=5)
        assert d2.by_provenance == {} and d2.cumulative_by_provenance == d.cumulative_by_provenance
        assert "martingale/replayed_tokens" not in d2.metrics()

    def test_monitor_replay_runs_on_a_record_with_replayed_rows(self, tmp_path):
        steps = replay(_ws(tmp_path).ledger)
        assert steps and steps[-1].diagnosis.cumulative_by_provenance["replayed"][0]["n_tokens"] == 4
