"""2026-10-10: the revision binds the engine's log-prob mode and whether the bits are temperature-scaled.

Record: the two optional sampler keys are inside the digest and absent from the canonical form when
unknown, so manifests written before the keys existed round-trip and verify. Checker: accepts both forms,
rejects a non-boolean. Doctor: at temperature != 1 with the two sides known to differ in temperature
scaling the lag-0 floor is withheld with one sentence, attribution has no floor, and the stale_server
alarm does not fire; when both sides agree the floor and the alarm behave as before."""
import json

import pytest

from checker.verify_tokens import verify_export, verify_revision_manifest
from martingale.diagnostics import (
    AlarmConfig,
    AlarmHistory,
    IncrementalDiagnosis,
    attribute,
    decompose,
    diagnosis,
    evaluate,
)
from martingale.diagnostics.comparability import floor_caveat, floor_comparability
from martingale.diagnostics.report import render_markdown
from martingale.record import SAMPLER_KEYS, LLMRevision, Recorder, SamplerConfig, temperature_applied_for
from martingale.record.bits import digest_json

W, T = "a" * 64, "b" * 64
TOK = "f" * 64
CFG = AlarmConfig(min_floor_tokens=2)


class TestSamplerConfig:
    def test_mode_source_and_temperature_applied_are_inside_the_digest(self):
        base = SamplerConfig(temperature=0.7, logprobs_mode="processed_logprobs", logprobs_source="engine_sampling",
                             temperature_applied=True)
        a = LLMRevision(W, T, base, step=0).digest
        for change in (dict(logprobs_mode="raw_logprobs"), dict(temperature_applied=False), dict(logprobs_source="trainer_scores"),
                       dict(temperature_applied=None), dict(logprobs_mode=None)):
            assert LLMRevision(W, T, SamplerConfig(**{**base.to_dict(), **change}), step=0).digest != a

    def test_optional_keys_are_omitted_when_unknown_so_old_manifests_digest_unchanged(self):
        old = SamplerConfig(temperature=1.0, logprobs_mode="engine_sampling")        # a pre-2026-10-10 record
        assert set(old.to_dict()) == set(SAMPLER_KEYS)
        twelve = {k: None for k in SAMPLER_KEYS} | {"temperature": 1.0, "logprobs_mode": "engine_sampling"}
        manifest = {"kind": "llm", "weights_digest": W, "tokenizer_digest": T, "sampler": twelve, "parent_digest": "0" * 64, "step": 3}
        rev = LLMRevision.from_dict({**manifest, "digest": digest_json(manifest)})
        assert rev.sampler == old and rev.sampler.temperature_applied is None
        assert temperature_applied_for(rev.sampler.logprobs_mode) is None

    def test_new_manifest_round_trips(self):
        s = SamplerConfig(temperature=0.7, logprobs_mode="raw_logprobs", logprobs_source="engine_sampling", temperature_applied=False)
        rev = LLMRevision(W, T, s, step=1)
        assert LLMRevision.from_dict(rev.to_dict()) == rev
        assert rev.to_dict()["sampler"]["temperature_applied"] is False

    def test_temperature_applied_must_be_a_bool(self):
        with pytest.raises(ValueError, match="bool"):
            SamplerConfig(temperature_applied=1)        # type: ignore[arg-type]
        with pytest.raises(ValueError, match="bool"):
            SamplerConfig.from_dict({"temperature_applied": "true"})

    @pytest.mark.parametrize("mode,applied", [("raw_logprobs", False), ("raw_logits", False), ("processed_logprobs", True),
                                              ("processed_logits", True), ("engine_sampling", None), (None, None)])
    def test_temperature_applied_for_vllm_modes(self, mode, applied):
        assert temperature_applied_for(mode) is applied


class TestChecker:
    def _manifest(self, sampler: dict) -> dict:
        m = {"kind": "llm", "weights_digest": W, "tokenizer_digest": T, "sampler": sampler, "parent_digest": "0" * 64, "step": 0}
        return {**m, "digest": digest_json(m)}

    def test_old_twelve_key_manifest_verifies(self):
        twelve = {k: None for k in SAMPLER_KEYS} | {"logprobs_mode": "engine_sampling"}
        assert verify_revision_manifest(self._manifest(twelve)) == []

    def test_new_manifest_verifies_and_the_producer_agrees(self):
        s = SamplerConfig(temperature=0.7, logprobs_mode="processed_logprobs", logprobs_source="engine_sampling", temperature_applied=True)
        d = LLMRevision(W, T, s, step=0).to_dict()
        assert verify_revision_manifest(d) == [] and "temperature_applied" in d["sampler"]

    def test_non_boolean_temperature_applied_and_unknown_key_are_forgeries(self):
        fourteen = {k: None for k in SAMPLER_KEYS} | {"logprobs_source": "engine_sampling", "temperature_applied": 1}
        assert any("not a boolean" in e for e in verify_revision_manifest(self._manifest(fourteen)))
        extra = {k: None for k in SAMPLER_KEYS} | {"beam_width": 4}
        assert any("sampler keys malformed" in e for e in verify_revision_manifest(self._manifest(extra)))
        missing = {k: None for k in SAMPLER_KEYS if k != "temperature"} | {"temperature_applied": True}
        assert any("sampler keys malformed" in e for e in verify_revision_manifest(self._manifest(missing)))

    def test_exported_record_with_both_manifest_forms_verifies(self, tmp_path):
        rec = Recorder(tmp_path / "ws")
        old = rec.publish_revision("1" * 64, TOK, SamplerConfig(temperature=1.0, logprobs_mode="engine_sampling"), step=0)
        new = rec.publish_revision("2" * 64, TOK, SamplerConfig(temperature=1.0, logprobs_mode="processed_logprobs",
                                                                logprobs_source="engine_sampling", temperature_applied=True), step=1)
        with rec.sequence(0, "a", [1]) as a:
            a.token(old.digest, 5, -0.5)
        with rec.sequence(0, "b", [1]) as b:
            b.token(new.digest, 5, -0.5)
        exp = rec.export_for_checker(tmp_path / "exp")
        report = verify_export(exp, expected_head=rec.ledger.head())
        assert report["ok"], report["errors"]
        forms = {len(json.loads(p.read_text())["sampler"]) for p in (exp / "revisions").glob("*.json")}
        assert forms == {12, 14}


def _record(tmp_path, behavior: SamplerConfig, train: SamplerConfig, n: int = 40):
    """n lag-0 tokens the engine was sure of, one scored nats lower (the stale_server pattern), plus a lag-2 bucket."""
    rec = Recorder(tmp_path / "ws")
    rb = rec.publish_revision("1" * 64, TOK, behavior, step=1)
    rt = rec.publish_revision("2" * 64, TOK, train, step=1)
    r3 = rec.publish_revision("3" * 64, TOK, train, step=3)
    with rec.sequence(0, "a", [1]) as a:
        for i in range(n):
            a.token(rb.digest, i, -0.01)
    lps = [-0.01] * n
    lps[3] = -9.0
    rec.score(a.record.digest, rt.digest, lps)
    with rec.sequence(0, "b", [1]) as b:
        b.token(rb.digest, 5, -1.0)
    rec.score(b.record.digest, r3.digest, [-1.5])
    return rec


RAW_ENGINE = SamplerConfig(temperature=0.7, logprobs_mode="raw_logprobs", logprobs_source="engine_sampling", temperature_applied=False)
PROCESSED_ENGINE = SamplerConfig(temperature=0.7, logprobs_mode="processed_logprobs", logprobs_source="engine_sampling",
                                 temperature_applied=True)
TRAINER = SamplerConfig(temperature=0.7, logprobs_source="trainer_scores", temperature_applied=True)
UNKNOWN_ENGINE = SamplerConfig(temperature=0.7, logprobs_mode="engine_sampling")          # a pre-change record


class TestComparability:
    def test_raw_engine_against_a_temperature_scaled_trainer_is_not_comparable_off_temperature_one(self):
        reason = floor_comparability(RAW_ENGINE, TRAINER)
        assert reason is not None and "temperature 0.7" in reason and "behavior logprobs_mode='raw_logprobs'" in reason
        assert floor_comparability(PROCESSED_ENGINE, TRAINER) is None
        assert floor_comparability(SamplerConfig(**{**RAW_ENGINE.to_dict(), "temperature": 1.0}),
                                   SamplerConfig(**{**TRAINER.to_dict(), "temperature": 1.0})) is None

    def test_both_sides_scaled_by_different_temperatures_is_not_comparable(self):
        hot = SamplerConfig(**{**TRAINER.to_dict(), "temperature": 1.0})
        assert "temperature change" in (floor_comparability(PROCESSED_ENGINE, hot) or "")
        assert floor_comparability(SamplerConfig(**{**RAW_ENGINE.to_dict(), "temperature": 1.0}), TRAINER) is not None

    def test_unknown_side_is_reported_with_a_caveat_not_withheld(self):
        assert floor_comparability(UNKNOWN_ENGINE, TRAINER) is None
        assert floor_caveat(UNKNOWN_ENGINE, TRAINER) is not None and floor_caveat(PROCESSED_ENGINE, TRAINER) is None
        assert floor_caveat(SamplerConfig(temperature=1.0), SamplerConfig(temperature=1.0)) is None


class TestDoctor:
    def test_floor_withheld_in_one_sentence_and_no_attribution(self, tmp_path):
        d = decompose(_record(tmp_path, RAW_ENGINE, TRAINER).ledger)
        assert d["lag0_floor"] is None and d["lag0_floor_withheld"]["n_tokens"] == 40
        assert [b["lag"] for b in d["by_lag"]] == [0, 2]                      # the bucket is still what the loss saw
        assert attribute(d)["staleness_share"] is None and attribute(d)["floor_source"] is None
        lines = diagnosis(d)
        assert sum("lag-0 floor withheld" in line for line in lines) == 1
        assert not any("No lag-0 tokens" in line or "DIFFERENT WEIGHTS" in line for line in lines)
        assert "lag-0 floor withheld" in render_markdown(d)
        assert d["by_provenance"]["fresh"]["lag0_floor"] is None

    def test_like_for_like_floor_is_reported_and_names_the_disagreement(self, tmp_path):
        d = decompose(_record(tmp_path, PROCESSED_ENGINE, TRAINER).ledger)
        assert d["lag0_floor"]["n_confident_disagreements"] == 1 and d["lag0_floor_withheld"] is None
        assert d["lag0_floor_caveat"] is None
        assert any("DIFFERENT WEIGHTS" in line for line in diagnosis(d))

    def test_pre_change_record_keeps_its_floor_with_a_caveat(self, tmp_path):
        d = decompose(_record(tmp_path, UNKNOWN_ENGINE, SamplerConfig(temperature=0.7, logprobs_mode="trainer_scores")).ledger)
        assert d["lag0_floor"] is not None and d["lag0_floor_withheld"] is None
        assert "may include a temperature effect" in d["lag0_floor_caveat"]
        assert any("may include a temperature effect" in line for line in diagnosis(d))

    def test_stale_server_alarm_does_not_fire_on_a_withheld_floor(self, tmp_path):
        rec = _record(tmp_path, RAW_ENGINE, TRAINER)
        diag = IncrementalDiagnosis(rec.ledger).update(step=3)
        assert diag.floor_withheld is not None and diag.floor_mean_abs is None
        m = diag.metrics()
        assert m["martingale/floor_withheld"] == 1.0 and "martingale/floor_mean_abs_log_ratio" not in m
        assert not any(k.endswith("lag0") for k in m)                   # no lag-0 number reaches the trainer log
        assert "martingale/ess_fraction_lag2" in m
        history = AlarmHistory()
        kinds = [a.kind for a in evaluate(diag, history, CFG)]
        assert kinds == ["floor_withheld"] and history.floors == []
        assert evaluate(IncrementalDiagnosis(rec.ledger).update(step=3), history, CFG) == []   # reported once

    def test_inspector_steps_withhold_the_floor_too(self, tmp_path):
        from martingale.inspector.view import RecordView
        steps = RecordView(_record(tmp_path, RAW_ENGINE, TRAINER).ledger).derived().steps
        assert [s["generation_step"] for s in steps] == [1]
        assert steps[0]["lag0_floor"] is None and steps[0]["n_confident_disagreements_lag0"] == 0
        assert "lag-0 floor withheld" in steps[0]["lag0_floor_withheld"]
        ok = RecordView(_record(tmp_path / "ok", PROCESSED_ENGINE, TRAINER).ledger).derived().steps
        assert ok[0]["lag0_floor"] is not None and ok[0]["n_confident_disagreements_lag0"] == 1

    def test_stale_server_alarm_still_fires_when_like_for_like(self, tmp_path):
        rec = _record(tmp_path, PROCESSED_ENGINE, TRAINER)
        diag = IncrementalDiagnosis(rec.ledger).update(step=3)
        assert diag.floor_withheld is None and diag.floor_mean_abs is not None
        assert [a.kind for a in evaluate(diag, AlarmHistory(), CFG)] == ["stale_server"]
