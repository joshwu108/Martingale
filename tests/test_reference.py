"""martingale recompute against a fake score function: stale engine vs fine vs engine-wrong."""
from martingale.diagnostics.reference import recompute, render
from martingale.record import Recorder, SamplerConfig

S = SamplerConfig(temperature=1.0)
TOK = "f" * 64


def _record(tmp_path, engine_lp, trainer_lp):
    rec = Recorder(tmp_path / "ws")
    r = rec.publish_revision("1" * 64, TOK, S, step=3)
    with rec.sequence(0, "a", [5, 6, 7]) as a:
        for i, lp in enumerate(engine_lp):
            a.token(r.digest, 100 + i, lp)
    rec.score(a.record.digest, r.digest, trainer_lp)
    return rec


def test_engine_stale_when_reference_matches_engine_not_trainer(tmp_path):
    rec = _record(tmp_path, engine_lp=[-0.01, -0.02, -0.03], trainer_lp=[-9.0, -0.02, -0.03])
    rep = recompute(rec.ledger, lambda p, c: [-0.01, -0.02, -0.03])
    g = rep["generations"][0]
    assert g["generation_step"] == 3 and g["n_tokens"] == 3
    assert "ENGINE GENERATED FROM THE REFERENCE WEIGHTS" in g["verdict"]
    assert "stale" in render(rep)


def test_both_match_is_numerics(tmp_path):
    rec = _record(tmp_path, engine_lp=[-0.01, -0.02], trainer_lp=[-0.011, -0.021])
    rep = recompute(rec.ledger, lambda p, c: [-0.0105, -0.0205])
    assert "numerics" in rep["generations"][0]["verdict"]


def test_prompt_ids_are_passed_and_records_without_them_are_skipped(tmp_path):
    seen = {}
    rec = _record(tmp_path, engine_lp=[-0.1], trainer_lp=[-0.1])
    recompute(rec.ledger, lambda p, c: seen.setdefault("args", (list(p), list(c))) and [-0.1])
    assert seen["args"] == ([5, 6, 7], [100])
    rec2 = Recorder(tmp_path / "ws2", keep_prompt_ids=False)
    r = rec2.publish_revision("1" * 64, TOK, S, step=0)
    with rec2.sequence(0, "a", [1]) as a:
        a.token(r.digest, 1, -0.1)
    rep = recompute(rec2.ledger, lambda p, c: [-0.1])
    assert rep["sequences_skipped_without_prompt_ids"] == 1 and rep["generations"] == []


def test_checker_verifies_prompt_ids_against_digest(tmp_path):
    import json

    from checker.verify_tokens import verify_export
    rec = _record(tmp_path, engine_lp=[-0.1], trainer_lp=[-0.1])
    exp = rec.export_for_checker(tmp_path / "exp")
    assert verify_export(exp, expected_head=rec.ledger.head())["ok"]
    f = next((exp / "sequences").glob("*.json"))
    d = json.loads(f.read_text()); d["prompt_ids"] = [5, 6, 8]; f.write_text(json.dumps(d))
    r = verify_export(exp)
    assert not r["ok"] and any("prompt_ids" in e for errs in r["errors"].values() for e in errs)
