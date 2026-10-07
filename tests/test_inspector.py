"""The rollout inspector's API over the real record and the demo record (FastAPI TestClient)."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from martingale.server import create_app
from tests.conftest import REAL_RECORD, needs_real_record

SEQ_172 = [16, 22, 17, 151645]   # "1", "7", "2", "<|im_end|>" in the Qwen2.5 vocabulary


class StubTokenizer:
    """Duck-typed stand-in for a transformers tokenizer: decode() only."""

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(f"<{int(i)}>" for i in ids)


@pytest.fixture(scope="module")
def real() -> TestClient:
    if not (REAL_RECORD / "tokens.db").exists():
        pytest.skip("real record not present")
    return TestClient(create_app(REAL_RECORD, tokenizer=StubTokenizer()))


@pytest.fixture(scope="module")
def demo(demo_ws: Path) -> TestClient:
    return TestClient(create_app(demo_ws))


# ---- panel 4: record and trust -------------------------------------------------------

@needs_real_record
def test_record_counts_head_and_anchored_checker_on_real_record(real):
    r = real.get("/api/record").json()
    assert (r["revisions"], r["sequences"], r["tokens"], r["scores"]) == (20, 64, 240, 480)
    assert r["head"] == (REAL_RECORD / "head.txt").read_text().strip()
    assert r["head_source"].endswith("head.txt")
    assert r["checker"]["anchored"] is True and r["checker"]["ok"] is True
    assert r["checker"]["expected_head"] == r["head"] and r["checker"]["errors"] == {}


def test_record_on_demo_is_unanchored_without_a_stored_head(demo):
    r = demo.get("/api/record").json()
    assert (r["revisions"], r["sequences"], r["tokens"], r["scores"]) == (4, 48, 384, 384)
    assert r["checker"]["anchored"] is False and r["checker"]["ok"] is True
    assert r["head_source"] is None


def test_record_anchors_on_an_explicit_head(demo_ws):
    from martingale.record import TokenLedger
    head = TokenLedger(demo_ws / "tokens.db").head()
    client = TestClient(create_app(demo_ws, head=head))
    r = client.get("/api/record").json()
    assert r["checker"]["anchored"] is True and r["checker"]["ok"] is True
    wrong = TestClient(create_app(demo_ws, head="0" * 64)).get("/api/record").json()
    assert wrong["checker"]["ok"] is False and "head" in wrong["checker"]["errors"]


def test_record_reads_checkpoint_head_file(demo_ws, tmp_path):
    import shutil

    from martingale.integrations._trl_callback import HEAD_FILE
    from martingale.record import TokenLedger
    ws = tmp_path / "ws"
    shutil.copytree(demo_ws, ws)
    (ws / "checkpoint-5").mkdir()
    (ws / "checkpoint-5" / HEAD_FILE).write_text(TokenLedger(ws / "tokens.db").head() + "\n")
    r = TestClient(create_app(ws)).get("/api/record").json()
    assert r["checker"]["anchored"] is True and r["checker"]["ok"] is True
    assert r["head_source"].endswith(HEAD_FILE)


# ---- panel 1: diagnosis ---------------------------------------------------------------

@needs_real_record
def test_doctor_reproduces_the_hand_analysis_on_the_real_record(real):
    d = real.get("/api/doctor").json()
    dec = d["decomposition"]
    assert dec["lags"] == [0, 1, 2, 3]
    assert [b["n_tokens"] for b in dec["by_lag"]] == [120, 120, 120, 120]
    assert dec["lag0_floor"]["n_confident_disagreements"] == 6
    assert Fraction(dec["lag0_floor"]["mean_abs_log_ratio"]) > Fraction(4, 10)   # exact string, parsed here only
    assert d["attribution"]["staleness_share"] is not None
    assert any("DIFFERENT WEIGHTS" in line for line in d["diagnosis"])
    stale = [a for a in d["alarms"] if a["kind"] == "stale_server"]
    assert {a["step"] for a in stale} == {5, 9}
    assert all(a["level"] == "error" for a in stale)
    assert d["alarm_config"]["confident_min_tokens"] == 1


def test_doctor_on_demo_has_lines_and_a_list_of_alarms(demo):
    d = demo.get("/api/doctor").json()
    assert d["decomposition"]["n_sequences"] == 48 and d["decomposition"]["lags"] == [0, 2, 3, 4]
    assert d["diagnosis"] and isinstance(d["alarms"], list)
    assert d["decomposition"]["by_lag"][0]["float_informational"]["mean_abs_log_ratio"] < 1e-5


@needs_real_record
def test_steps_are_per_generation_step_with_floor_ess_and_confident_counts(real):
    s = real.get("/api/steps").json()
    assert [x["generation_step"] for x in s["steps"]] == [0, 4, 8, 12]
    assert all(x["n_sequences"] == 16 for x in s["steps"])
    by_step = {x["generation_step"]: x for x in s["steps"]}
    assert by_step[8]["n_confident_disagreements_lag0"] == 3 and by_step[4]["n_confident_disagreements_lag0"] == 3
    assert by_step[0]["n_confident_disagreements_lag0"] == 0
    assert set(by_step[0]["float_informational"]["ess_fraction_by_lag"]) == {"0", "1", "2", "3"}
    assert by_step[0]["lag0_floor"] is not None and by_step[0]["float_informational"]["lag0_floor"] < 1e-2
    assert by_step[8]["float_informational"]["lag0_floor"] > 0.1
    assert [r["step"] for r in s["replay"]] == list(range(1, 17))
    assert [r["n_new_scores"] for r in s["replay"]][:4] == [34, 30, 34, 30]
    assert "stale_server" in s["replay"][4]["alarms"]


def test_steps_on_demo(demo):
    s = demo.get("/api/steps").json()
    assert [x["generation_step"] for x in s["steps"]] == [0, 1, 2, 4]
    assert sum(x["n_sequences"] for x in s["steps"]) == 48
    assert len(s["replay"]) == 1 and s["replay"][0]["step"] == 5


# ---- panel 2: sequences ---------------------------------------------------------------

@needs_real_record
def test_sequences_sorted_by_confident_disagreements_surface_the_172_rows_first(real):
    r = real.get("/api/sequences", params={"sort": "confident_disagreements", "lag": 0}).json()
    assert r["total"] == 64 and r["lag"] == 0 and r["lags"] == [0, 1, 2, 3]
    rows = r["rows"]
    assert len(rows) == 64
    flagged = [x for x in rows if x["confident_disagreements"].get("0", 0) > 0]
    assert len(flagged) == 6 and rows[:6] == flagged
    assert {x["generation_step"] for x in flagged} == {4, 8}
    row = next(x for x in flagged if x["sequence_id"] == "step8/row43")
    assert row["length"] == 4 and row["mixed_revision"] is False and row["prompt_len"] == 45
    assert row["text"] == "<16><22><17><151645>"
    assert Fraction(row["mean_abs_log_ratio"]["0"]) > 2
    assert row["float_informational"]["mean_abs_log_ratio"]["0"] > 2.0
    assert "reward" in row and row["float_informational"]["reward"] is not None


@needs_real_record
def test_sequences_filter_by_step_and_paginate(real):
    r = real.get("/api/sequences", params={"step": 8, "limit": 5, "offset": 3, "sort": "length"}).json()
    assert r["total"] == 16 and len(r["rows"]) == 5
    assert all(x["generation_step"] == 8 for x in r["rows"])
    full = real.get("/api/sequences", params={"step": 8, "sort": "length"}).json()["rows"]
    assert r["rows"] == full[3:8]
    lengths = [x["length"] for x in full]
    assert lengths == sorted(lengths, reverse=True)
    asc = real.get("/api/sequences", params={"step": 8, "sort": "length", "order": "asc"}).json()["rows"]
    assert [x["length"] for x in asc] == sorted(lengths)


@needs_real_record
def test_sequences_sort_by_mean_abs_log_ratio_at_the_selected_lag(real):
    for lag in (0, 3):
        rows = real.get("/api/sequences", params={"sort": "mean_abs_log_ratio", "lag": lag}).json()["rows"]
        vals = [Fraction(x["mean_abs_log_ratio"][str(lag)]) for x in rows if str(lag) in x["mean_abs_log_ratio"]]
        assert vals == sorted(vals, reverse=True) and len(vals) > 0
        unscored = [x for x in rows if str(lag) not in x["mean_abs_log_ratio"]]
        assert rows[len(vals):] == unscored     # sequences without a score at this lag sort last


def test_sequences_sort_by_advantage_uses_exact_reward(demo):
    rows = demo.get("/api/sequences", params={"sort": "advantage"}).json()["rows"]
    rewards = [Fraction(x["reward"]) for x in rows]
    assert rewards == sorted(rewards, reverse=True) and rewards[0] == 1 and rewards[-1] == 0
    assert rows[0]["text"] is None                      # no tokenizer configured


def test_sequences_rejects_unknown_sort_and_bad_lag(demo):
    assert demo.get("/api/sequences", params={"sort": "nope"}).status_code == 422
    assert demo.get("/api/sequences", params={"lag": "x"}).status_code == 422


def test_sequences_mixed_revision_flag_on_demo(demo):
    rows = demo.get("/api/sequences", params={"limit": 100}).json()["rows"]
    assert sum(x["mixed_revision"] for x in rows) == 8


# ---- panel 3: sequence detail -----------------------------------------------------------

@needs_real_record
def test_sequence_detail_shows_the_172_case(real):
    rows = real.get("/api/sequences", params={"sort": "confident_disagreements"}).json()["rows"]
    digest = next(x["digest"] for x in rows if x["sequence_id"] == "step8/row43")
    s = real.get(f"/api/sequence/{digest}").json()
    assert s["sequence_id"] == "step8/row43" and s["generation_step"] == 8 and s["lags"] == [0, 2]
    assert [t["token_id"] for t in s["tokens"]] == SEQ_172
    assert [t["text"] for t in s["tokens"]] == ["<16>", "<22>", "<17>", "<151645>"]
    assert s["text"] == "<16><22><17><151645>"
    assert s["prompt"]["ids"] is None and s["prompt"]["text"] is None and s["prompt"]["len"] == 45
    first = s["tokens"][0]
    assert first["revision_step"] == 8
    assert abs(first["float_informational"]["behavior_logprob"]) < 1e-4
    lag0 = first["scores"]["0"]
    assert lag0["train_step"] == 8 and lag0["confident_disagreement"] is True
    assert lag0["float_informational"]["logprob"] < -9 and lag0["float_informational"]["log_ratio"] < -9
    assert Fraction(lag0["log_ratio"]) == Fraction(lag0["logprob"]) - Fraction(first["behavior_logprob"])
    assert s["tokens"][1]["scores"]["0"]["confident_disagreement"] is False


def test_sequence_detail_decodes_prompt_ids_when_recorded(demo_ws):
    client = TestClient(create_app(demo_ws, tokenizer=StubTokenizer()))
    rows = client.get("/api/sequences", params={"limit": 1}).json()["rows"]
    s = client.get(f"/api/sequence/{rows[0]['digest']}").json()
    assert s["prompt"]["ids"] == [1, 2, 3, int(s["sequence_id"].split("-")[1])]
    assert s["prompt"]["text"] == "".join(f"<{i}>" for i in s["prompt"]["ids"])
    assert len(s["tokens"]) == 8 and all("4" in t["scores"] or "0" in t["scores"] or t["scores"] for t in s["tokens"])


def test_sequence_detail_404_on_unknown_digest(demo):
    assert demo.get("/api/sequence/" + "a" * 64).status_code == 404


# ---- the page -----------------------------------------------------------------------------

def test_page_is_served_with_the_four_panels(demo):
    r = demo.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    for pid in ("panel-diagnosis", "panel-sequences", "panel-sequence", "panel-record"):
        assert f'id="{pid}"' in r.text
    assert "cdn." not in r.text and "<script src=" not in r.text   # self-contained, no build step, no CDN


def test_empty_record_serves_every_panel(tmp_path):
    from martingale.record import Recorder
    Recorder(tmp_path)   # creates an empty tokens.db
    c = TestClient(create_app(tmp_path))
    d = c.get("/api/doctor").json()
    assert d["decomposition"]["lags"] == [] and d["alarms"] == [] and d["diagnosis"]
    assert c.get("/api/steps").json() == {"lags": [], "steps": [], "replay": []}
    assert c.get("/api/sequences").json()["total"] == 0
    r = c.get("/api/record").json()
    assert r["sequences"] == 0 and r["checker"]["ok"] is True
