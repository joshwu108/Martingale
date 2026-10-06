"""The monitor on a fake three-step trainer: metrics in TRL's log, a skipped sync alarm, halt, checkpoint head."""
from collections import defaultdict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from martingale.diagnostics import AlarmConfig, MartingaleAlarmError
from martingale.integrations._trl_callback import HEAD_FILE, MartingaleMonitor, build_callback_class
from martingale.integrations.trl import MartingaleRecorder
from tests.test_trl_integration import FakeTrainer, make_output


def _train_step(flight, tr, update_weights=True):
    """generation -> loss scoring -> optimizer step, as TRL orders them."""
    out = flight.on_generation(make_output(), tr)
    ids = torch.cat([out["prompt_ids"], out["completion_ids"]], dim=1)
    am = torch.cat([out["prompt_mask"], out["completion_mask"]], dim=1)
    lp = out["old_per_token_logps"] + 1e-6
    flight.on_scores(ids, am, 3, lp, tr, row_ids=out["martingale_row_id"])
    if update_weights:
        with torch.no_grad():
            tr.model.weight.add_(0.1)
    tr.state.global_step += 1


@pytest.fixture
def flight(tmp_path):
    return MartingaleRecorder(tmp_path / "ws", tokenizer=b"t")


def test_metrics_land_in_trainer_log_and_no_alarms_on_healthy_run(flight):
    tr = FakeTrainer()
    tr._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    monitor = MartingaleMonitor(flight, AlarmConfig(min_floor_tokens=1))
    cb = build_callback_class(object)(monitor)
    cb.trainer = tr
    for _ in range(3):
        _train_step(flight, tr)
        cb.on_step_end(tr.args, tr.state, None)
    m = tr._metrics["train"]
    assert len(m["martingale/alarms"]) == 3 and sum(m["martingale/alarms"]) == 0
    assert "martingale/floor_mean_abs_log_ratio" in m and "martingale/ess_fraction_lag0" in m
    assert monitor.alarms == [] and len(monitor.diagnoses) == 3


def test_skipped_weight_update_raises_weights_unchanged(flight):
    tr = FakeTrainer()
    monitor = MartingaleMonitor(flight, AlarmConfig(min_floor_tokens=1))
    _train_step(flight, tr)
    monitor.on_step_end(tr.state.global_step)
    _train_step(flight, tr, update_weights=False)          # optimizer "ran" but weights identical
    monitor.on_step_end(tr.state.global_step)
    _train_step(flight, tr)
    with pytest.raises(MartingaleAlarmError) as e:
        monitor.on_step_end(tr.state.global_step)
    assert e.value.alarms[0].kind == "weights_unchanged"


def test_halt_can_be_disabled_and_alarms_still_recorded(flight):
    tr = FakeTrainer()
    monitor = MartingaleMonitor(flight, AlarmConfig(min_floor_tokens=1), halt_on_error=False)
    _train_step(flight, tr); monitor.on_step_end(tr.state.global_step)
    _train_step(flight, tr, update_weights=False); monitor.on_step_end(tr.state.global_step)
    _train_step(flight, tr)
    with pytest.warns(RuntimeWarning, match="weights_unchanged"):
        monitor.on_step_end(tr.state.global_step)
    assert [a.kind for a in monitor.alarms] == ["weights_unchanged"]


def test_on_save_writes_head_next_to_checkpoint(flight, tmp_path):
    tr = FakeTrainer()
    tr.args.output_dir = str(tmp_path / "out")
    monitor = MartingaleMonitor(flight)
    cb = build_callback_class(object)(monitor)
    cb.trainer = tr
    _train_step(flight, tr)
    ckpt = tmp_path / "out" / f"checkpoint-{tr.state.global_step}"
    ckpt.mkdir(parents=True)
    cb.on_save(tr.args, tr.state, None)
    assert (ckpt / HEAD_FILE).read_text().strip() == flight.head()
