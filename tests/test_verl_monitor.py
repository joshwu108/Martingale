"""The shared live doctor wired to verl-style metrics sinks."""
import pytest

torch = pytest.importorskip("torch")

from martingale.diagnostics import AlarmConfig, MartingaleAlarmError
from martingale.integrations._trl_callback import HEAD_FILE, MartingaleMonitor
from martingale.integrations.verl import MartingaleVerlRecorder
from tests.test_verl_integration import make_batch


def test_metrics_and_checkpoint_head(tmp_path):
    flight = MartingaleVerlRecorder(tmp_path / "ws", tokenizer=b"t")
    monitor = MartingaleMonitor(flight, AlarmConfig(min_floor_tokens=1))
    weights = {"w": torch.ones(2)}
    metrics = []
    for step in range(3):
        batch = make_batch()
        flight.on_rollout(batch, step=step, weights=weights)
        flight.on_train_step(batch, batch.batch["rollout_log_probs"], step=step, weights=weights)
        monitor.on_step_end(step + 1, metrics.append)
        weights["w"] += 1
    assert len(metrics) == 3
    assert "martingale/floor_mean_abs_log_ratio" in metrics[0]
    assert all(m["martingale/alarms"] == 0 for m in metrics)
    checkpoint = tmp_path / "checkpoint-3"
    checkpoint.mkdir()
    monitor.on_save(checkpoint)
    assert (checkpoint / HEAD_FILE).read_text().strip() == flight.head()


def test_unchanged_weights_raise_alarm(tmp_path):
    flight = MartingaleVerlRecorder(tmp_path / "ws", tokenizer=b"t")
    monitor = MartingaleMonitor(flight, AlarmConfig(min_floor_tokens=1))
    weights = {"w": torch.ones(2)}
    for step in range(3):
        batch = make_batch()
        flight.on_rollout(batch, step=step, weights=weights)
        flight.on_train_step(batch, batch.batch["rollout_log_probs"], step=step, weights=weights)
        if step != 1:
            weights["w"] += 1
        if step < 2:
            monitor.on_step_end(step + 1)
    with pytest.raises(MartingaleAlarmError, match="weights_unchanged"):
        monitor.on_step_end(3)
