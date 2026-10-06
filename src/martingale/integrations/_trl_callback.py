"""
integrations/_trl_callback.py — the doctor as a trainer callback.

`MartingaleMonitor` is framework-neutral: call `on_step_end(step, metrics_sink)`
after each optimizer step and `on_save(dir)` at each checkpoint. `build_callback_class`
wraps it in TRL's `TrainerCallback`; `MartingaleGRPOTrainer` adds it automatically.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Callable

from martingale.diagnostics.alarms import Alarm, AlarmConfig, AlarmHistory, MartingaleAlarmError, evaluate
from martingale.diagnostics.incremental import IncrementalDiagnosis, StepDiagnosis

log = logging.getLogger("martingale")
HEAD_FILE = "martingale_head.txt"


class MartingaleMonitor:
    def __init__(self, flight: Any, config: AlarmConfig | None = None, *, halt_on_error: bool = True,
                 metric_prefix: str = "martingale/") -> None:
        self.flight = flight
        self.config = config or AlarmConfig()
        self.halt_on_error = halt_on_error
        self.prefix = metric_prefix
        self.history = AlarmHistory()
        self.alarms: list[Alarm] = []
        self.diagnoses: list[StepDiagnosis] = []
        self._inc = IncrementalDiagnosis(flight.recorder.ledger)
        self._matched_seen = 0
        self._unmatched_seen = 0

    def on_step_end(self, step: int, metrics_sink: Callable[[dict[str, float]], None] | None = None) -> StepDiagnosis:
        diag = self._inc.update(step)
        stats = getattr(self.flight, "stats", {})
        unmatched = stats.get("unmatched_rows", 0) - self._unmatched_seen
        matched = stats.get("scored_sequences", stats.get("sequences", 0))
        new_alarms = evaluate(diag, self.history, self.config, unmatched_rows=unmatched,
                              matched_rows=max(0, matched - self._matched_seen))
        self._unmatched_seen = stats.get("unmatched_rows", 0)
        self._matched_seen = matched
        self.alarms.extend(new_alarms)
        self.diagnoses.append(diag)
        metrics = diag.metrics(self.prefix)
        metrics[self.prefix + "alarms"] = float(len(new_alarms))
        if metrics_sink is not None:
            metrics_sink(metrics)
        for a in new_alarms:
            (log.error if a.level == "error" else log.warning)("martingale %s: %s", a.kind, a.message)
            warnings.warn(f"martingale {a.kind} at step {a.step}: {a.message}", RuntimeWarning, stacklevel=2)
        errors = [a for a in new_alarms if a.level == "error"]
        if errors and self.halt_on_error:
            raise MartingaleAlarmError(errors)
        return diag

    def on_save(self, checkpoint_dir: str | Path | None) -> str:
        """Write the ledger head next to the checkpoint so a resumed run anchors on the right head."""
        head = self.flight.recorder.ledger.head()
        if checkpoint_dir is not None:
            d = Path(checkpoint_dir)
            if d.is_dir():
                (d / HEAD_FILE).write_text(head + "\n")
        return head


def build_callback_class(trainer_callback_base: type) -> type:
    """A transformers TrainerCallback that delegates to a MartingaleMonitor."""

    class MartingaleTrainerCallback(trainer_callback_base):  # type: ignore[misc,valid-type]
        def __init__(self, monitor: MartingaleMonitor) -> None:
            self.monitor = monitor
            self.trainer: Any = None

        def on_step_end(self, args, state, control, **kwargs):
            trainer = self.trainer

            def sink(metrics: dict[str, float]) -> None:
                m = getattr(trainer, "_metrics", None)
                if isinstance(m, dict) and "train" in m:
                    for k, v in metrics.items():
                        m["train"][k].append(v)
            self.monitor.on_step_end(int(state.global_step), sink)
            return control

        def on_save(self, args, state, control, **kwargs):
            out = getattr(args, "output_dir", None)
            ckpt = None if out is None else Path(out) / f"checkpoint-{state.global_step}"
            self.monitor.on_save(ckpt)
            return control

    MartingaleTrainerCallback.__module__ = __name__
    return MartingaleTrainerCallback
