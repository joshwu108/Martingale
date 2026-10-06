"""martingale.diagnostics — staleness-versus-mismatch decomposition of a token record."""
from martingale.diagnostics.alarms import Alarm, AlarmConfig, AlarmHistory, MartingaleAlarmError, evaluate
from martingale.diagnostics.incremental import IncrementalDiagnosis, RunningBucket, StepDiagnosis
from martingale.diagnostics.report import render_markdown
from martingale.diagnostics.staleness import (
    DEFAULT_EPS,
    Bucket,
    ScoredToken,
    attribute,
    decompose,
    diagnosis,
    scored_tokens,
)

__all__ = ["DEFAULT_EPS", "Alarm", "AlarmConfig", "AlarmHistory", "Bucket", "IncrementalDiagnosis", "MartingaleAlarmError",
           "RunningBucket", "ScoredToken", "StepDiagnosis", "attribute", "decompose", "diagnosis", "evaluate",
           "render_markdown", "scored_tokens"]
