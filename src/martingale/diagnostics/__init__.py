"""martingale.diagnostics — staleness-versus-mismatch decomposition of a token record."""
from martingale.diagnostics.advise import (
    Advice,
    AdviseConfig,
    HistoryAdvice,
    advise,
    advise_decomposition,
    advise_table,
)
from martingale.diagnostics.alarms import Alarm, AlarmConfig, AlarmHistory, MartingaleAlarmError, evaluate
from martingale.diagnostics.incremental import IncrementalDiagnosis, RunningBucket, StepDiagnosis
from martingale.diagnostics.reference import GenerationAgreement, ScoreFn, hf_score_fn, recompute
from martingale.diagnostics.report import render_markdown
from martingale.diagnostics.staleness import (
    DEFAULT_EPS,
    Bucket,
    ScoredToken,
    attribute,
    decompose,
    diagnosis,
    replayed_diagnosis,
    replayed_weights,
    scored_tokens,
)

__all__ = [
    "DEFAULT_EPS",
    "Advice",
    "advise",
    "advise_decomposition",
    "advise_table",
    "AdviseConfig",
    "Alarm",
    "AlarmConfig",
    "AlarmHistory",
    "attribute",
    "Bucket",
    "decompose",
    "diagnosis",
    "evaluate",
    "GenerationAgreement",
    "hf_score_fn",
    "HistoryAdvice",
    "IncrementalDiagnosis",
    "MartingaleAlarmError",
    "recompute",
    "render_markdown",
    "replayed_diagnosis",
    "replayed_weights",
    "RunningBucket",
    "scored_tokens",
    "ScoredToken",
    "ScoreFn",
    "StepDiagnosis",
]
