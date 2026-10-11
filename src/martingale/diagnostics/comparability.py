"""
diagnostics/comparability.py — is a lag-0 log-ratio a like-with-like comparison?

A lag-0 token was scored by the weights that generated it, so its log-ratio
should be engine-versus-trainer mismatch. That holds only if both sides report
the same function of the logits. vLLM reports either the raw log-softmax
(`raw_logprobs`) or the log-softmax after temperature and the sampling filters
(`processed_logprobs`); TRL 1.13 and verl 0.9.1 divide the trainer's logits by
the sampling temperature before their log-softmax. At temperature != 1 a raw
engine log-prob against a temperature-scaled trainer log-prob measures the
temperature, not the mismatch, and the floor must not be read.

Every decision here is an exact comparison of recorded config values; no
floats are derived. Unknown (`None`) on either side means the floor is
reported as it was before the keys existed, with a note (docs/nonclaims.md).
"""
from __future__ import annotations

from typing import Iterable

from martingale.record.revision import SamplerConfig


def floor_comparability(behavior: SamplerConfig, train: SamplerConfig) -> str | None:
    """None when the lag-0 ratio compares like with like, else one sentence saying why it does not.

    Withheld only on a known disagreement: `temperature_applied` recorded on both sides and
    different, at a sampling temperature other than 1 (temperature 1 makes the scaling a no-op);
    or both sides scaled, by two different recorded temperatures.
    """
    b, s = behavior.temperature_applied, train.temperature_applied
    if b is None or s is None:
        return None
    tb, ts = behavior.temperature, train.temperature
    if b and s and tb is not None and ts is not None and tb != ts:
        return (f"lag-0 floor withheld: the behavior log-probs are scaled by temperature {tb!r} and the trainer's "
                f"by {ts!r}, so their ratio measures the temperature change, not engine mismatch")
    if b == s:
        return None
    scaled_t, other_t = (tb, ts) if b else (ts, tb)     # the temperature that was actually divided by
    t = scaled_t if scaled_t is not None else other_t
    if t is None or t == 1:
        return None
    scaled, unscaled = ("behavior", "trainer") if b else ("trainer", "behavior")
    return (f"lag-0 floor withheld: at temperature {t!r} the {scaled} log-probs are divided by the temperature "
            f"and the {unscaled} ones are not (behavior logprobs_mode={behavior.logprobs_mode!r}, "
            f"source={behavior.logprobs_source!r}), so their ratio measures the temperature, not engine mismatch; "
            "pass sampler_overrides or set the engine's logprobs_mode so both sides agree")


def floor_caveat(behavior: SamplerConfig, train: SamplerConfig) -> str | None:
    """A note when the floor is reported without knowing whether both sides scaled by temperature."""
    t = behavior.temperature if behavior.temperature is not None else train.temperature
    if t is None or t == 1:
        return None
    if behavior.temperature_applied is not None and train.temperature_applied is not None:
        return None
    return (f"temperature is {t!r} and the record does not say whether both the behavior and the trainer "
            f"log-probs were divided by it (behavior logprobs_mode={behavior.logprobs_mode!r}, "
            f"temperature_applied={behavior.temperature_applied!r}; trainer {train.temperature_applied!r}): "
            "the lag-0 floor may include a temperature effect")


def lag0_verdict(pairs: Iterable[tuple[SamplerConfig, SamplerConfig]]) -> tuple[str | None, str | None]:
    """(withheld reason, caveat) over every (behavior, trainer) sampler pair seen at lag 0.

    The first disagreement withholds the floor for the whole record (one floor, one verdict); the
    caveat is reported only when the floor is. Pass the pairs in a deterministic order."""
    withheld: str | None = None
    caveat: str | None = None
    for behavior, train in pairs:
        withheld = withheld or floor_comparability(behavior, train)
        caveat = caveat or floor_caveat(behavior, train)
    return withheld, (None if withheld else caveat)
