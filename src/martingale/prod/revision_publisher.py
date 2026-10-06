"""
prod/revision_publisher.py — publish model checkpoints as LLMRevisions.

Replaces the 2026-10-05 prototype that stored a checkpoint as a fake 1x1
rational table (docs/nonclaims.md, "Production layer"). A revision is now the
content-addressed manifest of design.md section 7: weights digest over raw
tensor bytes, tokenizer digest, sampler config, learner step, parent.

    rec = Recorder("./ws")
    pub = RevisionPublisher(rec, tokenizer=tokenizer_json_bytes, sampler=SamplerConfig(temperature=1.0))
    rev = pub.publish(model)            # step auto-increments; identical weights -> same revision
    rev = pub.publish(model, step=120)  # or bind the learner's own step counter
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Union

from martingale.record.bits import GENESIS_DIGEST
from martingale.record.recorder import Recorder
from martingale.record.revision import LLMRevision, SamplerConfig, weights_digest

if TYPE_CHECKING:  # torch is optional at import time
    import torch.nn as nn

NO_TOKENIZER = GENESIS_DIGEST
"""Tokenizer digest for non-LLM policies (tabular / vector observations)."""


def checkpoint_digest(model_or_state_dict: Union[nn.Module, Mapping[str, Any]]) -> str:
    """BLAKE2b-256 over sorted (name, dtype, shape, raw bytes) of every parameter.

    Stable across torch versions (unlike a digest of torch.save output) and
    equal for a module and its own state_dict.
    """
    state_dict = model_or_state_dict.state_dict() if hasattr(model_or_state_dict, "state_dict") else model_or_state_dict
    return weights_digest(state_dict)


class RevisionPublisher:
    def __init__(self, recorder: Recorder, tokenizer: bytes | str = NO_TOKENIZER,
                 sampler: SamplerConfig | None = None) -> None:
        self._rec = recorder
        self._tokenizer = tokenizer
        self._sampler = sampler or SamplerConfig()
        last = recorder.ledger.max_step()
        self._step = 0 if last is None else last + 1   # survives a restart on an existing ledger

    @property
    def recorder(self) -> Recorder:
        return self._rec

    @property
    def latest(self) -> LLMRevision | None:
        return self._rec.latest_revision

    @property
    def latest_digest(self) -> str | None:
        rev = self._rec.latest_revision
        return None if rev is None else rev.digest

    def publish(self, model_or_state_dict: Union[nn.Module, Mapping[str, Any]], step: int | None = None,
                weights_uri: str | None = None) -> LLMRevision:
        """Publish the weights as a revision. Unchanged weights with no explicit step reuse the
        latest revision rather than minting a new one (idempotent per checkpoint)."""
        wd = checkpoint_digest(model_or_state_dict)
        latest = self._rec.latest_revision
        if step is None and latest is not None and latest.weights_digest == wd \
                and latest.sampler == self._sampler:
            return latest
        if step is None:
            step = self._step if latest is None else latest.step + 1
        rev = self._rec.publish_revision(wd, self._tokenizer, self._sampler, step=step, weights_uri=weights_uri)
        self._step = step
        return rev
