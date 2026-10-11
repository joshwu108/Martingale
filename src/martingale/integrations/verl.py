"""verl 0.9.1 rollout and actor-loss hooks, plus the older revision publisher shim.

The recorder and monitor can be imported without verl. `wrap_ppo_loss` checks
the pinned framework only when called inside a real worker.

Sampler config: pass ``rollout_config`` (``config.actor_rollout_ref.rollout``,
a DictConfig, the ``RolloutConfig`` dataclass or a plain dict) and the revision
binds verl's real settings. ``logprobs_mode`` is ``rollout.logprobs_mode``,
which verl 0.9.1 defaults to ``processed_logprobs`` (verl/workers/config/
rollout.py:197, verl/trainer/config/rollout/rollout.yaml:93) and hands to the
vLLM engine unchanged (verl/workers/rollout/vllm_rollout/vllm_async_server.py:338);
``rollout_log_probs`` exist only with ``rollout.calculate_log_probs`` (rollout.py:219).
The actor divides its logits by the rollout temperature before the log-softmax
(verl/workers/engine/fsdp/transformer_impl.py:98-103 and 1159, legacy path
verl/utils/torch_functional.py:679-681; the temperature comes from
verl/trainer/ppo/ray_trainer.py:1331), so ``trainer_scores`` revisions carry
``temperature_applied=True``. Without ``rollout_config`` the mode and the
temperature are recorded as unknown and the doctor reports the floor as before.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from martingale.integrations._common import (
    KEEP_GENERATIONS,
    ROW_ID_KEY,
    ContentKey,
    Generation,
    _check_digestable,
    _prefix_lengths,
    _tokenizer_bytes,
    apply_sampler_overrides,
    check_sampler_overrides,
    next_by_content,
    pack_row_id,
    resolve_row_id,
)
from martingale.integrations._trl_callback import MartingaleMonitor
from martingale.prod.revision_publisher import RevisionPublisher
from martingale.record import Recorder, SamplerConfig, temperature_applied_for, weights_digest
from martingale.record.revision import LLMRevision, tokenizer_digest

LENGTH_KEY = "martingale_completion_length"
ACTOR_TEMPERATURE_APPLIED = True   # verl 0.9.1 transformer_impl.py:98-103; torch_functional.py:679-681
__all__ = ["MartingaleVerlRecorder", "MartingaleMonitor", "wrap_ppo_loss", "MartingaleVeRLCallback",
           "sampler_from_rollout_config"]


def _cfg(cfg: Any, key: str) -> Any:
    """A field of a DictConfig, dataclass or dict; None when absent or missing-mandatory."""
    if cfg is None:
        return None
    try:
        return cfg.get(key) if isinstance(cfg, Mapping) else getattr(cfg, key, None)
    except Exception:            # omegaconf raises on a '???' value
        return None


def _engine_version(name: Any) -> str | None:
    try:
        if name == "vllm":
            import vllm
            return str(vllm.__version__)
        if name == "sglang":
            import sglang
            return str(sglang.__version__)
    except Exception:
        return None
    return None


def sampler_from_rollout_config(rollout_config: Any, logprobs_source: str) -> SamplerConfig:
    """verl's rollout settings bound to where the bits came from. ``logprobs_mode`` is the engine's
    mode for engine-produced bits and null for the actor's scores; ``temperature_applied`` follows the
    mode for the engine and is True for the actor (module docstring has the verl source lines)."""
    c = rollout_config
    engine_mode = _cfg(c, "logprobs_mode") if logprobs_source == "engine_sampling" else None
    applied = ACTOR_TEMPERATURE_APPLIED if logprobs_source == "trainer_scores" else temperature_applied_for(engine_mode)
    name, seed = _cfg(c, "name"), _cfg(c, "seed")
    return SamplerConfig(
        temperature=_cfg(c, "temperature"), top_p=_cfg(c, "top_p"), top_k=_cfg(c, "top_k"),
        max_tokens=_cfg(c, "response_length"), logprobs_mode=None if engine_mode is None else str(engine_mode),
        dtype=_cfg(c, "dtype"), engine="verl" if name is None else f"verl:{name}", engine_version=_engine_version(name),
        tensor_parallel=_cfg(c, "tensor_model_parallel_size"), seed_policy=None if seed is None else f"seed={seed}",
        logprobs_source=logprobs_source, temperature_applied=applied,
    )


class MartingaleVerlRecorder:
    """Two explicit hooks for verl's driver and actor loss path.

    Call on_rollout before correction changes response_mask. Call on_train_step
    from every actor loss invocation, with the log-probs used for that loss.
    `step` must identify each optimizer weight version, including PPO mini-batches.

    Row ids follow the TRL adapter's contract (``martingale.integrations._common``): an id packs
    the rollout's ``actor_id`` with that rank's counter and is bound in the ledger's row-id table,
    so a worker process with its own recorder over the same workspace resolves it there.
    """

    def __init__(self, workspace: str, tokenizer: Any, *, logprob_dtype: str = "f32",
                 sampler_overrides: dict | None = None, record_scores: bool = True,
                 rollout_config: Any = None) -> None:
        self.recorder = Recorder(workspace)
        tok = _tokenizer_bytes(tokenizer)
        self._tok_digest = tok if isinstance(tok, str) else tokenizer_digest(tok)
        self._dtype = logprob_dtype
        self._overrides = check_sampler_overrides(sampler_overrides or {})
        self._rollout = rollout_config
        self._record_scores = record_scores
        self._weights: dict[int, str] = {}
        self._revisions: dict[tuple[int, str], LLMRevision] = {}
        self._next_local: dict[int, int] = {}     # per rank; seeded from the ledger on first use
        self._generations: list[Generation] = []
        self._cross_content_next: dict[ContentKey, int] = {}
        self.stats: dict[str, int] = {"generation_calls": 0, "sequences": 0, "tokens": 0,
                                      "score_calls": 0, "scored_tokens": 0, "scored_sequences": 0,
                                      "unmatched_rows": 0, "nan_rows": 0, "duplicate_scores": 0,
                                      "revisions": 0}

    def _revision(self, step: int, weights: Mapping[str, Any] | Any, mode: str) -> LLMRevision:
        key = (step, mode)
        if key in self._revisions:
            return self._revisions[key]
        if step not in self._weights:
            state = weights if isinstance(weights, Mapping) else weights.state_dict()
            _check_digestable(state)
            self._weights[step] = weights_digest(state)
            if len(self._weights) > 8:
                self._weights.pop(min(self._weights))
        sampler = apply_sampler_overrides(sampler_from_rollout_config(self._rollout, mode), self._overrides, mode)
        rev = self.recorder.publish_revision(self._weights[step], self._tok_digest, sampler, step=step)
        self._revisions[key] = rev
        self.stats["revisions"] += 1
        return rev

    def _allocate(self, rank: int, n: int) -> list[int]:
        """The next ``n`` local ids of ``rank``, continuing any ids already bound in the workspace."""
        start = self._next_local.get(rank)
        if start is None:
            start = self.recorder.ledger.next_row_local_id(rank)
        self._next_local[rank] = start + n
        return list(range(start, start + n))

    @staticmethod
    def _tensor_batch(batch: Any) -> Any:
        return batch.batch if hasattr(batch, "batch") else batch

    def on_rollout(self, batch: Any, *, step: int, weights: Mapping[str, Any] | Any,
                   actor_id: int = 0) -> Any:
        """Record engine probabilities and add a row-id tensor to DataProto.batch."""
        import torch

        data = self._tensor_batch(batch)
        if "rollout_log_probs" not in data:
            raise ValueError("rollout_log_probs required: enable rollout.calculate_log_probs")
        prompt, response = data["prompts"], data["responses"]
        mask = data["response_mask"]
        prompt_mask = data["attention_mask"][:, :prompt.shape[1]]
        plen = _prefix_lengths(prompt_mask, "suffix")
        clen = _prefix_lengths(mask, "prefix")
        if data["rollout_log_probs"].shape != response.shape:
            raise ValueError("rollout_log_probs must match responses")
        rev = self._revision(int(step), weights, "engine_sampling")
        n = response.shape[0]
        local_ids = self._allocate(actor_id, n)
        row_ids = [pack_row_id(actor_id, i) for i in local_ids]
        by_id: dict[int, str] = {}
        by_content: dict[ContentKey, list[str]] = {}
        bound: list[tuple[int, int, str]] = []
        lp = data["rollout_log_probs"].detach().to("cpu").tolist()
        advantages = data.get("advantages")
        self.stats["generation_calls"] += 1
        for row in range(n):
            tokens = [int(x) for x in response[row, :clen[row]].tolist()]
            if not tokens:
                continue
            probs = lp[row][:clen[row]]
            if any(not math.isfinite(x) for x in probs):
                self.stats["nan_rows"] += 1
                continue
            prompt_ids = [int(x) for x in prompt[row, prompt.shape[1] - plen[row]:].tolist()]
            with self.recorder.sequence(actor_id, f"step{step}/row{local_ids[row]}", prompt_ids) as seq:
                for token, prob in zip(tokens, probs):
                    seq.token(rev.digest, token, prob, dtype=self._dtype)
                if advantages is not None:
                    adv = advantages[row]
                    seq.reward(float(adv[0] if adv.ndim else adv), dtype=self._dtype)
            digest = seq.record.digest
            by_id[row_ids[row]] = digest
            bound.append((actor_id, local_ids[row], digest))
            by_content.setdefault((tuple(prompt_ids), tuple(tokens)), []).append(digest)
            self.stats["sequences"] += 1
            self.stats["tokens"] += len(tokens)
        self.recorder.ledger.bind_row_ids(bound)
        data[ROW_ID_KEY] = torch.tensor(row_ids, dtype=torch.int64, device=response.device)
        data[LENGTH_KEY] = torch.tensor(clen, dtype=torch.int64, device=response.device)
        self._generations.append((by_id, by_content))
        del self._generations[:-KEEP_GENERATIONS]
        return batch

    def sequence_digest(self, row_id: int) -> str | None:
        """The recorded sequence behind a ``martingale_row_id``: the kept maps, then the ledger's
        row-id table (any rank, any age; a Ray worker opens the driver's workspace in a separate
        process with empty maps)."""
        return resolve_row_id(self._generations, self.recorder.ledger, row_id)

    def _lookup(self, row_id: int | None, key: ContentKey) -> str | None:
        """Row id first, exact (``sequence_digest``). Without an id, content: the kept maps, then a
        scan of the ledger's sequences for a worker that has no maps."""
        if row_id is not None:
            return self.sequence_digest(row_id)
        digest = next_by_content(self._generations, key)
        if digest is not None:
            return digest
        matches = []
        for seq in self.recorder.ledger.all_sequences():
            if seq.prompt_ids is not None and (
                tuple(seq.prompt_ids), tuple(t.token_id for t in seq.tokens)
            ) == key:
                matches.append(seq.digest)
        if matches:
            index = self._cross_content_next.get(key, 0)
            self._cross_content_next[key] = index + 1
            return matches[-KEEP_GENERATIONS:][index % min(len(matches), KEEP_GENERATIONS)]
        return None

    def on_train_step(self, batch: Any, logps: Any, *, step: int,
                      weights: Mapping[str, Any] | Any) -> int:
        """Score each loss row once per train revision, including reused rollouts."""
        if not self._record_scores:
            return 0
        data = self._tensor_batch(batch)
        prompt, response = data["prompts"], data["responses"]
        prompt_mask = data["attention_mask"][:, :prompt.shape[1]]
        plen = _prefix_lengths(prompt_mask, "suffix")
        saved_lengths = data.get(LENGTH_KEY)
        clen = (_prefix_lengths(data["response_mask"], "prefix") if saved_lengths is None
                else [int(x) for x in saved_lengths.detach().to("cpu").tolist()])
        if logps.shape != response.shape:
            raise ValueError("training logps must match responses")
        row_ids = data.get(ROW_ID_KEY)
        row_ids = None if row_ids is None else row_ids.detach().to("cpu").tolist()
        probs = logps.detach().to("cpu").tolist()
        self.stats["score_calls"] += 1
        rev = None
        matched = 0
        for row in range(response.shape[0]):
            key = (tuple(int(x) for x in prompt[row, prompt.shape[1] - plen[row]:].tolist()),
                   tuple(int(x) for x in response[row, :clen[row]].tolist()))
            digest = self._lookup(None if row_ids is None else int(row_ids[row]), key)
            if digest is None:
                self.stats["unmatched_rows"] += 1
                continue
            values = probs[row][:clen[row]]
            if any(not math.isfinite(x) for x in values):
                self.stats["nan_rows"] += 1
                continue
            if rev is None:
                rev = self._revision(int(step), weights, "trainer_scores")
            written = self.recorder.score(digest, rev.digest, values, dtype=self._dtype)
            if not written:
                self.stats["duplicate_scores"] += 1
                continue
            matched += 1
            self.stats["scored_sequences"] += 1
            self.stats["scored_tokens"] += clen[row]
        return matched

    def head(self) -> str:
        if self.stats["unmatched_rows"] or self.stats["nan_rows"]:
            import warnings
            warnings.warn("martingale: skipped or unmatched verl rows; see flight.stats", RuntimeWarning, stacklevel=2)
        return self.recorder.ledger.head()


def wrap_ppo_loss(loss_fn: Callable, flight: MartingaleVerlRecorder, *,
                  step: Callable[[], int], weights: Callable[[], Any]) -> Callable:
    """Install on the actor worker via `worker.set_loss_fn(wrap_ppo_loss(...))`."""
    def wrapped(config, model_output, data, dp_group=None):
        from martingale.integrations._verl_compat import require_verl

        convert = require_verl().no_padding_2_padding
        logps = convert(model_output["log_probs"], data)
        flight.on_train_step(data.to_padded_tensor(), logps, step=step(), weights=weights())
        return loss_fn(config, model_output, data, dp_group=dp_group)

    return wrapped


class MartingaleVeRLCallback:
    """
    veRL training loop callback for Martingale provenance tracking.

    Hooks:
      on_update_actor(model, global_step)  — call after each policy optimizer step
      on_rollout_start(model)              — call before each rollout batch

    Both hooks publish the current model weights as a revision and return the
    revision digest. Actors can use this digest to pin the current revision
    before sampling actions.
    """

    def __init__(self, publisher: RevisionPublisher) -> None:
        self.publisher = publisher
        self._current_revision: str | None = None
        self._global_step: int = 0

    @property
    def current_revision(self) -> str | None:
        """The digest of the most recently published revision, or None."""
        return self._current_revision

    @property
    def global_step(self) -> int:
        """The global_step from the most recent on_update_actor call."""
        return self._global_step

    def on_update_actor(self, model, global_step: int = 0) -> str:
        """
        Publish model weights as a new revision after a policy optimizer step.

        Call this immediately after `optimizer.step()` in the learner loop.

        Args:
            model: The actor model (nn.Module or state dict).
            global_step: The learner's global training step counter.

        Returns:
            The new revision digest (hex string).
        """
        self._global_step = global_step
        rev = self.publisher.publish(model, step=global_step)
        self._current_revision = rev.digest
        return rev.digest

    def on_rollout_start(self, model) -> str:
        """
        Publish current model weights and return the revision digest to pin.

        Call this before dispatching a rollout batch to actors so each actor
        can call `actor.pin_revision(digest)` before sampling.

        Returns:
            The revision digest actors should pin for this rollout.
        """
        rev = self.publisher.publish(model)
        self._current_revision = rev.digest
        return rev.digest
