"""
integrations/trl.py — the flight recorder for TRL's GRPOTrainer.

    from martingale.integrations.trl import MartingaleRecorder, MartingaleGRPOTrainer

    flight = MartingaleRecorder("./martingale_ws", tokenizer=tokenizer)
    trainer = MartingaleGRPOTrainer(model=..., args=GRPOConfig(...), train_dataset=...,
                                    reward_funcs=..., flight_recorder=flight)
    trainer.train()
    print(flight.head())            # publish this line; the checker anchors on it
    # martingale verify --dir ./martingale_ws ; martingale report --dir ./martingale_ws

What is recorded, and where it comes from
-----------------------------------------
Generation (``GRPOTrainer._generate_and_score_completions``): one SequenceRecord
per batch row. Behavior log-probs are, in order of preference, vLLM's
``sampling_per_token_logps`` (the engine that actually sampled), TRL's
``old_per_token_logps`` (a no-grad forward under the generating weights), or a
no-grad forward made here. Which one was used is written into the revision's
sampler config (``logprobs_mode``), because the three mean different things.
The revision is the digest of the trainer's weights at this ``global_step``
plus tokenizer and sampler config (design.md 7). ``reward_bits`` holds TRL's
advantage for the row, the value the loss consumes.

Update (``GRPOTrainer._get_per_token_logps_and_entropies`` on the grad path):
one ScoreRecord per token with the trainer's log-prob under the weights it is
training with, so lag = train step - behavior step is known per token and
``martingale report`` can separate staleness from engine mismatch.

Costs: one state-dict digest per global_step (same class as a checkpoint
save; refused for sharded/meta state dicts) and ~250 bytes per token. Rows
are matched between generation and update by a per-row id tensor
(``martingale_row_id``) that the recorder adds to the generation output;
TRL's shuffle and split helpers carry it to ``_compute_loss``. When the id is
absent (another trainer) rows fall back to content matching with a FIFO per
(prompt, completion), so duplicates are scored once each. Rows whose
behavior log-probs contain NaN (vLLM reports None for some tokens) are
skipped and counted in ``stats["nan_rows"]``. Multi-process: give each rank
its own workspace; the ledger serialises writers on one file but the
actor-id chains are per rank. Nothing on this path claims the draw itself
is verifiable.

Tested against TRL 1.13.0 (``_trl_compat``). ``MartingaleRecorder`` imports
nothing from TRL; ``MartingaleGRPOTrainer`` is built on first access.
"""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from martingale.record import Recorder, SamplerConfig, weights_digest
from martingale.record.revision import LLMRevision, tokenizer_digest

if TYPE_CHECKING:
    import torch

SAMPLING_LOGPROBS_KEY = "sampling_per_token_logps"
OLD_LOGPROBS_KEY = "old_per_token_logps"
ROW_ID_KEY = "martingale_row_id"
KEEP_GENERATIONS = 2          # row maps kept for the last N generation batches


def _tokenizer_bytes(tokenizer: Any) -> bytes | str:
    """tokenizer.json bytes from a HF tokenizer, raw bytes, or a precomputed digest."""
    if isinstance(tokenizer, (bytes, str)):
        return tokenizer
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        return backend.to_str().encode("utf-8")
    vocab = getattr(tokenizer, "get_vocab", None)
    if vocab is not None:
        import json
        return json.dumps(vocab(), sort_keys=True).encode("utf-8")
    raise TypeError("tokenizer must be a HF tokenizer, tokenizer.json bytes, or a 64-hex digest")


def _engine_version(use_vllm: bool) -> str | None:
    try:
        if use_vllm:
            import vllm
            return str(vllm.__version__)
        import transformers
        return str(transformers.__version__)
    except Exception:
        return None


def sampler_from_trainer(trainer: Any, logprobs_mode: str) -> SamplerConfig:
    a = trainer.args
    use_vllm = bool(getattr(a, "use_vllm", False))
    model = getattr(trainer, "model", None)
    dtype = None
    try:
        dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    except Exception:
        pass
    return SamplerConfig(
        temperature=getattr(a, "temperature", None), top_p=getattr(a, "top_p", None),
        top_k=getattr(a, "top_k", None), min_p=getattr(a, "min_p", None),
        repetition_penalty=getattr(a, "repetition_penalty", None),
        max_tokens=getattr(a, "max_completion_length", None), logprobs_mode=logprobs_mode,
        dtype=dtype, engine=("vllm:" + str(getattr(a, "vllm_mode", "?"))) if use_vllm else "transformers",
        engine_version=_engine_version(use_vllm),
        tensor_parallel=getattr(a, "vllm_tensor_parallel_size", None) if use_vllm else None,
        seed_policy=f"seed={getattr(a, 'seed', None)}",
    )


def _prefix_lengths(mask: torch.Tensor, side: str) -> list[int]:
    """Lengths of a right-padded ("prefix": 1..1 0..0) or left-padded ("suffix": 0..0 1..1) mask.
    Refuses a mask that is not pure padding, because slicing by sum() would misalign tokens silently."""
    import torch

    m = mask.to("cpu").long()
    lengths = m.sum(dim=1)
    n = m.size(1)
    idx = torch.arange(n).unsqueeze(0)
    expected = (idx < lengths.unsqueeze(1)) if side == "prefix" else (idx >= (n - lengths).unsqueeze(1))
    if not bool((m.bool() == expected).all()):
        raise ValueError(f"{side} mask is not pure padding; cannot slice tokens safely")
    return [int(x) for x in lengths.tolist()]


def _check_digestable(state_dict: dict) -> None:
    for name, t in state_dict.items():
        if getattr(t, "is_meta", False) or (hasattr(t, "numel") and t.numel() == 0):
            raise RuntimeError(
                f"parameter {name!r} is sharded or on the meta device; the weights digest would be meaningless. "
                "Gather the full state dict (FSDP/ZeRO-3) before recording.")


class MartingaleRecorder:
    """Recorder side of the integration. Imports nothing from TRL."""

    def __init__(self, workspace: str, tokenizer: Any, *, logprob_dtype: str = "f32",
                 record_scores: bool = True, sampler_overrides: dict | None = None) -> None:
        self.recorder = Recorder(workspace)
        self._tok = _tokenizer_bytes(tokenizer)
        self._tok_digest = self._tok if isinstance(self._tok, str) else tokenizer_digest(self._tok)
        self._dtype = logprob_dtype
        self._record_scores = record_scores
        self._overrides = dict(sampler_overrides or {})
        self._rev_cache: dict[tuple[int, str], LLMRevision] = {}
        self._wd_cache: dict[int, str] = {}
        self._next_row_id = 0
        self._rows_by_id: dict[int, str] = {}
        self._generations: list[tuple[dict[int, str], dict[tuple[tuple[int, ...], tuple[int, ...]], list[str]]]] = []
        self.stats: dict[str, int] = {"generation_calls": 0, "sequences": 0, "tokens": 0,
                                      "score_calls": 0, "scored_tokens": 0, "unmatched_rows": 0,
                                      "nan_rows": 0, "revisions": 0}

    # ---- revisions ----------------------------------------------------------------

    def revision_for(self, trainer: Any, logprobs_mode: str) -> LLMRevision:
        """One weights digest per (global_step, logprobs_mode); cached."""
        step = int(trainer.state.global_step)
        key = (step, logprobs_mode)
        if key in self._rev_cache:
            return self._rev_cache[key]
        sampler = sampler_from_trainer(trainer, logprobs_mode)
        if self._overrides:
            sampler = SamplerConfig.from_dict({**sampler.to_dict(), **self._overrides})
        if step not in self._wd_cache:               # one full-weights digest per global step
            model = trainer.accelerator.unwrap_model(trainer.model) if hasattr(trainer, "accelerator") else trainer.model
            sd = model.state_dict()
            _check_digestable(sd)
            self._wd_cache[step] = weights_digest(sd)
            if len(self._wd_cache) > 8:
                self._wd_cache.pop(min(self._wd_cache))
        rev = self.recorder.publish_revision(self._wd_cache[step], self._tok_digest, sampler, step=step)
        self._rev_cache[key] = rev
        self.stats["revisions"] += 1
        return rev

    # ---- generation ---------------------------------------------------------------

    def _behavior_logprobs(self, output: dict, trainer: Any) -> tuple[torch.Tensor, str]:
        if SAMPLING_LOGPROBS_KEY in output and output[SAMPLING_LOGPROBS_KEY] is not None:
            return output[SAMPLING_LOGPROBS_KEY], "engine_sampling"
        if OLD_LOGPROBS_KEY in output and output[OLD_LOGPROBS_KEY] is not None:
            return output[OLD_LOGPROBS_KEY], "trainer_old_logps"
        import torch
        input_ids = torch.cat([output["prompt_ids"], output["completion_ids"]], dim=1)
        attention_mask = torch.cat([output["prompt_mask"], output["completion_mask"]], dim=1)
        with torch.no_grad():
            lp, *_ = trainer._get_per_token_logps_and_entropies(
                trainer.model, input_ids, attention_mask, output["completion_ids"].size(1),
                batch_size=getattr(trainer.args, "per_device_train_batch_size", None),
            )
        return lp, "trainer_recompute"

    def on_generation(self, output: dict, trainer: Any) -> dict:
        """Record every row of a generation batch. Returns a copy of `output` with a
        ``martingale_row_id`` tensor added (TRL shuffles/splits it with the batch)."""
        import math

        import torch

        self.stats["generation_calls"] += 1
        logps, mode = self._behavior_logprobs(output, trainer)
        rev = self.revision_for(trainer, mode)
        actor = int(getattr(getattr(trainer, "accelerator", None), "process_index", 0))
        step = int(trainer.state.global_step)
        p_ids, p_mask = output["prompt_ids"].to("cpu"), output["prompt_mask"].to("cpu")
        c_ids, c_mask = output["completion_ids"].to("cpu"), output["completion_mask"].to("cpu")
        lp = logps.detach().to("cpu").tolist()
        adv = output.get("advantages")
        adv = None if adv is None else adv.detach().to("cpu").tolist()
        plen, clen = _prefix_lengths(p_mask, "suffix"), _prefix_lengths(c_mask, "prefix")
        width = p_ids.size(1)
        n = c_ids.size(0)
        row_ids = list(range(self._next_row_id, self._next_row_id + n))
        self._next_row_id += n
        by_id: dict[int, str] = {}
        by_content: dict[tuple[tuple[int, ...], tuple[int, ...]], list[str]] = {}
        for r in range(n):
            prompt = [int(x) for x in p_ids[r, width - plen[r]:].tolist()]           # left-padded
            comp = [int(x) for x in c_ids[r, :clen[r]].tolist()]                       # right-padded
            if not comp:
                continue
            row_lp = lp[r][:clen[r]]
            if any(math.isnan(x) or math.isinf(x) for x in row_lp):
                self.stats["nan_rows"] += 1
                continue
            with self.recorder.sequence(actor, f"step{step}/row{row_ids[r]}", prompt) as seq:
                for pos, tok in enumerate(comp):
                    seq.token(rev.digest, tok, row_lp[pos], dtype=self._dtype)
                if adv is not None:
                    seq.reward(float(adv[r]), dtype=self._dtype)
            digest = seq.record.digest
            by_id[row_ids[r]] = digest
            by_content.setdefault((tuple(prompt), tuple(comp)), []).append(digest)
            self.stats["sequences"] += 1
            self.stats["tokens"] += len(comp)
        self._generations.append((by_id, by_content))
        del self._generations[:-KEEP_GENERATIONS]
        out = dict(output)
        out[ROW_ID_KEY] = torch.tensor(row_ids, dtype=torch.int64, device=output["completion_ids"].device)
        return out

    # ---- scores -------------------------------------------------------------------

    def _lookup(self, row_id: int | None, key: tuple) -> str | None:
        """Row id first (exact, reusable: TRL scores the same row once per iteration and per
        optimizer step, each under a different revision); else content, round-robin over
        duplicates so identical rows in one batch map to distinct sequences."""
        for by_id, by_content in reversed(self._generations):
            if row_id is not None:
                if row_id in by_id:
                    return by_id[row_id]
                continue
            lst = by_content.get(key)
            if lst:
                digest = lst.pop(0)
                lst.append(digest)
                return digest
        return None

    def on_scores(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, logits_to_keep: int,
                  per_token_logps: torch.Tensor, trainer: Any, row_ids: torch.Tensor | None = None) -> int:
        """Record the trainer's log-probs for a loss micro-batch. Returns rows matched."""
        if not self._record_scores:
            return 0
        self.stats["score_calls"] += 1
        rev = None
        ids, am = input_ids.to("cpu"), attention_mask.to("cpu")
        lp = per_token_logps.detach().to("cpu").tolist()
        p_ids, c_ids = ids[:, :-logits_to_keep], ids[:, -logits_to_keep:]
        p_mask, c_mask = am[:, :-logits_to_keep], am[:, -logits_to_keep:]
        plen, clen = _prefix_lengths(p_mask, "suffix"), _prefix_lengths(c_mask, "prefix")
        rid = None if row_ids is None else [int(x) for x in row_ids.to("cpu").tolist()]
        width = p_ids.size(1)
        matched = 0
        for r in range(c_ids.size(0)):
            key = (tuple(int(x) for x in p_ids[r, width - plen[r]:].tolist()),
                   tuple(int(x) for x in c_ids[r, :clen[r]].tolist()))
            digest = self._lookup(None if rid is None else rid[r], key)
            if digest is None:
                self.stats["unmatched_rows"] += 1
                continue
            if rev is None:
                rev = self.revision_for(trainer, "trainer_scores")
            written = self.recorder.score(digest, rev.digest, lp[r][:clen[r]], dtype=self._dtype)
            if not written:                       # same row, same weights: gradient accumulation
                self.stats["duplicate_scores"] = self.stats.get("duplicate_scores", 0) + 1
                continue
            self.stats["scored_tokens"] += clen[r]
            matched += 1
        return matched

    def head(self) -> str:
        if self.stats["unmatched_rows"] or self.stats["nan_rows"]:
            import warnings
            warnings.warn(
                f"martingale: {self.stats['unmatched_rows']} loss rows were not matched to a recorded sequence and "
                f"{self.stats['nan_rows']} generation rows were skipped for non-finite log-probs; the report's "
                "unscored count reflects this", RuntimeWarning, stacklevel=2)
        return self.recorder.ledger.head()


class MartingaleGRPOMixin:
    """Place before GRPOTrainer in the base list; the class must set ``self.flight_recorder``."""

    flight_recorder: MartingaleRecorder

    def _generate_and_score_completions(self, inputs):  # type: ignore[override]
        output = super()._generate_and_score_completions(inputs)  # type: ignore[misc]
        if not self.model.training:  # type: ignore[attr-defined]
            return output
        return self.flight_recorder.on_generation(output, self)

    def _compute_loss(self, model, inputs, *args, **kwargs):  # type: ignore[override]
        self._martingale_row_ids = inputs.get(ROW_ID_KEY) if isinstance(inputs, dict) else None
        try:
            return super()._compute_loss(model, inputs, *args, **kwargs)  # type: ignore[misc]
        finally:
            self._martingale_row_ids = None

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, *args, **kwargs):  # type: ignore[override]
        result = super()._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep, *args, **kwargs)  # type: ignore[misc]
        import torch
        is_ref = getattr(self, "ref_model", None) is not None and model is self.ref_model
        if torch.is_grad_enabled() and getattr(self.model, "training", False) and not is_ref:  # the loss path
            logps = result[0] if isinstance(result, (tuple, list)) else result
            self.flight_recorder.on_scores(input_ids, attention_mask, logits_to_keep, logps, self,
                                           row_ids=getattr(self, "_martingale_row_ids", None))
        return result


@functools.cache
def build_trainer_class() -> type:
    from martingale.integrations._trl_compat import require_trl
    support = require_trl()

    from martingale.integrations._trl_callback import MartingaleMonitor, build_callback_class
    callback_cls = build_callback_class(support.trainer_callback)

    class MartingaleGRPOTrainer(MartingaleGRPOMixin, support.grpo_trainer):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, flight_recorder: MartingaleRecorder, alarm_config: Any = None,
                     halt_on_error: bool = True, **kwargs: Any) -> None:
            if not isinstance(flight_recorder, MartingaleRecorder):
                raise TypeError("flight_recorder must be a MartingaleRecorder")
            super().__init__(*args, **kwargs)
            self.flight_recorder = flight_recorder
            self.martingale_monitor = MartingaleMonitor(flight_recorder, alarm_config, halt_on_error=halt_on_error)
            cb = callback_cls(self.martingale_monitor)
            cb.trainer = self
            self.add_callback(cb)

    MartingaleGRPOTrainer.__module__ = __name__
    return MartingaleGRPOTrainer


def __getattr__(name: str) -> Any:
    if name == "MartingaleGRPOTrainer":
        return build_trainer_class()
    raise AttributeError(name)


__all__ = ["MartingaleRecorder", "MartingaleGRPOMixin", "build_trainer_class", "sampler_from_trainer"]
# MartingaleGRPOTrainer is served lazily by __getattr__ (it needs TRL).
