"""benchmarks/modal/verl_hooks.py — the framework-free half of verl_grpo_vllm.py.

Everything here runs on CPU against duck-typed objects (tests/test_verl_grpo_runner.py
drives it with a fake trainer and a fake worker); verl_grpo_vllm.py only subclasses
verl's worker and hands the real objects in. Top-level imports are stdlib only:
verl loads this file by path as the custom reward module (``compute_score``).

Two processes write one record. The TaskRunner (driver) records each rollout with
``MartingaleVerlRecorder.on_rollout`` on the DataProto that ``generate_sequences``
returns, before reward, balancing or any correction touches it. The actor worker
records a score on every loss call through ``wrap_ppo_loss``. Both open the same
SQLite workspace; the ledger serialises the two writers.

Steps are applied optimizer updates, counted on the worker: verl's fp16 path
(a T4 has no bf16) uses a grad scaler that skips the update when a gradient is
not finite, and a skipped update is not a new weight version. The driver reads
that counter together with the weights it digests, so a rollout's revision and
the first loss call after it hash the same tensors under the same step.

The worker gathers weights right after each applied step, never inside the loss:
a state_dict() between forward and backward runs FSDP2's pre-save hook, which
reshards the root group while autograd still holds its unsharded parameters.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

DATA_SOURCE = "martingale/addition"
ROLLOUT_KEYS = ("prompts", "responses", "attention_mask", "response_mask", "rollout_log_probs")
SCORE_KEYS = ("score_calls", "scored_tokens", "scored_sequences", "unmatched_rows", "duplicate_scores")
METRIC_PREFIX = "martingale/"


# ---- task: the TRL runner's 2-digit addition, in verl's RL dataset schema -------------------

def addition_rows(n: int, seed: int) -> list[dict]:
    """The prompts of trl_grpo_vllm._addition_dataset (same draws for a seed), as verl parquet rows."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        rows.append({
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": f"What is {a} + {b}? Answer with the number only."}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(a + b)},
            "extra_info": {"index": i, "split": "train"},
        })
    return rows


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """trl_grpo_vllm.exact_match_reward for one response (verl decodes with skip_special_tokens)."""
    return 1.0 if str(solution_str).strip().rstrip(".") == str(ground_truth) else 0.0


def overrides(*, train_parquet: str, reward_py: str, model_id: str, max_steps: int, seed: int, n: int,
              train_batch_size: int, ppo_mini_batch_size: int, ppo_epochs: int, lr: float,
              max_prompt_length: int, max_response_length: int, temperature: float) -> list[str]:
    """Hydra overrides: Reservoir's one-T4 verl 0.9.1 GRPO recipe (verl_replay_real.py), plus
    calculate_log_probs, the addition task, and PPO mini-batches x epochs so one rollout is
    trained over several optimizer steps (lags 0..ppo_epochs*mini_batches-1, as in the TRL run)."""
    return [
        "trainer.use_v1=false",                       # the DataProto RayPPOTrainer, not the TransferQueue loop
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files={train_parquet}",
        f"data.val_files={train_parquet}",
        f"data.train_batch_size={train_batch_size}",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={max_response_length}",
        "data.filter_overlong_prompts=True",
        "data.dataloader_num_workers=0",
        f"actor_rollout_ref.model.path={model_id}",
        "actor_rollout_ref.model.use_remove_padding=False",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        f"actor_rollout_ref.actor.optim.lr={lr}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_epochs={ppo_epochs}",
        "actor_rollout_ref.actor.shuffle=True",       # mini-batch order != rollout order: row ids must carry it
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.dtype=float16",
        "+actor_rollout_ref.actor.fsdp_config.mixed_precision={param_dtype:fp16,reduce_dtype:fp32,buffer_dtype:fp32}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={n}",
        f"actor_rollout_ref.rollout.temperature={temperature}",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "actor_rollout_ref.rollout.dtype=float16",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.3",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8",
        f"actor_rollout_ref.rollout.seed={seed}",
        f"reward.custom_reward_function.path={reward_py}",
        "reward.custom_reward_function.name=compute_score",
        "trainer.logger=[console]",
        "trainer.project_name=martingale",
        "trainer.experiment_name=verl_grpo_vllm",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        "trainer.save_freq=-1",
        f"trainer.total_training_steps={max_steps}",
        "trainer.total_epochs=1000",
        "trainer.balance_batch=True",
        "trainer.resume_mode=disable",
        "ray_kwargs.ray_init.num_cpus=8",
    ]


# ---- small pieces ---------------------------------------------------------------------------

class LazyWeights:
    """Not a Mapping, so MartingaleVerlRecorder calls state_dict() only when a step is not yet digested."""

    def __init__(self, gather: Callable[[], Mapping[str, Any]]) -> None:
        self._gather = gather

    def state_dict(self) -> Mapping[str, Any]:
        return self._gather()


def full_state(engine: Any) -> dict[str, Any]:
    """The tensors verl syncs into the rollout (``engine.get_per_tensor_param``), gathered to CPU.
    For FSDP2 these are ``DTensor.full_tensor()`` values: the unsharded fp32 master weights."""
    params, _peft = engine.get_per_tensor_param()
    return {name: t.detach().to("cpu").clone() for name, t in params}


def mask_holes(mask: Any) -> int:
    """Rows of a right-padded response mask that are not 1..1 0..0 (multi-turn tool text)."""
    m = mask.to("cpu").long()
    lengths = m.sum(dim=1, keepdim=True)
    import torch
    expected = (torch.arange(m.shape[1]).unsqueeze(0) < lengths).long()
    return int((m != expected).any(dim=1).sum())


def _plain(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


# ---- worker side ----------------------------------------------------------------------------

class WorkerHooks:
    """Installed on verl's actor ``TrainingWorker`` (``ActorRolloutRefWorker.actor``) in the worker process."""

    def __init__(self, flight: Any, engine: Any) -> None:
        self.flight = flight
        self.engine = engine
        self.applied_steps = 0
        self.skipped_steps = 0
        self.loss_time_gathers = 0
        self._pending: dict[str, Any] | None = None   # this step's weights, until the recorder digests them
        self.loss = {"calls": 0, "rows": 0, "calls_with_row_ids": 0, "live_calls": 0, "detached_calls": 0}

    @classmethod
    def install(cls, training_worker: Any, *, workspace: str, tokenizer_digest: str,
                sampler_overrides: dict | None = None) -> WorkerHooks:
        from martingale.integrations.verl import MartingaleVerlRecorder, wrap_ppo_loss

        flight = MartingaleVerlRecorder(workspace, tokenizer_digest, sampler_overrides=sampler_overrides)
        hooks = cls(flight, training_worker.engine)
        hooks._pending = full_state(hooks.engine)
        base = training_worker.loss_fn
        if not isinstance(base, partial):
            raise TypeError(f"expected verl's partial(ppo_loss, config=...), got {base!r}")
        recorded = wrap_ppo_loss(base.func, flight, step=lambda: hooks.applied_steps,
                                 weights=lambda: LazyWeights(hooks._take))
        training_worker.loss_fn = partial(hooks._inspected(recorded), *base.args, **base.keywords)
        step = hooks.engine.optimizer_step
        hooks.engine.optimizer_step = lambda: hooks._count(step())
        return hooks

    def _inspected(self, loss_fn: Callable) -> Callable:
        def inspected(config, model_output, data, dp_group=None):
            from martingale.integrations.verl import ROW_ID_KEY

            self.loss["calls"] += 1
            self.loss["rows"] += int(data["responses"].shape[0])
            self.loss["calls_with_row_ids"] += int(ROW_ID_KEY in data.keys())
            live = bool(getattr(model_output["log_probs"], "requires_grad", False))
            self.loss["live_calls" if live else "detached_calls"] += 1
            return loss_fn(config, model_output, data, dp_group=dp_group)
        return inspected

    def _count(self, grad_norm: Any) -> Any:
        # verl skips the update when the norm is not finite (scaler.step or its own check)
        if math.isfinite(float(grad_norm)):
            self.applied_steps += 1
            self._pending = full_state(self.engine)
        else:
            self.skipped_steps += 1
        return grad_norm

    def _take(self) -> dict[str, Any]:
        state, self._pending = self._pending, None
        if state is None:                    # not expected: the step was already digested once
            self.loss_time_gathers += 1
            state = full_state(self.engine)
        return state

    def snapshot(self) -> tuple[int, dict[str, Any]]:
        return self.applied_steps, full_state(self.engine)

    def status(self) -> dict:
        return {"applied_steps": self.applied_steps, "skipped_steps": self.skipped_steps,
                "loss_time_gathers": self.loss_time_gathers, "loss": dict(self.loss), "stats": dict(self.flight.stats)}

    def close(self) -> None:
        self.flight.recorder.ledger.close()


# ---- driver side ----------------------------------------------------------------------------

class FlightView:
    """What MartingaleMonitor reads: the driver's recorder, with rollout stats from the driver and
    score stats (unmatched rows, scored sequences) from the worker, which is where scoring happens."""

    def __init__(self, flight: Any) -> None:
        self.flight = flight
        self.recorder = flight.recorder
        self.worker_stats: dict[str, int] = {}

    @property
    def stats(self) -> dict[str, int]:
        out = dict(self.flight.stats)
        for key, value in self.worker_stats.items():
            out[key] = value if key in SCORE_KEYS else out.get(key, 0) + value
        return out


class DriverHooks:
    """Instance-patches a RayPPOTrainer after init_workers.

    ``worker`` exposes snapshot() -> (applied_steps, full weights) and status() -> dict, the
    two RPCs of the worker subclass in verl_grpo_vllm.py.
    """

    def __init__(self, flight: Any, monitor: Any, view: FlightView, worker: Any) -> None:
        self.flight = flight
        self.monitor = monitor
        self.view = view
        self.worker = worker
        self.rollouts: list[dict] = []
        self.updates: list[dict] = []
        self.last_status: dict = {}

    def install(self, trainer: Any) -> None:
        manager = trainer.async_rollout_manager
        manager.generate_sequences = self.wrap_generate(manager.generate_sequences)
        trainer._update_actor = self.wrap_update(trainer._update_actor, trainer)

    def wrap_generate(self, generate: Callable) -> Callable:
        def generate_sequences(*args, **kwargs):
            return self.on_rollout(generate(*args, **kwargs))
        return generate_sequences

    def on_rollout(self, output: Any) -> Any:
        from martingale.integrations.verl import ROW_ID_KEY

        data = output.batch
        keys = sorted(data.keys())
        missing = [k for k in ROLLOUT_KEYS if k not in keys]
        holes = mask_holes(data["response_mask"]) if "response_mask" in keys else None
        step, weights = self.worker.snapshot()
        entry = {"step": step, "rows": int(data["responses"].shape[0]) if "responses" in keys else None,
                 "response_width": int(data["responses"].shape[1]) if "responses" in keys else None,
                 "keys": keys, "missing": missing, "mask_hole_rows": holes}
        self.rollouts.append(entry)
        self.flight.on_rollout(output, step=step, weights=weights)   # raises on missing keys or holes
        entry["row_ids"] = [int(x) for x in data[ROW_ID_KEY].tolist()]
        return output

    def wrap_update(self, update: Callable, trainer: Any) -> Callable:
        def _update_actor(batch, *args, **kwargs):
            from martingale.integrations.verl import ROW_ID_KEY

            ids = batch.batch[ROW_ID_KEY].tolist() if ROW_ID_KEY in batch.batch.keys() else None
            out = update(batch, *args, **kwargs)
            status = self.worker.status()
            self.last_status = status
            self.view.worker_stats = status["stats"]
            sink = out.meta_info.setdefault("metrics", {}).update
            diag = self.monitor.on_step_end(status["applied_steps"], sink)
            recorded = self.rollouts[-1].get("row_ids") if self.rollouts else None
            self.updates.append({
                "global_step": int(getattr(trainer, "global_steps", len(self.updates) + 1)),
                "applied_steps": status["applied_steps"], "skipped_steps": status["skipped_steps"],
                "row_ids_present": ids is not None,
                "row_ids_match_rollout": ids is not None and recorded is not None and sorted(ids) == sorted(recorded),
                "row_order_changed": ids is not None and recorded is not None and ids != recorded,
                "new_scores": diag.n_new_scores,
            })
            return out
        return _update_actor


def capture_logs(tracking_cls: type) -> list[dict]:
    """Patch verl's Tracking.log (class-wide, in this process) to keep every logged step."""
    records: list[dict] = []
    original = tracking_cls.log

    def log(self, data, step, backend=None):
        records.append({"step": int(step), **{k: _plain(v) for k, v in data.items()}})
        return original(self, data, step, backend)

    tracking_cls.log = log
    return records


# ---- what the run establishes about the adapter ------------------------------------------------

def adapter_checks(*, rollouts: list[dict], updates: list[dict], worker: dict, revisions: list[dict],
                   log_history: list[dict]) -> dict:
    """One verdict per adapter assumption, from the hooks' audit and the record."""
    by_step: dict[int, dict[str, str]] = {}
    for rev in revisions:
        by_step.setdefault(int(rev["step"]), {})[rev["sampler"]["logprobs_mode"]] = rev["weights_digest"]
    paired = {s: d for s, d in by_step.items() if {"engine_sampling", "trainer_scores"} <= d.keys()}
    loss = worker.get("loss", {})
    stats = worker.get("stats", {})
    logged = [r for r in log_history if any(k.startswith(METRIC_PREFIX) for k in r)]
    checks = {
        "batch_keys": {"ok": bool(rollouts) and not any(r["missing"] for r in rollouts),
                       "required": list(ROLLOUT_KEYS), "missing_by_rollout": [r["missing"] for r in rollouts]},
        "mask_holes": {"ok": bool(rollouts) and all(r["mask_hole_rows"] == 0 for r in rollouts),
                       "rows_with_holes": sum(r["mask_hole_rows"] or 0 for r in rollouts)},
        "row_ids_survive_union_reorder": {
            "ok": bool(updates) and all(u["row_ids_match_rollout"] for u in updates),
            "updates_with_reordered_rows": sum(u["row_order_changed"] for u in updates),
            "loss_calls_with_row_ids": loss.get("calls_with_row_ids", 0), "loss_calls": loss.get("calls", 0),
            "unmatched_rows": stats.get("unmatched_rows", 0)},
        "loss_wrapper_sees_live_logps": {
            "ok": loss.get("calls", 0) > 0 and loss.get("detached_calls", 0) == 0
                  and stats.get("scored_tokens", 0) > 0,
            "live_calls": loss.get("live_calls", 0), "detached_calls": loss.get("detached_calls", 0),
            "scored_tokens": stats.get("scored_tokens", 0), "nan_rows": stats.get("nan_rows", 0)},
        "rollout_digest_equals_trainer_digest": {
            "ok": bool(paired) and all(d["engine_sampling"] == d["trainer_scores"] for d in paired.values()),
            "steps_compared": sorted(paired),
            "mismatched_steps": sorted(s for s, d in paired.items() if d["engine_sampling"] != d["trainer_scores"])},
        "metrics_in_verl_log": {"ok": bool(logged), "logged_steps_with_martingale_metrics": len(logged),
                                "logged_steps": len(log_history)},
        "optimizer_steps": {"applied": worker.get("applied_steps", 0), "skipped": worker.get("skipped_steps", 0),
                            "loss_time_gathers": worker.get("loss_time_gathers", 0)},
    }
    checks["all_ok"] = all(v["ok"] for v in checks.values() if isinstance(v, dict) and "ok" in v)
    return checks


def worst_tokens(ledger: Any, tokenizer: Any, n: int = 40) -> list[dict]:
    """As trl_grpo_vllm._worst_tokens: the n scored tokens with the largest |trainer - behavior|."""
    from martingale.record import bits_to_fraction
    revs = {r.digest: r for r in ledger.all_revisions()}
    rows = []
    for seq in ledger.all_sequences():
        comp = [t.token_id for t in seq.tokens]
        for sc in ledger.scores_for(seq.digest):
            t = seq.tokens[sc.position]
            b = float(bits_to_fraction(t.logprob_bits))
            tr = float(bits_to_fraction(sc.logprob_bits))
            rows.append({
                "sequence_id": seq.sequence_id, "position": sc.position, "token_id": t.token_id,
                "token": tokenizer.decode([t.token_id]), "completion": tokenizer.decode(comp),
                "lag": revs[sc.train_revision_digest].step - revs[t.revision_digest].step,
                "behavior_logprob": b, "trainer_logprob": tr, "log_ratio": tr - b,
                "completion_length": len(comp),
            })
    rows.sort(key=lambda r: -abs(r["log_ratio"]))
    return rows[:n]


def write_results(res: dict, out: Path) -> Path:
    """The TRL runner's files, plus adapter_checks.json."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokens.db").write_bytes(res["tokens_db"])
    (out / "report.md").write_text(res["report_md"])
    (out / "head.txt").write_text(res["head"] + "\n")
    (out / "verify.json").write_text(json.dumps(res["verify"], indent=1))
    (out / "report.json").write_text(json.dumps(res["report"], indent=1))
    (out / "trainer_log.json").write_text(json.dumps(res["log_history"], indent=1))
    (out / "alarms.json").write_text(json.dumps({"alarms": res["alarms"], "per_step_metrics": res["per_step_metrics"]},
                                                indent=1))
    (out / "worst_tokens.json").write_text(json.dumps(res["worst_tokens"], indent=1))
    (out / "adapter_checks.json").write_text(json.dumps(res["checks"], indent=1))
    (out / "config.json").write_text(json.dumps({**res["config"], "stats": res["stats"]}, indent=1))
    return out
