"""benchmarks/modal/verl_grpo_vllm.py — the verl adapter's first real run: verl 0.9.1 GRPO on a
vLLM rollout with the flight recorder on, then verify + report.

What it produces (under benchmarks/modal/results/verl_grpo_vllm_t4_<n>steps_seed<seed>/):
  tokens.db             the record, after a WAL checkpoint (copied back from the container)
  verify.json           checker/verify_tokens.py output, anchored on the published head
  report.json/.md       martingale report: staleness-vs-mismatch decomposition
  head.txt              the ledger head printed by the driver (the out-of-band anchor)
  alarms.json           MartingaleMonitor alarms and per-step diagnoses
  trainer_log.json      every step verl's Tracking logged (martingale/* included)
  worst_tokens.json     the 40 largest |trainer - engine| tokens with decoded context
  adapter_checks.json   one verdict per adapter assumption (verl_hooks.adapter_checks)
  config.json           run config and recorder stats

Setup:
  model   Qwen/Qwen2.5-0.5B-Instruct
  GPU     one T4; verl 0.9.1 with its vllm extra (vLLM 0.24.0, torch 2.11.0, transformers 5.9.0)
  verl    RayPPOTrainer (trainer.use_v1=false), GRPO advantages, FSDP actor, vLLM rollout via the
          agent loop, rollout.calculate_log_probs=True. Image, GPU and verl config are Reservoir's
          verl_replay_real.py, which runs this stack on a T4. The T4 has no bf16: actor mixed
          precision and the rollout are fp16 (verl attaches a grad scaler), attention is SDPA.
          The trainer forward and vLLM both see fp16 casts of the fp32 master weights, so the
          lag-0 floor compares like with like (docs/findings/2026-10-06-trl-colocate-sync.md).
  lag     ppo_mini_batch_size = train_batch_size/2 and ppo_epochs=2: each rollout is trained over
          4 optimizer steps, so the record holds lag 0..3 scores; lag 0 is the vLLM-vs-FSDP floor.
  reward  exact match on 2-digit addition prompts (the TRL runner's task, same draws per seed)

Hooks (verl_hooks.py): the driver records each generate_sequences output with
MartingaleVerlRecorder.on_rollout before reward/balance/correction; the actor worker installs
wrap_ppo_loss on its loss and counts applied optimizer steps; MartingaleMonitor runs after each
actor update and its martingale/* metrics go into the actor metrics verl logs.

Usage
  pip install modal && modal setup                          # once
  modal run benchmarks/modal/verl_grpo_vllm.py              # 16 steps on one T4
  modal run benchmarks/modal/verl_grpo_vllm.py --max-steps 8
  modal run benchmarks/modal/verl_grpo_vllm.py --tag probe  # keep the result dir separate
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import modal

VERL_VERSION = "0.9.1"
VLLM_VERSION = "0.24.0"                   # pinned by verl[vllm]==0.9.1
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GPU = "T4"
RESULTS_DIR = Path(__file__).parent / "results"
WORK = Path("/tmp/martingale_verl")
# Modal re-imports this file as /root/verl_grpo_vllm.py inside the container, where it has
# no repo above it; the mounts below only matter locally, at image-definition time.
_here = Path(__file__).resolve()
REPO_ROOT = _here.parents[2] if len(_here.parents) > 2 else Path("/root")

# torch from the cu130 index first (Modal's T4 hosts run a CUDA 13 driver; the vLLM 0.24.0
# wheel is a cu130 build), then verl with its vllm extra, which pins vllm, torch, transformers
# and brings Ray. Ray actors do not inherit the driver's sys.path, so the sources go on PYTHONPATH,
# with /root, where Modal puts this file.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0",
                 index_url="https://download.pytorch.org/whl/cu130")
    .pip_install(f"verl[vllm]=={VERL_VERSION}", "click>=8.0")
    .env({"HF_HOME": "/hf_cache", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/repo/src:/repo:/root"})
    .add_local_dir(str(REPO_ROOT / "src"), "/repo/src")
    .add_local_dir(str(REPO_ROOT / "checker"), "/repo/checker")
    .add_local_file(str(_here.parent / "verl_hooks.py"), "/repo/verl_hooks.py")
)
app = modal.App("martingale-verl-grpo-vllm")
hf_cache = modal.Volume.from_name("martingale-hf-cache", create_if_missing=True)
# The container writes the result files here too, so a `modal run --detach` run keeps them with no client
# attached: modal volume get martingale-results <run name> benchmarks/modal/results/
results_vol = modal.Volume.from_name("martingale-results", create_if_missing=True)


def run_name(max_steps: int, seed: int, temperature: float, tag: str) -> str:
    suffix = (f"_t{temperature}" if temperature != 1.0 else "") + (f"_{tag}" if tag else "")
    return f"verl_grpo_vllm_{GPU.lower()}_{max_steps}steps_seed{seed}{suffix}"


def make_task_runner():
    """The Ray actor that builds verl's RayPPOTrainer as ``verl.trainer.main_ppo_v0.TaskRunner.run``
    does (verl 0.9.1), with the actor worker swapped for a subclass and the hooks attached. Everything
    the actor pickles is local here or in verl_hooks, so no Ray process has to import this Modal file."""
    import ray
    from verl.trainer.main_ppo_v0 import BaseTaskRunner

    def build_worker_class():
        """verl's ActorRolloutRefWorker plus four driver-callable RPCs that own the worker-side hooks."""
        from verl.single_controller.base.decorator import Dispatch, register
        from verl.workers.engine_workers import ActorRolloutRefWorker

        class MartingaleActorRolloutRefWorker(ActorRolloutRefWorker):
            @register(dispatch_mode=Dispatch.ONE_TO_ALL)
            def martingale_install(self, workspace: str, tokenizer_digest: str, sampler_overrides: dict):
                from verl_hooks import WorkerHooks
                self._martingale = WorkerHooks.install(self.actor, workspace=workspace, tokenizer_digest=tokenizer_digest,
                                                       sampler_overrides=sampler_overrides)

            @register(dispatch_mode=Dispatch.ONE_TO_ALL)
            def martingale_snapshot(self):
                return self._martingale.snapshot()

            @register(dispatch_mode=Dispatch.ONE_TO_ALL)
            def martingale_status(self):
                return self._martingale.status()

            @register(dispatch_mode=Dispatch.ONE_TO_ALL)
            def martingale_close(self):
                self._martingale.close()

        return MartingaleActorRolloutRefWorker

    @ray.remote
    class MartingaleTaskRunner(BaseTaskRunner):
        def run(self, config) -> dict:
            from omegaconf import OmegaConf
            from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
            from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic, need_reference_policy
            from verl.utils.config import omega_conf_to_dataclass, validate_config
            from verl.utils.dataset.rl_dataset import collate_fn
            from verl.utils.tracking import Tracking
            from verl.workers.config import HFModelConfig
            from verl_hooks import DriverHooks, FlightView, WorkerRPC, adapter_checks, capture_logs, worst_tokens

            from checker.verify_tokens import verify_export
            from martingale.diagnostics import AlarmConfig, decompose, render_markdown
            from martingale.integrations.trl import _tokenizer_bytes
            from martingale.integrations.verl import MartingaleMonitor, MartingaleVerlRecorder
            from martingale.record.revision import tokenizer_digest

            OmegaConf.resolve(config)
            _, ray_worker_group_cls = self.add_actor_rollout_worker(config)
            worker_cls = build_worker_class()
            for role in (Role.ActorRollout, Role.ActorRolloutRef):
                if role in self.role_worker_mapping:
                    self.role_worker_mapping[role] = ray.remote(worker_cls)
            self.add_critic_worker(config)
            self.add_reward_model_resource_pool(config)
            self.add_teacher_model_resource_pool(config)
            self.add_ref_policy_worker(config, worker_cls)
            validate_config(config=config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
            model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
            tokenizer, processor = model_config.tokenizer, model_config.processor
            resource_pool_manager = self.init_resource_pool_mgr(config)
            train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True,
                                              max_samples=config.data.get("train_max_samples", -1))
            val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False,
                                            max_samples=config.data.get("val_max_samples", -1))
            trainer = RayPPOTrainer(
                config=config, tokenizer=tokenizer, processor=processor, role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager, ray_worker_group_cls=ray_worker_group_cls,
                train_dataset=train_dataset, val_dataset=val_dataset, collate_fn=collate_fn,
                train_sampler=create_rl_sampler(config.data, train_dataset),
            )
            trainer.init_workers()

            workspace = str(WORK / "ws")
            tok_digest = tokenizer_digest(_tokenizer_bytes(tokenizer))
            overrides = {"engine_version": f"verl {VERL_VERSION} vllm {VLLM_VERSION} fp16"}
            flight = MartingaleVerlRecorder(workspace, tok_digest, sampler_overrides=overrides)
            trainer.actor_rollout_wg.martingale_install(workspace, tok_digest, overrides)
            view = FlightView(flight)
            monitor = MartingaleMonitor(view, AlarmConfig(), halt_on_error=False)  # record alarms, keep the data
            hooks = DriverHooks(flight, monitor, view, WorkerRPC(trainer.actor_rollout_wg))
            hooks.install(trainer)
            log_history = capture_logs(Tracking)
            started, fit_error = time.time(), None
            try:
                trainer.fit()
            except Exception:       # a refused rollout is a result: keep the record and the audit
                fit_error = traceback.format_exc()
                print(fit_error, flush=True)
            wall_clock = time.time() - started

            try:
                status = trainer.actor_rollout_wg.martingale_status()[0]
                trainer.actor_rollout_wg.martingale_close()      # worker's connection checkpoints and closes
            except Exception:       # the actor died with fit(): keep the driver's half of the record
                status = {"applied_steps": 0, "skipped_steps": 0, "loss": {}, "stats": {},
                          **hooks.last_status, "worker_lost": True}
            view.worker_stats = status["stats"]
            stats = {"driver": dict(flight.stats), "worker": status["stats"], "applied_steps": status["applied_steps"],
                     "skipped_steps": status["skipped_steps"], "loss_calls": status["loss"]}
            head = flight.head()
            print(f"MARTINGALE_LEDGER_HEAD {head}", flush=True)
            ledger = flight.recorder.ledger
            worst = worst_tokens(ledger, tokenizer, n=40)
            ledger.checkpoint_wal()                              # tokens.db alone must hold everything
            verify = verify_export(flight.recorder.export_for_checker(WORK / "export"), expected_head=head)
            report = decompose(ledger)
            checks = adapter_checks(rollouts=hooks.rollouts, updates=hooks.updates, worker=status,
                                    revisions=[r.to_dict() for r in ledger.all_revisions()], log_history=log_history)
            checks["fit_error"] = fit_error
            checks["all_ok"] = checks["all_ok"] and fit_error is None
            checks["rollouts"], checks["updates"] = hooks.rollouts, hooks.updates
            return {
                "head": head, "verify": verify, "report": report, "report_md": render_markdown(report),
                "stats": stats, "log_history": log_history, "checks": checks, "wall_clock_seconds": wall_clock,
                "alarms": [{"kind": a.kind, "level": a.level, "step": a.step, "message": a.message,
                            "evidence": a.evidence} for a in monitor.alarms],
                "per_step_metrics": [d.metrics() for d in monitor.diagnoses], "worst_tokens": worst,
                "tokens_db": (WORK / "ws" / "tokens.db").read_bytes(),
            }

    return MartingaleTaskRunner


@app.function(image=image, gpu=GPU, timeout=60 * 60, cpu=4, memory=32768, volumes={"/hf_cache": hf_cache, "/results": results_vol})
def run_grpo(max_steps: int = 16, seed: int = 0, temperature: float = 1.0, lr: float = 1e-6,
             tag: str = "") -> dict:
    import ray
    import torch
    import verl
    import verl.trainer
    import vllm
    from datasets import Dataset
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    from verl.trainer.main_ppo import get_ppo_ray_runtime_env

    sys.path.insert(0, "/repo")
    WORK.mkdir(parents=True, exist_ok=True)
    train_parquet = WORK / "train.parquet"
    from verl_hooks import addition_rows, overrides
    Dataset.from_list(addition_rows(512, seed)).to_parquet(str(train_parquet))
    shape = {"n": 8, "train_batch_size": 8, "ppo_mini_batch_size": 4, "ppo_epochs": 2,
             "max_prompt_length": 96, "max_response_length": 16}
    with initialize_config_dir(config_dir=str(Path(verl.trainer.__file__).parent / "config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides(
            train_parquet=str(train_parquet), reward_py="/repo/verl_hooks.py", model_id=MODEL_ID,
            max_steps=max_steps, seed=seed, lr=lr, temperature=temperature, **shape))
    with open_dict(config.data):
        config.data.seed = seed
    if not ray.is_initialized():
        runtime_env = OmegaConf.merge(get_ppo_ray_runtime_env(config),
                                      config.ray_kwargs.get("ray_init", {}).get("runtime_env", {}))
        ray.init(num_cpus=8, runtime_env=OmegaConf.to_container(runtime_env))
    res = ray.get(make_task_runner().remote().run.remote(config))
    ray.shutdown()
    res["config"] = {
        "model": MODEL_ID, "gpu": GPU, "verl": verl.__version__, "vllm": vllm.__version__, "torch": torch.__version__,
        "trainer": "verl.trainer.ppo.ray_trainer.RayPPOTrainer (use_v1=false)", "adv_estimator": "grpo",
        "max_steps": max_steps, "seed": seed, "temperature": temperature, "lr": lr, **shape,
        "actor_precision": "fp32 master, fp16 mixed precision", "rollout_dtype": "float16",
        "calculate_log_probs": True, "wall_clock_seconds": res.pop("wall_clock_seconds"),
    }
    from verl_hooks import write_results
    print("results in volume martingale-results:", write_results(res, Path("/results") / run_name(max_steps, seed, temperature, tag)))
    results_vol.commit()
    return res


@app.local_entrypoint()
def main(max_steps: int = 16, seed: int = 0, temperature: float = 1.0, lr: float = 1e-6, tag: str = ""):
    sys.path.insert(0, str(_here.parent))
    from verl_hooks import write_results

    res = run_grpo.remote(max_steps=max_steps, seed=seed, temperature=temperature, lr=lr, tag=tag)
    out = write_results(res, RESULTS_DIR / run_name(max_steps, seed, temperature, tag))
    print(res["report_md"])
    print(json.dumps({k: v["ok"] for k, v in res["checks"].items() if isinstance(v, dict) and "ok" in v}, indent=1))
    if res["checks"]["fit_error"]:
        print("fit() raised; the record up to that point was kept:\n" + res["checks"]["fit_error"])
    print("verify ok:", res["verify"]["ok"], "| adapter checks ok:", res["checks"]["all_ok"],
          "| head:", res["head"], "| alarms:", len(res["alarms"]))
    print("results in", out)
