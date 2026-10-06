"""benchmarks/modal/trl_grpo_vllm.py — the headline run: a real TRL GRPO job on vLLM
with the flight recorder on, induced staleness, then verify + report.

What it produces (under benchmarks/modal/results/<run>/):
  tokens.db             the record (copied back from the container)
  verify.json           checker/verify_tokens.py output, anchored on the published head
  report.json/.md       martingale report: staleness-vs-mismatch decomposition
  head.txt              the ledger head printed by the trainer (the out-of-band anchor)
  trainer_log.json      TRL's log history

Setup (mirrors Reservoir's working vLLM tier, 2026-10-05):
  model   Qwen/Qwen2.5-0.5B-Instruct, trainer in float32, server serves bf16
  GPU     A10G:2 (GPU 0 trains, GPU 1 serves); TRL 1.13.0, vLLM 0.28.0
  TRL     use_vllm=True, vllm_mode="server", vllm_importance_sampling_correction=False
          (the loss never reads sampling logprobs; the recorder still records them)
  lag     num_iterations=2 and steps_per_generation=2 make each generation batch
          train over several optimizer steps, so the record contains lag 0, 1, 2, 3
          tokens; lag 0 is the vLLM-vs-trainer mismatch floor.
  reward  exact-match on 2-digit addition prompts (synthetic, no dataset download)

Usage
  pip install modal && modal setup                            # once
  modal run benchmarks/modal/trl_grpo_vllm.py                 # default engine, temperature 1.0
  modal run benchmarks/modal/trl_grpo_vllm.py --temperature 0.7
  modal run benchmarks/modal/trl_grpo_vllm.py --batch-invariant   # deterministic vLLM kernels: is the floor smaller?
  modal run benchmarks/modal/trl_grpo_vllm.py --max-steps 24

Not run yet by the author (needs Modal credentials and a GPU budget); the
local test suite covers the integration against a fake trainer only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

VLLM_VERSION = os.environ.get("MARTINGALE_VLLM_VERSION", "0.28.0")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GPU = "A10G:2"
SERVER_HOST, SERVER_PORT, GROUP_PORT = "127.0.0.1", 8000, 51216
SERVER_START_TIMEOUT_S = 900.0
RESULTS_DIR = Path(__file__).parent / "results"
REPO_ROOT = Path(__file__).resolve().parents[2]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}")
    .pip_install("trl==1.13.0", "transformers==5.17.0", "datasets==5.0.1", "accelerate==1.15.0",
                 "numpy==2.2.6", "click>=8.0")
    .env({"HF_HOME": "/hf_cache", "PYTHONPATH": "/repo/src:/repo"})
    .add_local_dir(str(REPO_ROOT / "src"), "/repo/src")
    .add_local_dir(str(REPO_ROOT / "checker"), "/repo/checker")
)
app = modal.App("martingale-trl-grpo-vllm")
hf_cache = modal.Volume.from_name("martingale-hf-cache", create_if_missing=True)


def _addition_dataset(n: int, seed: int):
    import random

    from datasets import Dataset
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        rows.append({"prompt": [{"role": "user", "content": f"What is {a} + {b}? Answer with the number only."}],
                     "answer": str(a + b)})
    return Dataset.from_list(rows)


def exact_match_reward(completions, answer, **kwargs):
    out = []
    for comp, ans in zip(completions, answer):
        text = comp[0]["content"] if isinstance(comp, list) else str(comp)
        out.append(1.0 if text.strip().rstrip(".") == ans else 0.0)
    return out


def _start_vllm_server(log_path: Path, batch_invariant: bool) -> subprocess.Popen:
    """Launch vLLM the way `trl vllm-serve` does on TRL 1.13 (Reservoir's working recipe, 2026-10-05):
    plain `vllm serve` with the NCCL weight-transfer engine the trainer pushes weights through,
    processed log-probs (what TRL's client expects) and dev-mode endpoints."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1", "VLLM_SERVER_DEV_MODE": "1",
           "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "NCCL_DEBUG": "WARN"}
    if batch_invariant:
        env["VLLM_BATCH_INVARIANT"] = "1"      # inference-only deterministic kernels; server process only
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL_ID,
           "--host", SERVER_HOST, "--port", str(SERVER_PORT),
           "--gpu-memory-utilization", "0.5", "--max-model-len", "1024", "--dtype", "bfloat16",
           "--enforce-eager", "--weight-transfer-config", json.dumps({"backend": "nccl"}),
           "--logprobs-mode", "processed_logprobs", "--max-logprobs", "-1",
           "--uvicorn-log-level", "warning"]
    log = open(log_path, "wb")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + SERVER_START_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited early; see {log_path}")
        try:
            with urllib.request.urlopen(f"http://{SERVER_HOST}:{SERVER_PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return proc
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(5)
    raise TimeoutError("vLLM server did not become healthy")


@app.function(image=image, gpu=GPU, timeout=60 * 60, volumes={"/hf_cache": hf_cache})
def run_grpo(max_steps: int = 16, seed: int = 0, temperature: float = 1.0, batch_invariant: bool = False) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig

    from checker.verify_tokens import verify_export
    from martingale.diagnostics import decompose, render_markdown
    from martingale.integrations.trl import MartingaleRecorder, build_trainer_class

    work = Path("/tmp/martingale_run")
    work.mkdir(parents=True, exist_ok=True)
    server = _start_vllm_server(work / "vllm_server.log", batch_invariant)
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["NCCL_DEBUG"] = "WARN"
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        flight = MartingaleRecorder(str(work / "ws"), tokenizer=tok, sampler_overrides={
            "engine_version": f"vllm {VLLM_VERSION} processed_logprobs"
                              + (" batch_invariant" if batch_invariant else "")})
        args = GRPOConfig(
            output_dir=str(work / "out"), seed=seed, max_steps=max_steps, logging_steps=1,
            per_device_train_batch_size=8, gradient_accumulation_steps=1, num_generations=8,
            max_completion_length=16, max_prompt_length=64, temperature=temperature, learning_rate=5e-6,
            num_iterations=2, steps_per_generation=2, report_to=[], save_strategy="no", bf16=False,
            use_vllm=True, vllm_mode="server", vllm_server_base_url=f"http://{SERVER_HOST}:{SERVER_PORT}",
            vllm_group_port=GROUP_PORT, vllm_server_timeout=120.0, vllm_importance_sampling_correction=False,
        )
        Trainer = build_trainer_class()
        trainer = Trainer(model=model, args=args, train_dataset=_addition_dataset(512, seed),
                          reward_funcs=exact_match_reward, processing_class=tok, flight_recorder=flight)
        trainer.train()
        head = flight.head()
        print(f"MARTINGALE_LEDGER_HEAD {head}")
        export = flight.recorder.export_for_checker(work / "export")
        verify = verify_export(export, expected_head=head)
        report = decompose(flight.recorder.ledger)
        return {
            "head": head, "verify": verify, "report": report, "report_md": render_markdown(report),
            "stats": flight.stats, "log_history": trainer.state.log_history,
            "tokens_db": (work / "ws" / "tokens.db").read_bytes(),
            "config": {"model": MODEL_ID, "vllm": VLLM_VERSION, "trl": "1.13.0", "gpu": GPU,
                       "max_steps": max_steps, "seed": seed, "temperature": temperature,
                       "batch_invariant": batch_invariant, "num_iterations": 2, "steps_per_generation": 2},
        }
    finally:
        server.terminate()


@app.local_entrypoint()
def main(max_steps: int = 16, seed: int = 0, temperature: float = 1.0, batch_invariant: bool = False):
    res = run_grpo.remote(max_steps=max_steps, seed=seed, temperature=temperature, batch_invariant=batch_invariant)
    tag = f"_t{temperature}" + ("_bi" if batch_invariant else "")
    out = RESULTS_DIR / f"trl_grpo_vllm_a10g_{max_steps}steps_seed{seed}{tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokens.db").write_bytes(res.pop("tokens_db"))
    (out / "report.md").write_text(res.pop("report_md"))
    (out / "head.txt").write_text(res["head"] + "\n")
    (out / "verify.json").write_text(json.dumps(res["verify"], indent=1))
    (out / "report.json").write_text(json.dumps(res["report"], indent=1))
    (out / "trainer_log.json").write_text(json.dumps(res["log_history"], indent=1))
    (out / "config.json").write_text(json.dumps({**res["config"], "stats": res["stats"]}, indent=1))
    print((out / "report.md").read_text())
    print("verify ok:", res["verify"]["ok"], "| head:", res["head"])
    print("results in", out)
