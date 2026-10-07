"""benchmarks/modal/trl_grpo_vllm.py — the headline run: a real TRL GRPO job on vLLM
with the flight recorder on, induced staleness, then verify + report.

What it produces (under benchmarks/modal/results/<run>/):
  tokens.db             the record (copied back from the container)
  verify.json           checker/verify_tokens.py output, anchored on the published head
  report.json/.md       martingale report: staleness-vs-mismatch decomposition
  head.txt              the ledger head printed by the trainer (the out-of-band anchor)
  trainer_log.json      TRL's log history

Setup:
  model   Qwen/Qwen2.5-0.5B-Instruct, fp32 master weights under bf16 autocast, vLLM serves bf16 in-process
  GPU     one A10G; TRL 1.13.0, vLLM 0.28.0
  TRL     use_vllm=True, vllm_mode="colocate", vllm_importance_sampling_correction=False
          (the loss never reads sampling logprobs; the recorder still records them)
  Why colocate, not server: TRL's server mode pushes weights over NCCL between two
  single-GPU-masked processes, and on Modal that handshake hangs in ncclCommInitRank.
  Reservoir probed eleven NCCL environments on 2026-10-06 (benchmarks/modal/nccl_probe.py
  there); every one hung, and this project's first attempts hung the same way. Colocate
  runs vLLM inside the trainer process and loads weights in-process, so there is no
  handshake. The lag-0 floor still measures vLLM's kernels against the trainer's forward.
  lag     num_iterations=2 and steps_per_generation=2 make each generation batch
          train over several optimizer steps, so the record contains lag 0, 1, 2, 3
          tokens; lag 0 is the vLLM-vs-trainer mismatch floor.
  reward  exact-match on 2-digit addition prompts (synthetic, no dataset download)

Usage
  pip install modal && modal setup                            # once
  modal run benchmarks/modal/trl_grpo_vllm.py                 # ~10 min on one A10G, about a dollar
  modal run benchmarks/modal/trl_grpo_vllm.py --temperature 0.7
  modal run benchmarks/modal/trl_grpo_vllm.py --batch-invariant   # deterministic vLLM kernels (colocate: the
                                                                   # override also hits the trainer's backward; may fail)
  modal run benchmarks/modal/trl_grpo_vllm.py --max-steps 24
  modal run benchmarks/modal/trl_grpo_vllm.py --tag probe            # keep the result dir separate
  modal run benchmarks/modal/trl_grpo_vllm.py --fp32-trainer         # the first run's misconfiguration

Precision. The trainer runs under bf16 autocast (GRPOConfig bf16=True) so that its
forward sees the same bf16-rounded weights vLLM holds. The first run (2026-10-06) used
an fp32 forward (bf16=False) against the bf16 engine: Adam updates of ~1e-5 per weight
are below the bf16 ulp of most weights, so the synced engine stayed within numerics of
the initial checkpoint while the fp32 trainer moved by a nat. That looked like a stale
engine and was not one; see docs/findings/2026-10-06-trl-colocate-sync.md.
--fp32-trainer reproduces it. Every run also writes sync_probe.json: at each weight
sync, every vLLM parameter compared elementwise with the trainer's (sync_probe.py).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

VLLM_VERSION = os.environ.get("MARTINGALE_VLLM_VERSION", "0.28.0")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GPU = "A10G"
RESULTS_DIR = Path(__file__).parent / "results"
# Modal re-imports this file as /root/trl_grpo_vllm.py inside the container, where it has
# no repo above it; the mounts below only matter locally, at image-definition time.
_here = Path(__file__).resolve()
REPO_ROOT = _here.parents[2] if len(_here.parents) > 2 else Path("/root")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}")
    .pip_install("trl==1.13.0", "transformers==5.17.0", "datasets==5.0.1", "accelerate==1.15.0",
                 "numpy==2.2.6", "click>=8.0")
    .env({"HF_HOME": "/hf_cache", "PYTHONPATH": "/repo/src:/repo"})
    .add_local_dir(str(REPO_ROOT / "src"), "/repo/src")
    .add_local_dir(str(REPO_ROOT / "checker"), "/repo/checker")
    .add_local_file(str(_here.parent / "sync_probe.py"), "/repo/sync_probe.py")
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


def _worst_tokens(ledger, tokenizer, n: int = 40) -> list[dict]:
    """The n scored tokens with the largest |trainer - behavior| log-prob, with decoded context,
    so a large floor can be read: tail tokens on a peaked distribution (numerics) or common
    tokens (sync/alignment)."""
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


@app.function(image=image, gpu=GPU, timeout=60 * 60, memory=32768, volumes={"/hf_cache": hf_cache})
def run_grpo(max_steps: int = 16, seed: int = 0, temperature: float = 1.0, batch_invariant: bool = False,
             fp32_trainer: bool = False) -> dict:
    import torch
    from sync_probe import SyncProbe
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig

    from checker.verify_tokens import verify_export
    from martingale.diagnostics import decompose, render_markdown
    from martingale.integrations.trl import MartingaleRecorder, build_trainer_class

    work = Path("/tmp/martingale_run")
    work.mkdir(parents=True, exist_ok=True)
    if batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    flight = MartingaleRecorder(str(work / "ws"), tokenizer=tok, sampler_overrides={
        "engine_version": f"vllm {VLLM_VERSION} colocate" + (" batch_invariant" if batch_invariant else "")})
    args = GRPOConfig(
        output_dir=str(work / "out"), seed=seed, max_steps=max_steps, logging_steps=1,
        per_device_train_batch_size=8, gradient_accumulation_steps=1, num_generations=8,
        max_completion_length=16, temperature=temperature, learning_rate=5e-6,
        num_iterations=2, steps_per_generation=2, report_to=[], save_strategy="no", bf16=not fp32_trainer,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.3, vllm_max_model_length=512,
        vllm_importance_sampling_correction=False,
    )
    Trainer = build_trainer_class()
    trainer = Trainer(model=model, args=args, train_dataset=_addition_dataset(512, seed),
                      reward_funcs=exact_match_reward, processing_class=tok, flight_recorder=flight,
                      halt_on_error=False)       # first real run: record alarms, never lose the data to one
    probe = SyncProbe(trainer)                   # elementwise vLLM-vs-trainer check at every weight sync
    print("SYNC_PROBE facts", json.dumps(probe.facts), flush=True)
    trainer.train()
    sync_probe = probe.summary()
    monitor = trainer.martingale_monitor
    alarms = [{"kind": a.kind, "level": a.level, "step": a.step, "message": a.message, "evidence": a.evidence}
              for a in monitor.alarms]
    per_step = [d.metrics() for d in monitor.diagnoses]
    head = flight.head()
    print(f"MARTINGALE_LEDGER_HEAD {head}")
    worst = _worst_tokens(flight.recorder.ledger, tok, n=40)
    flight.recorder.ledger.checkpoint_wal()      # tokens.db alone must hold everything before it is copied
    export = flight.recorder.export_for_checker(work / "export")
    verify = verify_export(export, expected_head=head)
    report = decompose(flight.recorder.ledger)
    return {
        "head": head, "verify": verify, "report": report, "report_md": render_markdown(report),
        "stats": flight.stats, "log_history": trainer.state.log_history,
        "alarms": alarms, "per_step_metrics": per_step, "worst_tokens": worst, "sync_probe": sync_probe,
        "tokens_db": (work / "ws" / "tokens.db").read_bytes(),
        "config": {"model": MODEL_ID, "vllm": VLLM_VERSION, "trl": "1.13.0", "gpu": GPU, "vllm_mode": "colocate",
                   "max_steps": max_steps, "seed": seed, "temperature": temperature,
                   "batch_invariant": batch_invariant, "trainer_bf16": not fp32_trainer,
                   "num_iterations": 2, "steps_per_generation": 2},
    }


@app.local_entrypoint()
def main(max_steps: int = 16, seed: int = 0, temperature: float = 1.0, batch_invariant: bool = False,
         fp32_trainer: bool = False, tag: str = ""):
    res = run_grpo.remote(max_steps=max_steps, seed=seed, temperature=temperature, batch_invariant=batch_invariant,
                          fp32_trainer=fp32_trainer)
    tag = f"_t{temperature}" + ("_bi" if batch_invariant else "") + ("_fp32" if fp32_trainer else "") + (f"_{tag}" if tag else "")
    out = RESULTS_DIR / f"trl_grpo_vllm_a10g_{max_steps}steps_seed{seed}{tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokens.db").write_bytes(res.pop("tokens_db"))
    (out / "report.md").write_text(res.pop("report_md"))
    (out / "head.txt").write_text(res["head"] + "\n")
    (out / "verify.json").write_text(json.dumps(res["verify"], indent=1))
    (out / "report.json").write_text(json.dumps(res["report"], indent=1))
    (out / "trainer_log.json").write_text(json.dumps(res["log_history"], indent=1))
    (out / "alarms.json").write_text(json.dumps({"alarms": res["alarms"], "per_step_metrics": res["per_step_metrics"]}, indent=1))
    (out / "worst_tokens.json").write_text(json.dumps(res["worst_tokens"], indent=1))
    (out / "sync_probe.json").write_text(json.dumps(res["sync_probe"], indent=1))
    (out / "config.json").write_text(json.dumps({**res["config"], "stats": res["stats"]}, indent=1))
    print((out / "report.md").read_text())
    print("verify ok:", res["verify"]["ok"], "| head:", res["head"], "| alarms:", len(res["alarms"]))
    print("results in", out)
