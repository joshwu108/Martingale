# Why TRL's colocated vLLM looked stale: the sync was exact, the fp32 trainer was invisible in bf16

Date: 2026-10-06. Setup: Qwen/Qwen2.5-0.5B-Instruct, TRL 1.13.0 GRPO, vLLM 0.28.0, `vllm_mode="colocate"`,
one A10G on Modal, 16 optimizer steps, `num_iterations=2`, `steps_per_generation=2`, lr 5e-6, seed 0,
temperature 1.0. Runner: `benchmarks/modal/trl_grpo_vllm.py`; probe: `benchmarks/modal/sync_probe.py`.

## Verdict

The weight sync works. At every generation TRL pushed the trainer's weights into the in-process vLLM
model, and after each push **every one of the 494,032,768 vLLM parameters was bit-identical to the
trainer's fp32 parameter rounded to bf16** (probe, three syncs after step 0, zero differing elements).

What made the engine look stale is precision, not plumbing. The first run trained the model in float32
with no autocast (`GRPOConfig(bf16=False)`, model loaded in float32) while vLLM holds bf16 weights.
Adam moved each weight by about 1e-5 per step (max 1.8e-5 after four steps, 3.1e-5 after twelve). The
bf16 ulp of a weight of the checkpoint's median magnitude (0.0114) is 6.1e-5, so a change of 1e-5 rounds
back to the same bf16 value. The sync transmitted exactly what bf16 can represent: 12% of the elements
had moved by one ulp after four steps, 17% after twelve, all among the smallest weights, and the engine's
output stayed within numerics of the initial checkpoint. The fp32 trainer, whose forward sees every
sub-ulp change, had moved by 0.8 to 1.0 nats on the same tokens. The disagreement was real, it was
engine-versus-trainer mismatch, and it was our configuration.

Running the trainer under bf16 autocast (`bf16=True`, the TRL default recipe) removes it: the lag-0 floor
is 0.0001 to 0.03 nats at every generation, the recompute shows engine and trainer moving together, and
no `stale_server` alarm fires.

This is the mechanism Dirhoussi et al. (Hugging Face, April 2026) describe for TRL, including the
condition `K_cross ≈ |W| / (256 η)` steps for a weight to cross a bf16 boundary, and their Run B
(`DTYPE=float32, BF16=False, lr=1e-6`) is our first run's configuration. Miahi and Belilovsky
(arXiv 2602.03839) report that about 99% of per-step Adam updates at RL learning rates are invisible
after the bf16 cast.

## What is proven by the probe and the records, and what is inferred from code

Proven (artifacts under `benchmarks/modal/results/`, both runs verified by the independent checker):

- `sync_weights` ran four times, at global steps 0, 4, 8 and 12, once per generation
  (`sync_probe.json`, `sync_calls`, `records[*].global_step`).
- After each sync, `n_diff_vllm_vs_trainer = 0` and `max_abs_vllm_vs_trainer = 0.0` over all 170 vLLM
  parameters (494,032,768 elements). Fused `qkv_proj` and `gate_up_proj` and the tied embedding are
  included via the Qwen2 name mapping; `unmapped_vllm_params = []`.
- Each sync changed the tensors: immediately before the step-8 sync the embedding differed from the
  trainer in 299,381 elements and from the initial weights in 356,135 (the step-4 state); after it, 0 and
  506,630.
- vLLM's parameter storage did not move (`data_ptr_changed = []`), so the captured CUDA graphs
  (`cudagraph_mode FULL_AND_PIECEWISE`) read the updated memory. `model_runner.model` is the
  `Qwen2ForCausalLM` itself in this configuration and `load_weights` reports the name it loaded.
- The trainer's fp32 drift from the initial checkpoint and the number of elements whose bf16 value
  changed:

| sync at step | max abs fp32 drift | mean abs fp32 drift | elements changed in bf16 | share |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0% |
| 4 | 1.82e-05 | 7.60e-06 | 60,776,378 | 12.3% |
| 8 | 2.75e-05 | 9.65e-06 | 81,492,198 | 16.5% |
| 12 | 3.10e-05 | 1.05e-05 | 85,234,763 | 17.3% |

  The 896 elements of each RMSNorm weight (magnitude about 1, ulp 2^-7 = 0.0078) changed in fp32 by up to
  3e-5 and in bf16 not at all. The embedding (136M elements, 27% of the model) changed in 0.26% of its
  elements by step 4; `layers.0.mlp.down_proj` in 18%, `layers.0.self_attn.o_proj` in 29%.
- Offline, every fp32 value of the checkpoint sits exactly on the bf16 grid (it is stored in bf16), and
  the fraction of elements a uniform shift would move across a rounding boundary is 9.6% at 5e-6, 18.9%
  at 1.2e-5, 36% at 2e-5, 91% at 1e-4 (`|w|` quantiles 10/50/90%: 0.0020 / 0.0114 / 0.0302).
- Reference recompute of the fp32 run's record with the untouched initial checkpoint (CPU, fp32), per
  generation, mean |Δ log-prob| over the generated tokens:

| generation at step | ref vs engine | ref vs trainer (lag 0) | engine vs trainer |
|---|---|---|---|
| 0 | 0.0015 | 0.0000 | 0.0027 |
| 4 | 0.0013 | 0.8075 | 0.8072 |
| 8 | 0.0006 | 1.0289 | 1.0287 |
| 12 | 0.1123 | 0.2591 | 0.1481 |

  The engine's output at steps 4 and 8 is within numerics of the initial checkpoint although 61M and 81M
  of its weights had changed by one ulp. The trainer's disagreement sits entirely on the first completion
  token (the first digit of the answer): at step 4, position 0 differs by 2.72 nats on average and up to
  7.25, positions 1 to 3 by at most 0.003; at step 8, position 0 by 3.47 (max 9.26), later positions by
  at most 0.001. The engine gave the correct first digit log-prob -0.00007 and the trainer -7.26.
- The run is bit-reproducible: the probe run's per-step metrics and ledger head
  (`fc9ee53f8526e46b...`) equal the first run's, so the first run's record (which predates prompt-id
  storage and cannot be recomputed on its own) is the same record.
- Rewards stayed at 1.0 from step 5 to step 12 in the fp32 run: the engine was answering from
  (effectively) the initial policy, so every gradient was zero and TRL's logs showed nothing while the
  fp32 trainer, pushed by Adam momentum alone, kept moving.

The one-variable follow-up, `bf16=True` (fp32 master weights, bf16 autocast forward), everything else
identical (`trl_grpo_vllm_a10g_16steps_seed0_t1.0_probe/`):

| generation at step | bf16 elements changed | ref vs engine | ref vs trainer (lag 0) | engine vs trainer | reward of this batch |
|---|---|---|---|---|---|
| 0 | 0 | 0.0015 | 0.0107 | 0.0080 | 0.94 |
| 4 | 59,212,138 | 0.0014 | 0.0012 | 0.0001 | 1.00 |
| 8 | 89,035,250 | 2.7515 | 2.4577 | 0.0027 | 0.19 |
| 12 | 90,763,914 | 3.5841 | 3.4810 | 0.0274 | 0.00 |

  Engine and trainer agree at lag 0 at every generation (floor 0.008, 0.0001, 0.0027, 0.027), the sync is
  again exact (zero differing elements), and the policy change is now visible to both sides: by step 8
  both are 2.5 to 2.8 nats from the initial checkpoint. The doctor attributes 96% of the off-policy
  signal to staleness and 4% to mismatch; no `stale_server` alarm, one `floor_jump` warning at step 13.
  The reward collapse (1.0 to 0.19 to 0.0) is the same trainer trajectory the fp32 run hid: the first
  Adam step after a clipped gradient of norm 296 moves every weight by the learning rate, which on this
  saturated two-prompt task wrecks the first-digit decision. That is a property of the recipe, not of
  the sync, and is not analysed further here.

Inferred from the code read (TRL 1.13.0 in `/Users/joshuawu/Reservoir/.venv`, vLLM v0.28.0 source):

- The sync is in-place `copy_` end to end, and `torch.Tensor.copy_` from fp32 into a bf16 tensor rounds
  to nearest even. Nothing in the path is a no-op for an unquantized bf16 Qwen2 on CUDA.
- There is one silent no-op path in vLLM's loaders, `online_process_loader` returning early when a
  layer's reload info is not armed (`vllm/model_executor/model_loader/reload/layerwise.py:158-170`), but
  it is installed only for online-quantized models (`base_loader.py:75-80`, `torchao_decorator.py:29-55`)
  and not here.

## The code path

TRL (`trl/trainer/grpo_trainer.py`, `trl/generation/vllm_generation.py`):

- Generation happens every `steps_per_generation * num_iterations` micro-steps
  (`grpo_trainer.py:1595-1603`); with our config at `_step` 0, 4, 8, 12, i.e. global steps 0, 4, 8, 12,
  after the optimizer step of the previous micro-step.
- `sync_weights` runs right before `generate` whenever `self.state.global_step != self._last_loaded_step`
  (`grpo_trainer.py:1827-1830`, `_last_loaded_step = -1` at `:1124`; the same guard for `rollout_func` at
  `:2233-2236`). Under gradient accumulation it runs once per optimizer step; under `num_iterations > 1`
  it runs once per generation. There is no sync between iterations, by design.
- `VLLMGeneration._init_vllm` builds `LLM(model=model.name_or_path, ...)` without a `dtype`
  (`vllm_generation.py:345-364`), so vLLM uses the checkpoint's `torch_dtype`, bf16 for Qwen2.5. TRL 1.13
  exposes no vLLM dtype option for colocate mode (`GRPOConfig` has `cast_lm_head_to_fp32` only).
- `_iter_named_params` yields `(name, param.data)` from `model.named_parameters()`
  (`vllm_generation.py:473-479`): fp32 tensors, HF names (`model.layers.N.self_attn.q_proj.weight`,
  ...). With tied embeddings `named_parameters()` yields `model.embed_tokens.weight` once and no
  `lm_head.weight`.
- Colocate sync: for each tensor,
  `self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights([(name, param)])`
  (`vllm_generation.py:510-511`), then `self.llm.reset_prefix_cache()` (`:517`). Sleep mode
  (`vllm_enable_sleep_mode`) only adds `wake_up(tags=["weights"])` before the loop (`:489-492`); it was off.

vLLM 0.28.0:

- The attribute path exists in-process because TRL uses `distributed_executor_backend="external_launcher"`:
  `LLMEngine.model_executor` is set when the engine core is not a subprocess (`vllm/v1/engine/llm_engine.py:125`),
  `ExecutorWithExternalLauncher(UniProcExecutor)` has a `driver_worker` (`vllm/v1/executor/uniproc_executor.py:54,161`),
  `WorkerWrapperBase.__getattr__` forwards to the worker (`vllm/v1/worker/worker_base.py:337-338`).
- `model_runner.model` may be a `CUDAGraphWrapper` (`vllm/v1/worker/gpu_model_runner.py:5572`) whose
  `__getattr__` forwards `load_weights` to the wrapped `Qwen2ForCausalLM` (`vllm/compilation/cuda_graph.py:211-222`).
  In the probe run it was the model itself.
- `Qwen2ForCausalLM.load_weights` uses `AutoWeightsLoader` and skips `lm_head.` when embeddings are tied
  (`vllm/model_executor/models/qwen2.py:502-506`); `Qwen2Model.hf_to_vllm_mapper` maps `.q_proj/.k_proj/.v_proj`
  to `.qkv_proj` with shard ids `q/k/v` and `.gate_proj/.up_proj` to `.gate_up_proj` with ids 0/1
  (`qwen2.py:323-331`); `WeightsMapper.apply` attaches the shard id to the incoming tensor
  (`vllm/model_executor/models/utils.py:136-146`). An unknown name raises (`utils.py:360-393`); nothing is
  skipped silently.
- Every leaf loader copies into the live parameter: `QKVParallelLinear.load_weights` to `weight_loader`
  (`vllm/model_executor/layers/linear.py:1293-1316`, `:1165-1289`, ends in `param_data.copy_(loaded_weight)`),
  `MergedColumnParallelLinear` (`:939-962`, `:724-832`), `ColumnParallelLinear.weight_loader` (`:542-558`),
  `RowParallelLinear.weight_loader` (`:1608-1624`), `VocabParallelEmbedding.weight_loader`
  (`vllm/model_executor/layers/vocab_parallel_embedding.py:445-485`, `param[:n].data.copy_(...)`),
  `default_weight_loader` for the norms (`vllm/model_executor/model_loader/weight_utils.py:1222-1241`).
  The dtype cast happens inside these `copy_` calls.
- `UnquantizedLinearMethod.process_weights_after_loading` is CPU-only (`linear.py:198-202`); on CUDA the
  parameters created at load time are the ones the kernels read, so the copies land where the forward
  looks. CUDA graphs capture addresses, not values, and the addresses did not change (probe).
- Tied embeddings: `ParallelLMHead.tie_weights` sets `layer.weight = embed_tokens.weight`
  (`vocab_parallel_embedding.py:575-577`, `:80-85`), so loading the embedding updates the head.

Questions from the task, answered:

| question | answer |
|---|---|
| when does `sync_weights` run relative to generation and `optimizer.step()` | immediately before each vLLM `generate`, after the optimizer step that advanced `global_step`; never between the `num_iterations` passes over one generation batch |
| does it run under gradient accumulation / `num_iterations > 1` as we exercised it | yes: four calls at steps 0, 4, 8, 12 (probe) |
| does it cast fp32 trainer weights to bf16 | yes, implicitly, in `copy_` inside vLLM's weight loaders; the cast is round-to-nearest and is where sub-ulp updates vanish |
| tied embeddings and fused QKV / gate-up for Qwen2 | handled: HF names are mapped to `qkv_proj` / `gate_up_proj` shards, `lm_head.` is skipped because the head shares the embedding tensor; probe confirms 0 differing elements including these |
| any condition under which `load_weights` is a no-op or loads into a copy | not in this configuration. Sleep mode adds a wake-up but still copies in place; `model_impl="auto"` resolves to vLLM's native Qwen2; cudagraphs read the same storage; the only silent-return loader is the online-quantization reload wrapper, not installed here |

## What this means for the record and the doctor

- The record was right and the first reading was wrong twice: the lag-0 disagreement is neither bf16
  tail noise nor a failed sync. It is a trainer whose forward runs at a precision the engine's weights
  cannot represent. `stale_server` fired on a confident disagreement, which is the right signal (the
  engine and the trainer disagreed on a token both were certain about), but the alarm text names only
  two causes (different weights, different inputs). A third exists: the trainer's forward and the engine's
  weights live at different precisions. The thresholds and rules were not changed in this branch; the
  text should mention the third cause, and `martingale recompute` already separates them: the engine
  matching the initial checkpoint while the probe shows the sync exact is the signature of this case.
- The engine floor measured under matched precision (bf16 autocast) at temperature 1 on this task is
  0.0001 to 0.03 nats mean |log r|, with a maximum of 0.25.
- The earlier "T=0.7 floor 6e-5" (server-mode run) is not a temperature effect: that run had zero
  gradient at every step (all rewards 1.0), so the trainer never moved and the floor is pure kernel
  numerics.

## Minimal reproducer

No GPU:

```python
import torch
w = torch.tensor([0.0114])                     # median |w| of Qwen2.5-0.5B-Instruct, on the bf16 grid
assert (w + 1e-5).to(torch.bfloat16) == w.to(torch.bfloat16)   # one Adam step at lr 5e-6 ... 1e-5: invisible
assert (w + 4e-5).to(torch.bfloat16) != w.to(torch.bfloat16)   # half-ulp at this magnitude is 3.05e-5
```

```python
# fraction of a real checkpoint that a uniform fp32 shift can move in bf16 (CPU, ~1 min)
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
a = torch.cat([p.detach().flatten().abs() for p in m.parameters()])
half_ulp = torch.exp2(torch.floor(torch.log2(a.clamp_min(1e-30))) - 8)
for d in (5e-6, 1.2e-5, 2e-5, 1e-4):
    print(d, (half_ulp < d).float().mean().item())      # 0.096, 0.189, 0.362, 0.909
```

With a GPU (one A10G, about 10 minutes, about a dollar each), the two runs above:

```bash
modal run benchmarks/modal/trl_grpo_vllm.py --fp32-trainer --tag probe   # fp32 forward: floor 0.81 / 1.03 at steps 4 / 8
modal run benchmarks/modal/trl_grpo_vllm.py --tag probe                  # bf16 autocast: floor 1e-4 / 3e-3
uv run --with transformers --with torch martingale recompute --dir benchmarks/modal/results/<run> --model Qwen/Qwen2.5-0.5B-Instruct
```

Both write `sync_probe.json`, the elementwise comparison after every sync.

## Upstream

Not a TRL or vLLM bug: both do what their code says. Two usability points are worth raising with TRL,
and the HF team has already published the analysis, so this is optional. Draft, in case it is wanted:

> **Title:** GRPOTrainer with vLLM: warn when the policy forward runs in fp32 against a bf16 engine
>
> **Versions:** trl 1.13.0, vllm 0.28.0, transformers 5.17.0, torch 2.14.0; Qwen/Qwen2.5-0.5B-Instruct,
> `vllm_mode="colocate"`, one A10G.
>
> **What happens.** With the policy loaded in float32 and `bf16=False`, the trainer's forward sees
> every Adam update while the colocated vLLM model, which is bf16, sees only the updates that cross a
> bf16 rounding boundary. At lr 5e-6 the per-weight update is about 1e-5, below the ulp of most weights
> (median |w| 0.0114, ulp 6.1e-5). An elementwise probe after each `sync_weights` shows vLLM's parameters
> bit-identical to `trainer_param.to(bfloat16)` (0 of 494M elements differ), while the engine's
> completions stay within 0.001 nats of the initial checkpoint for 8 optimizer steps and the trainer's
> log-probs on the same tokens move by 0.8 to 1.0 nats. Rewards stay flat, so nothing in the logs shows
> it; the trainer diverges silently. With `bf16=True` and nothing else changed, engine and trainer agree
> to 0.0001 to 0.03 nats at every generation. This is the mechanism in the HF post "Defeating the
> trainer-generator precision mismatch in TRL" (April 2026), Run B.
>
> **Expected.** Either a warning at `GRPOTrainer.__init__` when `use_vllm` is set, the policy's dtype is
> float32 and neither `bf16` nor `fp16` autocast is enabled (the engine will run bf16 and cannot see the
> trainer's updates), or a `vllm_dtype` option for colocate mode so the engine can be run in float32
> for debugging. A sentence in the vLLM integration docs would also do.
>
> **Repro.** `GRPOConfig(use_vllm=True, vllm_mode="colocate", bf16=False, learning_rate=5e-6,
> num_iterations=2, steps_per_generation=2, ...)` with `AutoModelForCausalLM.from_pretrained(id,
> dtype=torch.float32)`; compare `vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner.model`
> parameters with `trainer.model` parameters cast to bf16 after each `sync_weights`, and the engine's
> sampling log-probs with a recompute from the initial checkpoint. Full probe and records:
> github.com/joshwu108/Martingale `benchmarks/modal/sync_probe.py`,
> `benchmarks/modal/results/trl_grpo_vllm_a10g_16steps_seed0_t1.0_fp32_probe/`.

## Changes made in this branch

- `benchmarks/modal/sync_probe.py` (new): the elementwise sync probe, run on every generation.
- `benchmarks/modal/trl_grpo_vllm.py`: trainer under bf16 autocast by default (`--fp32-trainer` reproduces
  the first run), `--tag` for the results directory, `memory=32768` for the probe's snapshots,
  `sync_probe.json` written next to `report.md`.
- `benchmarks/modal/results/trl_grpo_vllm_a10g_16steps_seed0_t1.0_fp32_probe/`: the first run re-executed
  with the probe (bit-identical record) and `..._t1.0_probe/`: the bf16 autocast run.
- README, `benchmarks/modal/results/README.md`, `paper/claim_evidence.md`, `docs/nonclaims.md` corrected.
  Diagnostics and alarm thresholds untouched.

## Sources

- Dirhoussi et al., "Defeating the trainer-generator precision mismatch in TRL", Hugging Face, 2026-04-04,
  https://aminedirohf-trainer-generator-bf16-mismatch.hf.space/
- Miahi, Belilovsky, "Understanding and Exploiting Weight Update Sparsity for Communication-Efficient
  Distributed RL", arXiv:2602.03839.
- TRL 1.13.0 source: `trl/trainer/grpo_trainer.py`, `trl/generation/vllm_generation.py`.
- vLLM v0.28.0 source (tag `v0.28.0`): files cited inline above.
