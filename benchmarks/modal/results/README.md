# Real-run results (2026-10-06)

Five runs of `benchmarks/modal/trl_grpo_vllm.py`: Qwen2.5-0.5B-Instruct, TRL 1.13.0 GRPO, vLLM 0.28.0,
16 optimizer steps, `num_iterations=2`, `steps_per_generation=2`, 8 generations per prompt, 2-digit
addition with exact-match reward, seed 0, lr 5e-6. Each directory holds the doctor report (`report.md`,
`report.json`), the anchored checker verdict (`verify.json`, `head.txt`), the alarms and per-step metrics
(`alarms.json`), TRL's log (`trainer_log.json`), the record (`tokens.db`), the 40 worst tokens with
decoded text (`worst_tokens.json`) and `config.json`. The two `_probe` directories also hold
`sync_probe.json`: after every weight sync, every vLLM parameter compared elementwise with the trainer's
(`benchmarks/modal/sync_probe.py`).

| run | mode | trainer forward | lag-0 mean abs log r (per generation) | engine == bf16(trainer) after sync | checker |
|---|---|---|---|---|---|
| `..._t1.0` | colocate, 1x A10G | fp32, no autocast | 0.0027, 0.81, 1.03, 0.15 | not probed | ok, anchored |
| `..._t1.0_fp32_probe` | colocate, 1x A10G | fp32, no autocast | 0.0027, 0.81, 1.03, 0.15 (bit-identical re-run) | yes: 0 of 494,032,768 elements differ, at all 4 syncs | ok, anchored |
| `..._t1.0_probe` | colocate, 1x A10G | bf16 autocast (`bf16=True`) | 0.008, 0.0001, 0.0027, 0.027 | yes: 0 differ | ok, anchored |
| `..._t0.7` | server, 2x A10G | fp32, no autocast | 6.1e-05 (zero gradient at every step; trainer never moved) | n/a | ok, anchored |
| `..._t1.0_replay1of4_replay` | colocate, 1x A10G | bf16 autocast; 4 of 16 rows per batch after the first replayed from earlier batches (`replay_inject.py`) | 0.008, 0.0001, 0.0065, 0.0007 | yes: 0 differ at all 4 syncs | ok, anchored; 76 sequences, 12 replayed with origin |
| `verl_grpo_vllm_t4_8steps_seed0` (2026-10-09) | verl 0.9.1 `RayPPOTrainer`, vLLM 0.24.0 rollout, 1x T4 | fp16 mixed precision, fp32 master | 2.6e-05, 1.3e-04, 7.8e-04, 3.2e-06, 5.8e-06, 2.2e-06, 3.5e-04, 2.7e-04 (whole run 2.0e-04, 910 tokens) | not probed; the rollout revision digest equals the trainer digest at all 8 rollouts (expected weights, not the served ones) | ok, anchored |

## The verl run (2026-10-09)

`benchmarks/modal/verl_grpo_vllm.py --max-steps 8`: Qwen2.5-0.5B-Instruct, verl 0.9.1 GRPO, 8 prompts x 8
generations per step, `ppo_mini_batch_size=4`, `ppo_epochs=2` (4 optimizer steps per rollout), lr 1e-6, the
same addition task, seed 0, 10.2 min of training. 512 sequences, 3,541 scored tokens, 0 unmatched rows, lags 0
to 3. Every assumption in `adapter_checks.json` held (batch keys, no mask holes, row ids through union and
balance reorder in 8/8 updates and 256/256 loss calls, live log-probs in 256/256 loss calls, rollout digest ==
trainer digest, `martingale/*` in all 8 logged steps). 31 optimizer steps applied, 1 skipped by the fp16 grad
scaler (`grad_norm` inf at global step 2). The `weights_unchanged` error alarm is correct: global step 1 had
reward 1.0 on every completion, `grad_norm` 0 and `pg_loss` 0, so the rollouts at applied steps 0 and 4 carry
the same weights digest (the alarm names them by the monitor's post-update step, 4 and 7). Four floor
warnings (`floor_jump` at 7 and 11, `floor_tail` at 27 and 31) were raised and not investigated.

Against `..._t1.0_probe`, the lag-0 floor is not in the same range: 2.0e-04 for the whole verl run against
1.0e-02 for the TRL run, with per-generation values overlapping only between 1.3e-04 and 7.8e-04. The runs
differ in framework, GPU, dtype (fp16 vs bf16), step count and learning rate; which of these accounts for the
difference was not measured.

## The headline, corrected

The `_t1.0` run looked like a stale engine: a reference recompute with the initial checkpoint agrees
with vLLM to 0.001 nats at the generations at steps 4 and 8 and disagrees with the trainer by 0.8 to
1.0. The probe shows the sync was exact every time. The cause was ours: the trainer ran its forward in
float32 with no autocast while vLLM holds bf16 weights. Adam's updates (about 1e-5 per weight) are
below the bf16 ulp of most weights, so after the (correct) sync the engine held the initial value for
83 to 88% of its elements and stayed within numerics of the initial policy, while the fp32 forward saw
every update. Rewards stayed at 1.0 because the engine kept answering from the initial policy, so the
gradients were zero and nothing in TRL's logs moved. Details, code lines and numbers:
`docs/findings/2026-10-06-trl-colocate-sync.md`.

`_t1.0_probe` is the one-variable fix (`bf16=True`, now the runner default). Engine and trainer agree
at every generation, the policy change is visible to both (both 2.5 to 3.6 nats from the initial
checkpoint by steps 8 and 12), and the same recipe is seen collapsing the policy (reward 1.0, 1.0, 0.19,
0.0 over the four generation batches), which the fp32 run had hidden.

The `_t0.7` server-mode floor of 6e-5 is not a temperature result: every step of that run had zero
gradient (all rewards 1.0), so its lag-0 numbers measure kernel numerics on unchanged weights.

## Replayed rows (2026-10-09)

`_t1.0_replay1of4_replay` is the bf16 recipe with a synthetic replay buffer attached
(`benchmarks/modal/replay_inject.py`, `--replay-fraction 1/4`): after each generation batch
after the first, 4 of its 16 rows are replaced by completions from earlier batches and
registered as replayed rows with their origin, draw id, content digest and an exact importance
weight. The directory also holds `replay.json` (rows replaced per step, draw ids, row ids, pool
size). The checker accepts the record anchored on `head.txt` with 12 replayed sequences bound to
fresh origins; the doctor buckets them apart from fresh rows at lags 4..15 and reports the
buffer's declared sequence-level ESS/n (0.83) next to the one measured from the trainer's scores
(0.83); the one alarm is an `ess_collapse` at step 11 on fresh rows at lag 2. Predictions made
before the run and what held are in `docs/replay-provenance.md` §5.

## Known defects of earlier adapter versions (fixed the same day)

- The row lookup consumed ids on first use, so the second `num_iterations` pass was not scored; the
  `_t0.7` record holds lags 0 and 1 only and its `alarms.json` shows `unmatched` errors.
- `tokens.db` was copied before the SQLite write-ahead log was checkpointed and was empty for the first
  two runs; the exports used by the checker and the reports were complete. The empty files were removed.
  The `_t1.0` record predates prompt-id storage and cannot be recomputed on its own; `_t1.0_fp32_probe`
  is the same record (same ledger head) with prompt ids.
- `stale_server` fired on the `_t1.0` run's floor jumps and was re-labelled as tail numerics because
  the median ratio stayed at 1. The alarm now uses confident disagreement (engine >= 50% sure, trainer
  > 2 nats lower), which kernel numerics cannot produce; `floor_tail` covers the genuine tail case. The
  disagreement it caught here was a precision mismatch between the trainer's forward and the engine's
  weights, a third cause next to stale weights and different inputs.
