# verl 0.9.1 integration notes

Pinned release: [`verl==0.9.1`](https://pypi.org/project/verl/0.9.1/), the latest stable PyPI release on 2026-10-07. Source references below are relative to the installed `verl/` wheel in `/Users/joshuawu/scratchpad/verl-0.9.1/.venv` (installed with `uv pip install --no-deps`).

## Data and hook points

- `experimental/agent_loop/agent_loop.py:1063-1087` builds a `TensorDict` with `prompts` (left padded), `responses` (right padded), `response_mask`, and optional `rollout_log_probs`, then returns a `DataProto`. `:122-127` shows the same names on the per-output path. `workers/config/rollout.py:219` has `calculate_log_probs`; it must be true to record engine probabilities. A missing `rollout_log_probs` is an error for this adapter: `old_log_probs` are actor recomputation and cannot prove what the engine reported.
- `trainer/ppo/ray_trainer.py:1509-1524` calls `async_rollout_manager.generate_sequences`, then merges those rows into the training batch. The driver should call `flight.on_rollout(batch, step=trainer.global_steps, weights=full_actor_weights)` on the generated `DataProto` **before** reward/correction changes masks. The hook adds `martingale_row_id` and `martingale_completion_length` tensors to `batch.batch`; `protocol.py:652-670, 702-716, 765-800` shows row selection, slicing, reordering and union carry batch tensors. The saved length keeps scores aligned if correction later changes `response_mask`. The driver must pass the same batch onward. Multi-turn `response_mask` may contain holes for tool text (`experimental/agent_loop/agent_loop.py:575-586`); the adapter refuses each such row, because the token record cannot represent an unscored tool token inside a completion without changing score positions: the row is not recorded, its `martingale_row_id` is -1 and its saved length 0, the loss path skips it (`stats["mask_hole_rows"]` at rollout, `stats["skipped_rows"]` per loss row), and the rest of the batch is recorded.
- `trainer/ppo/ray_trainer.py:1290-1324` computes actor `old_log_probs` with `actor_rollout_wg.compute_log_prob`. `:1571-1617` may instead alias `old_log_probs` to `rollout_log_probs` in bypass mode. Neither is a live optimizer-step score.
- `workers/utils/losses.py:57-65` materializes live `model_output["log_probs"]` with `no_padding_2_padding`; `:91-119` passes it to the policy loss. `workers/engine_workers.py:639-646` installs this function on the actor; `:243-340` loops mini-batches and epochs, and `:343-368` invokes it on each training batch. Wrap `ppo_loss` at `set_loss_fn` with `wrap_ppo_loss`: the wrapper calls `on_train_step(data, logps, step, weights=...)` for each loss invocation, before delegating to the original loss. A driver-only `_update_actor` hook sees no live per-token log-probs and misses repeated optimizer steps.
- `trainer/ppo/ray_trainer.py:1683-1690,1716` calls `checkpoint_manager.update_weights(global_steps)` after actor training; `checkpoint_engine/base.py:510-541` dispatches to the actor worker and rollout replicas. `workers/engine_workers.py:728-805` either sends checkpoint-engine weights or gathers actor `get_per_tensor_param()` and calls rollout `update_weights`. A numeric `global_steps` tag is not a digest. A full, unsharded actor state dict (or fully gathered sync payload) can be hashed once per step as the **expected** rollout revision. Sharded/meta/empty tensors are refused, as in TRL. A real engine sync requires a runtime check to establish that the served weights equal that digest; `on_rollout(..., received_weights_digest=...)` binds the replica's digest of what its last sync delivered into the rollout revision (see "Received-weights check" below), and without one the revision makes no sync claim.

## Existing mismatch diagnostics

`trainer/ppo/ray_trainer.py:1571-1617,1650-1660` supports bypass versus recomputed old log-probs and calls rollout correction in decoupled mode. `trainer/config/_generated_ppo_trainer.yaml:849-859` exposes `rollout_is`, threshold, rejection settings and `bypass_mode`; `workers/config/rollout.py:219` controls engine log-probs. `trainer/ppo/rollout_corr_helper.py:787-889,905-990` computes clipped IS/rejection and `rollout_corr/*` KL, perplexity, ratio and chi-square metrics. Those aggregate old-versus-rollout comparisons do not retain a per-sequence weight revision or score every reuse at its current weights. Martingale's record and monitor supply those dimensions.

## Integration constraints

Every recorder of a run, the driver's and each actor worker's, opens one shared workspace on a filesystem they all see. Row ids follow the TRL adapter's contract (`martingale.integrations._common`; docs/replay-provenance.md section 6): `on_rollout` packs its `actor_id` into each `martingale_row_id` (`pack_row_id`) and binds every id to its sequence digest in the ledger's `row_ids` table (`bind_row_ids`), so a worker whose separate recorder has an empty in-memory map resolves an id with one keyed lookup, not a scan; a recorder that reopens the workspace continues each rank's counter from that table. An id the workspace never bound (a separate workspace per rank) is unmatched, never matched by content. The paired hooks support a custom trainer/worker integration; the loss wrapper is the 0.9.1 extension point. An explicit `martingale_row_id` tensor is preferred. Content fallback (rows without an id) is ambiguous for repeated identical prompt/response rows across in-flight batches; it rotates among recent candidates, and in a worker with no in-memory map it still scans the ledger's sequences. Record each loss call under a distinct train revision when weights change; a cached full-weights digest per step assumes one optimizer weight version per step. For multiple PPO mini-batch updates inside a global step, pass a unique optimizer-step number to the worker hook. `response_mask` after rollout rejection is not the original generation mask; record rollout before correction.

## Real-run harness

`benchmarks/modal/verl_grpo_vllm.py` runs `RayPPOTrainer` (`trainer.use_v1=false`) with GRPO on one T4, and `benchmarks/modal/verl_hooks.py` holds its framework-free hooks (CPU-tested in `tests/test_verl_grpo_runner.py`; `tests/test_verl_real_data_path.py` drives verl's own union/reorder/`left_right_2_no_padding`/mini- and micro-batch path into `wrap_ppo_loss` when verl is installed). The driver wraps `async_rollout_manager.generate_sequences` and calls `on_rollout` on its output; an `ActorRolloutRefWorker` subclass wraps the actor's `partial(ppo_loss, config=...)` and the engine's `optimizer_step`. Steps are applied optimizer updates: with fp16 the grad scaler skips non-finite updates, and a skipped update is not a weight version. Gather weights after `optimizer_step`, never inside the loss: a `state_dict()` between forward and backward runs FSDP2's pre-save hook (`_to_sharded`), which reshards the root group while autograd holds its unsharded parameters, and would digest the fp16 unsharded views instead of the fp32 master weights. The digest source is `engine.get_per_tensor_param()`, the payload verl's naive sync sends to vLLM.

## What the first real run confirmed (2026-10-09)

`benchmarks/modal/results/verl_grpo_vllm_t4_8steps_seed0` (8 global steps, 1x T4; `adapter_checks.json` holds the verdicts). Line numbers refer to this file.

- Confirmed, line 7: every `generate_sequences` output (8/8) carried `prompts`, `responses`, `attention_mask`, `response_mask` and `rollout_log_probs`, with `calculate_log_probs=True`.
- Confirmed for single-turn only, line 8: 0 rows with mask holes in 512. The multi-turn refusal (`agent_loop.py:575-586`) was not exercised.
- Confirmed, line 8: `martingale_row_id` survived `batch.union` and `_balance_batch`'s reorder (order changed in 8/8 updates, same id set), and reached all 256 actor loss calls; 0 unmatched rows. With one GPU, `_balance_batch` still permutes rows (`ray_trainer.py:1240-1246`).
- Confirmed, lines 10 and 19: the wrapped loss saw log-probs carrying a grad graph in 256/256 calls and scored 3,541 tokens at lags 0 to 3. The worker scored through the ledger fallback of line 19 (separate process, empty in-memory map); training took 10.2 min for 8 steps. The cost of that scan was not separated from the rest. At the time that fallback scanned sequence labels for a `/row<N>` suffix; it is now the row-id table lookup (see "Row ids are shared with TRL" below).
- Contradicted, the README snippet that wrapped `worker.loss_fn` directly: the actor's loss is `partial(ppo_loss, config=actor_config)` (`workers/engine_workers.py:643`) and the FSDP engine calls it with keywords only (`workers/engine/fsdp/transformer_impl.py:1576`). Wrap `loss_fn.func` and rebind `loss_fn.keywords`, as `benchmarks/modal/verl_hooks.py` `WorkerHooks.install` does.
- Confirmed as far as it goes, line 11: digesting `engine.get_per_tensor_param()` gave the same weights digest for each rollout and for the first loss call at the same optimizer step (8/8). That is the expected rollout revision; the served vLLM weights were not compared, so line 11's runtime-check caveat stands.
- New: with fp16 the grad scaler skipped 1 of 32 optimizer steps (`grad_norm` inf), and a step with all-equal rewards gives `grad_norm` 0 and unchanged weights, which the shared monitor reports as `weights_unchanged` (error). Count steps by applied updates, not loss calls.

## Before the next GPU run (2026-10-13)

The three open items of the first run, implemented and CPU-tested against verl 0.9.1; none has run on a GPU yet.

- **Received-weights check** (line 11's runtime check). `LLMRevision.received_weights_digest` is an optional manifest key, digested when present (design.md 7.10). `benchmarks/modal/verl_served.py` registers a `vLLMReplica` subclass under verl's `"vllm"` name (`workers/rollout/replica.py:312-390`) whose `vLLMHttpServer` names a `vLLMColocateWorkerExtension` subclass; that extension wraps `_update_weights` for each `update_weights_from_ipc` call (`workers/rollout/vllm_rollout/utils.py:248-330`) and digests every bucket with `ReceivedWeights` before vLLM loads it, the same formula as the revision's weights digest, which is the worker's per-step snapshot of the tensors `engine.get_per_tensor_param()` yields (not a tee of the stream itself: a cast or rename between the two would show as a mismatch). The driver reads one report per TP worker of each server (`MartingaleHttpServer.martingale_received`; verl's own `collective_rpc` drops results; with DP > 1 only one shard per server) before each rollout; workers that disagree, or some reporting and some not, become a mismatch rather than an abort. A parameter received twice withholds the claim without breaking the sync. The replica holds one host copy of the payload per sync while digesting, fine at 0.5B, a cost to revisit for larger models. Equal: `sync_verified` is true. Different: the revision records the replica's digest, and the monitor raises `stale_server` with `observed: true`. A failed or absent report: no digest, no claim (`unverified_rollouts` in `adapter_checks.json` `received_weights_equal_rollout_revision`, which is then not ok). verl syncs before every rollout (`ray_trainer.py:1430,1716`, `rollout.load_format: dummy`), so every rollout of the next run should be verified. The digest proves what arrived, not what vLLM's loader did with it.
- **Multi-turn refusal** (line 8). Tested with fake `DataProto`s whose `response_mask` has holes, through `on_rollout`/`on_train_step` (`tests/test_verl_integration.py`), the fake trainer's union, reorder, mini- and micro-batches (`tests/test_verl_grpo_runner.py`), and verl's own union/reorder/`left_right_2_no_padding`/`make_iterator` path (`tests/test_verl_real_data_path.py`). `adapter_checks.json` `mask_holes` now also reports `refused_rows` and `skipped_loss_rows`.
- **Row-id lookup timing** (lines 10 and 19). The worker times each `ledger.row_digest` call (its maps are empty, so every row id goes there) and, separately, all of `on_train_step`: `adapter_checks.json` `row_id_lookup` and `config.json` `stats.row_lookup` carry `calls`, `found`, `seconds`, `max_seconds` and `score_seconds`.

## Offline floor-warning investigation (2026-10-10)

The four warnings in `alarms.json` were traced through `TokenLedger.all_sequences()`, `scores_for()`,
and `all_revisions()`; the token details are in the [results README](../benchmarks/modal/results/README.md).
`floor_jump@7` is the 103 lag-0 tokens scored at applied step 4; its largest gap is 0.0019655 nats
on answer digit `2` (`step4/row89`, position 0), and four copies account for 57.8% of the floor.
`floor_jump@11` is the 119 tokens scored at step 7; `3`/`4` at position 1 in four sequences account
for 92.9% of the floor, with a maximum gap of 0.0242845 nats. `floor_tail@27` is the 123 tokens
scored at step 23; two position-0 `8` tokens differ by 0.0133224 nats and the top four supply 79.2%.
`floor_tail@31` is the 107 tokens scored at step 27; two position-1 `7` tokens differ by 0.0084255
nats and the top four supply 86.4%. All those comparisons have lag 0 and matching **expected**
weights digests. None is a confident disagreement. The run's 910 lag-0 tokens have zero such cases.

The record proves small, concentrated gaps on ordinary answer tokens, not a low-probability tail or
an adapter error. fp16 vLLM versus the mixed-precision trainer is a plausible numerical source, but
the run did not compare the actual served weights or perform a same-kernel recomputation, so that
cause is inferred. The adapter checks passed (8/8 rollout/first-score expected digest pairs, all
256 loss calls with row IDs, zero unmatched rows). The large relative factors, especially 61× and
47×, divide by a trailing floor of only 5.81e-6. The previous `floor_tail` message claimed tail
tokens disagreed by nats; it now asks for token inspection and carries the score revision step.

The apparent step boundary is accounting: `DriverHooks.wrap_update` calls the monitor **after**
the update, using the worker's count of applied optimizer steps. Warnings at monitor steps 7, 11,
27, and 31 summarize first scores at revisions 4, 7, 23, and 27 respectively. At monitor steps
7, 11, and 27 the next rollout's published digest differs from that earlier score digest; after
the final update there is no next rollout/revision. This boundary explains the warning label and
must not be mistaken for a lag-0 comparison across weights. The applied-step count and thresholds
stay intact; alarm evidence now reports both the score revision step and monitor step.

`trainer_log.json` records `actor/grad_norm = inf` at global step 2. The worker audit records 31
applied updates and one skipped attempt. The ledger publishes trainer-score revisions at applied
steps 0–30; the skipped attempt adds no weight version. The global-step log does not locate the
overflow within its four mini-batches. A CPU fake-trainer test now inspects the revision list
immediately after a skipped attempt.
