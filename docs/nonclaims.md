# Non-claims

This document lists explicit non-claims of the `martingale` project.
Every result is scoped to the constraints below.

## Scope of MDPs

- Results apply **only** to finite, tabular MDPs with a small number of
  states (|S| ≤ 5), actions (|A| ≤ 3), and horizon (H ≤ 5).
- No claim is made about continuous state/action spaces.
- No claim is made about function-approximated value functions or policies.

## Policy parameterization

- The rational simplex parameterization is **not** a softmax parameterization.
  The exact identities proven here do not automatically transfer to softmax
  or neural-network policies.
- No claim is made about importance-weighted estimators under softmax policies
  beyond what the mathematics directly implies.

## Scale

- All experiments run on a single host, CPU-only.
- No claim is made about distributed asynchronous systems beyond the
  two-actor, single-host prototype in `pipeline.py`.
- No claim is made about RLHF-scale or deep-RL-scale pipelines.

## Randomness / security

- The BLAKE2b keyed draw is a reproducible pseudorandom device.
  It is **not** a cryptographic security boundary.
- Seeds are exposed and documented; this is intentional for reproducibility.

## Float results (T4)

- T4 documents the frequency and magnitude of clip-boundary flips.
  It does **not** claim that these flips cause learning failures in practice.
- The float study is limited to float32 and float64 with the specific
  parameterizations enumerated in `campaigns/float_baselines.py`.

## Generality of staleness results (T3)

- Staleness scaling results apply to the exact rational simplex policies
  in the preregistered MDP family (T3 protocol in `docs/preregistration.md`).
- No claim is made about monotone bias growth outside this MDP family.

## Protocol (T5)

- The TLA+ model covers a finite scope (2 actors, 3 revisions, 2 actions).
  Correctness beyond this scope is not claimed.
- The implementation in `pipeline.py` is a single-host prototype using
  Python `multiprocessing`. No claim is made about correctness under
  OS-level scheduling policies other than those tested.

## Token record and production layer (`src/martingale/record/`, `prod/`, `integrations/`)

- **The draw is not verifiable.** The checker verifies binding (digests,
  chains, revision existence, sampler consistency, the anchored head). It
  cannot verify that a token was actually sampled from the recorded
  distribution: inference engines do not expose a keyed draw. This is the
  difference between the exact ledger (draw re-derived) and the token record.
- **Log-probs are whatever the source reported.** `logprobs_source` in the
  revision says whether they came from the engine's sampler, the trainer's
  no-grad pass under the generating weights, or a recompute. The record binds
  the bits; it does not claim they are the true sampling probabilities.
- **Tamper evidence needs the anchor.** Without `TokenLedger.head()` published
  out of band, a forger who re-signs a whole chain, deletes the last sequence
  of an actor, or drops an unreferenced revision is not detected
  (`results/mutation_tokens_report.json`, column `rejected_unanchored`).
- **The TRL integration has run four times for real** (2026-10-06, Qwen2.5-0.5B,
  TRL 1.13, vLLM 0.28, 16 steps each; `benchmarks/modal/results/`), three in
  colocate mode and once in server mode. The adapter version of the first two
  runs had two bugs (second `num_iterations` pass not scored; database copied
  before the WAL checkpoint), fixed the same day, covered by tests and re-run.
  The server-mode run had zero gradient at every step, so it is evidence of
  the plumbing only, not of any floor. TRL's server mode on Modal hangs in the
  NCCL weight-transfer handshake more often than not (one success in four
  attempts here, zero in Reservoir's eleven probes); colocate mode is the
  recommended path.
- No distributed-filesystem, multi-host, or performance claims for the record.
  Volume: about 250 bytes per token in JSON, less in SQLite; no compression
  or sampling of sequences is implemented. The diagnostics keep one float per
  scored token in memory for percentiles.
- **One workspace per training process.** `TokenLedger` serialises writers on
  one SQLite file (BEGIN IMMEDIATE, 30 s busy timeout) and each rank records
  under its own actor id, but sharing one file across ranks over a network
  filesystem is untested. Sharded (FSDP / ZeRO-3) state dicts are refused
  rather than digested wrongly; gather the weights first.
- **The log-prob mode is bound, with these limits.** Since 2026-10-10 the
  revision carries vLLM's `logprobs_mode` (raw or processed), `logprobs_source`
  and `temperature_applied`, and the doctor withholds the lag-0 floor when the
  record shows the behavior and trainer log-probs differ in temperature scaling
  at temperature != 1 (design.md 7.3 has the TRL and verl source lines). Limits:
  (1) records written before that date carry the source name in `logprobs_mode`
  and no `temperature_applied`; their floor is reported as before, with a caveat
  line at temperature != 1. (2) In TRL server mode the recorded mode is the flag
  TRL's own `vllm-serve` passes, not an observation of the server; a server
  started by hand without `--logprobs-mode processed_logprobs` returns raw
  log-probs and must be declared through `sampler_overrides`. For verl the mode
  is read from the `rollout_config` passed to the recorder; without it the mode
  is recorded as unknown. (3) Processed log-probs are renormalised after
  top-k/top-p/min-p and repetition penalty, which neither trainer applies; at
  non-default filters the floor includes that effect and the record does not
  separate it. (4) `temperature_applied` for a trainer pass is read from the
  pinned framework's source, not measured.
- The veRL and Lightning hooks publish revisions only; they do not record
  tokens. No claim of veRL or Lightning integration beyond that.
- History: until 2026-10-06 this layer was a prototype whose ledger the checker
  rejected. `tests/test_prod_checker.py` records the fix.

## Live diagnosis and alarms (`martingale.diagnostics.alarms`, TRL callback)

- Alarm thresholds (floor jump 3x, stale-server proxy 10x, ESS 30%/10%, span 5%,
  unmatched 1%, 32 lag-0 tokens minimum) are heuristics chosen before any real
  run existed. They are configuration, not findings, and should be revisited
  against the first vLLM run.
- `stale_server` is a proxy except where the replica reports what it received.
  The record holds the trainer's weights digest at generation time. Since
  2026-10-13 a rollout revision can also hold `received_weights_digest`, the
  digest the rollout replica computed over the payload its last sync delivered
  (the verl runner reports it; the TRL adapter does not). A mismatch there is
  an observed `stale_server` (evidence `observed: true`). It proves what
  arrived, before vLLM's loader runs, not that the loader applied it, and it is
  CPU-tested only until the next GPU run. Without that digest the revision
  makes no sync claim and a server serving old weights is still inferred from
  the lag-0 floor (`observed: false`). `weights_unchanged` is exact but detects
  a trainer whose weights did not change, which is a different fault.
- Percentiles in the live diagnosis come from a bounded reservoir (4096 ratios
  per lag bucket); means, maxima and shares are exact.
- The monitor ran on four real runs. `stale_server` now fires on confident
  disagreement (engine >= 50% sure, trainer > 2 nats lower at lag 0), which
  kernel numerics on the same weights cannot produce; the thresholds (ln 0.5,
  2 nats) are chosen for that reason, not fitted. It names a symptom, not a
  cause: stale weights, different inputs, and a trainer forward at a precision
  the engine's weights cannot represent (an fp32 forward against a bf16 engine,
  which is what the first run had) all raise it. `martingale recompute` needs a
  checkpoint and the prompt ids in the record (stored by default, outside the
  digest); the elementwise sync probe (`benchmarks/modal/sync_probe.py`) is
  what separates a failed sync from a precision gap, and it is a benchmark
  script, not part of the library.
- The "stale engine" reading of the first run was wrong. TRL 1.13's colocate
  sync was exact on that run (`docs/findings/2026-10-06-trl-colocate-sync.md`);
  the finding is a precision gap, our configuration, on one model, one seed,
  one version pair, tensor parallel 1, no sleep mode, no quantization. No claim
  about the sync under other configurations beyond the code read.
- The matched-precision engine floor measured here (0.0001 to 0.03 nats mean
  abs log ratio at temperature 1) is one task and one model; the task saturates
  the reward and the recipe collapses the policy within 12 steps, so the floor
  numbers at steps 8 and 12 are measured on a degenerate policy.

## Knob advisor (`martingale doctor --advise`, `diagnostics/advise.py`)

- **A suggestion, never a setting.** Nothing reads the advice back into a trainer;
  the doctor does not check whether a later run followed it.
- **No promise that the suggested knobs train well.** The advice keeps a statistic
  (token-level ESS/n per lag bucket) above a threshold. ESS/n says how many tokens
  the importance weights effectively use; it says nothing about reward, accuracy,
  collapse, or whether the clipped objective was already compensating.
- **Nothing beyond the lags that ran.** `within_observed` means every qualifying lag
  passed, not that a larger interval would. The sync interval is never
  extrapolated; only the lr ceiling at `--target-interval` is, and it is marked
  "not measured" unless the target is the passing interval.
- **The lr ceiling is a published empirical law applied, not measured here.**
  Song et al. (arXiv:2607.01083) fit "largest stable lr ~ 1/S" on vanilla GRPO
  sweeps and prove an O(S·η) upper bound on the staleness bias under local
  regularity conditions. Martingale holds S·η at the budget that passed in the
  record; none of the committed runs varied S or η, so no record here tests the
  law, its constant, or whether it holds for PPO, other models, or our recipes.
- **An ESS failure is not shown to be lr-sensitive.** Cutting the lr in proportion
  to the interval assumes the failing lag fails from staleness bias. A failure from
  the engine floor or from sampling noise does not respond to a lower lr, and
  the record cannot tell these apart. The number is a heuristic budget, not a
  measured stability limit.
- **The threshold is calibrated on five records, four runs, one model.**
  0.5 falls in the gap between the sorted bucket values in `docs/advise.md`
  (0.904 and up versus 0.351 and 0.398). The split comes from that gap, not from
  labels given beforehand; only one of the two low buckets has a known cause.
  Any value in (0.398, 0.904] splits them the same way. Buckets
  hold 93 to 930 tokens, and an ESS from ~100 tokens moves with its largest few
  ratios: the replay run's lag-2 bucket fails while lags 1 and 3 pass.
- **The knob translation assumes lag comes from the knobs.** S // U iterations and
  S - U async steps hold for a recipe whose lag is exactly U·K-1 or U+a-1; in-flight
  weight updates, partial rollouts, or a framework that counts async in generations
  rather than optimizer steps change the mapping, and the doctor cannot see which.
- **Exact where the inputs are.** Shares, intervals and learning rates are exact;
  ESS/n is a float64 statistic and is compared as one, so a bucket within float
  rounding of the threshold can land on either side.

## Replayed rows (`src/martingale/record/replay.py`, replay-buffer provenance)

- **The buffer's draw is the buffer's claim.** The record binds `draw_id`,
  `content_digest`, the importance weight and the rescale into the sequence
  digest, so none of them can change afterwards. It does not and cannot verify
  that `draw_id` names a draw the buffer actually made, that the draw followed
  the buffer's declared priorities or admission policy, or that
  `content_digest` is the digest of this row under the buffer's own digest
  function: the record never sees the buffer's sampler or its log. Those are
  the buffer's checker's job (Reservoir's `reservoir-verify` over its own
  attestation log); the two records are joined by draw id and content digest,
  and that join is a manual step with no tool behind it yet.
- **The weight is what the buffer reported.** The checker verifies that the
  importance weight and the rescale are bound and well-formed (reduced,
  non-negative, positive rescale), not that the weight equals the true
  importance ratio, nor that the rescale reached the loss (Reservoir's batch
  witness covers the tensors the loss consumed). The doctor's declared-versus-
  measured comparison is a float report with a heuristic threshold (1 nat of
  per-row dispersion after centring per train step); it cannot tell a buffer
  computing ratios against the wrong revision from a buffer whose stored
  log-probs come from a different source than the engine's (trainer forward
  versus vLLM sampler), and because buffers normalise weights per batch only
  the centred dispersion is meaningful, not the raw mean.
- **Origin binding holds only inside one record.** A replayed row names the
  fresh sequence it copies only when that sequence is in the same `tokens.db`;
  a row restored from another run, or from a buffer filled before the recorder
  attached, has no origin and its behaviour bits are the buffer's claim. The
  trainer-level lookup matches a fresh sequence by prompt, completion and
  generation step; two identical rows in one generation are told apart only by
  the behaviour log-probs the buffer passes, and when the buffer's log-probs
  come from a different source than the record's bits (trainer forward versus
  vLLM sampler) no row matches and the row is skipped rather than guessed.
- **A copy witnesses its origin's digest, not its scores.** Re-signing anything
  inside the origin (tokens, reward, header) or deleting it is caught through
  the copy without an anchor; the origin's score chain hangs off the digest and
  is not witnessed. Dropping the replay block or the origin from a copy and
  re-signing it is caught only with the anchor.
- **Lag is from the record, age is from the buffer.** A replayed token's lag is
  the train revision's step minus the recorded behaviour revision's step; the
  buffer's own notion of age (`max_policy_age`, model versions) is not read.
- **Everything true of fresh rows stays true.** The token draw is not
  verifiable; the behaviour log-probs are whatever the engine reported when the
  row was fresh; the lag-0 floor assumptions carry over to the replayed bucket,
  which borrows the fresh floor when it has no lag-0 tokens of its own.
- **Multi-process row ids are tested with two fake ranks in one process.** The
  rank-packed ids and the shared-workspace resolution (`docs/replay-provenance.md` §6)
  have not run on two GPUs or through `accelerate`; nothing is claimed about SQLite
  write contention under many ranks, and a record without a shared workspace loses
  the scattered rows to `unmatched` rather than recording them.
- **One real run with replayed rows, with a synthetic buffer.** The run of
  2026-10-09 (`docs/replay-provenance.md` §5) used `benchmarks/modal/replay_inject.py`,
  not Reservoir: 12 replayed rows of 76, one seed, one model, one recipe. It shows the
  Martingale half of D2 (record, doctor, checker) on real numbers and nothing about
  Reservoir's wiring, its checker, or the join between the two records, which waits
  on `docs/replay-provenance.md` §4. The buffer's declared weights were exact at the
  step they were computed and off by nats a step later on one tail row; the doctor
  reports that gap and does not judge it.
