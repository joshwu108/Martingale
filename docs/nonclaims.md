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
- **Log-probs are whatever the source reported.** `logprobs_mode` in the
  revision says whether they came from the engine's sampler, the trainer's
  no-grad pass under the generating weights, or a recompute. The record binds
  the bits; it does not claim they are the true sampling probabilities.
- **Tamper evidence needs the anchor.** Without `TokenLedger.head()` published
  out of band, a forger who re-signs a whole chain, deletes the last sequence
  of an actor, or drops an unreferenced revision is not detected
  (`results/mutation_tokens_report.json`, column `rejected_unanchored`).
- **The TRL integration has run twice for real** (2026-10-06, Qwen2.5-0.5B,
  TRL 1.13, vLLM 0.28, 16 steps each; `benchmarks/modal/results/`), once in
  colocate mode and once in server mode. The adapter version that ran had two
  bugs (second `num_iterations` pass not scored; database copied before the
  WAL checkpoint), fixed the same day and covered by tests, but not yet re-run.
  TRL's server mode on Modal hangs in the NCCL weight-transfer handshake more
  often than not (one success in four attempts here, zero in Reservoir's
  eleven probes); colocate mode is the recommended path.
- No distributed-filesystem, multi-host, or performance claims for the record.
  Volume: about 250 bytes per token in JSON, less in SQLite; no compression
  or sampling of sequences is implemented. The diagnostics keep one float per
  scored token in memory for percentiles.
- **One workspace per training process.** `TokenLedger` serialises writers on
  one SQLite file (BEGIN IMMEDIATE, 30 s busy timeout) and each rank records
  under its own actor id, but sharing one file across ranks over a network
  filesystem is untested. Sharded (FSDP / ZeRO-3) state dicts are refused
  rather than digested wrongly; gather the weights first.
- **`logprobs_mode` names the source, not the engine's filtering.** It records
  whether the bits came from the engine's sampler, the trainer's no-grad
  pass, or a recompute. Whether vLLM reported pre- or post-temperature
  log-probs is a property of the engine configuration (vLLM `logprobs_mode`)
  and is not yet captured; at temperature != 1 the lag-0 floor therefore
  includes a temperature effect.
- The veRL and Lightning hooks publish revisions only; they do not record
  tokens. No claim of veRL or Lightning integration beyond that.
- History: until 2026-10-06 this layer was a prototype whose ledger the checker
  rejected. `tests/test_prod_checker.py` records the fix.

## Live diagnosis and alarms (`martingale.diagnostics.alarms`, TRL callback)

- Alarm thresholds (floor jump 3x, stale-server proxy 10x, ESS 30%/10%, span 5%,
  unmatched 1%, 32 lag-0 tokens minimum) are heuristics chosen before any real
  run existed. They are configuration, not findings, and should be revisited
  against the first vLLM run.
- `stale_server` is a proxy. The record holds the trainer's weights digest at
  generation time, not the inference server's, so a server serving old weights
  is inferred from the lag-0 floor jumping, not observed. `weights_unchanged`
  is exact but detects a trainer whose weights did not change, which is a
  different fault.
- Percentiles in the live diagnosis come from a bounded reservoir (4096 ratios
  per lag bucket); means, maxima and shares are exact.
- The monitor ran on three real runs. `stale_server` now fires on confident
  disagreement (engine >= 50% sure, trainer > 2 nats lower at lag 0), which
  bf16 rounding cannot produce; the thresholds (ln 0.5, 2 nats) are chosen
  for that reason, not fitted. It says the engine used different weights or
  inputs; it does not say why. `martingale recompute` needs a checkpoint and
  the prompt ids in the record (stored by default, outside the digest).
- The stale-engine finding on TRL 1.13 colocate + vLLM 0.28 is one model,
  one seed, one version pair; the mechanism inside TRL's weight sync has not
  been diagnosed.

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
- **No real run with replayed rows yet.** The evidence is the fake-trainer test
  (`tests/test_trl_replayed.py`), the record and checker tests, and the mutation
  campaign. D2's acceptance in Reservoir's plan (one TRL run with both adapters
  whose record both checkers accept and whose doctor report shows the replayed
  bucket) waits on the Reservoir-side wiring in `docs/replay-provenance.md` §4.
