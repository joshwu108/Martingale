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
- **The TRL integration has been exercised only against a fake trainer**
  (`tests/test_trl_integration.py`). The Modal runner
  (`benchmarks/modal/trl_grpo_vllm.py`) that would produce a real vLLM + TRL
  record has been written but not run (2026-10-06). Row matching between
  generation and update is by (prompt ids, completion ids); duplicate rows in
  one batch map to the first.
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
