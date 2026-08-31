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
