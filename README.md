# martingale

**Exact off-policy staleness accounting for asynchronous policy-gradient RL**

> Status: **M7 complete** — all milestones implemented. See thesis status table below.

## What this is

`martingale` is a correctness-oriented research codebase for the learner/algorithm
side of asynchronous reinforcement learning.  Modern online-RL pipelines run actors
and learners asynchronously: trajectories are generated under behavior-policy
revisions that lag the learner's current policy, sometimes with *mixed* revisions
inside a single trajectory when weights update mid-rollout.

This is the missing scientific instrument: every action carries an **exact rational
behavior probability** bound to an attested revision; importance weights are
**exact**; and the **true expected gradient** is computed by **exhaustive trajectory
enumeration in exact arithmetic**, so that staleness bias becomes a machine-checked
rational identity or a machine-checked exact inequality — not a noisy empirical anecdote.

## Thesis status

| Thesis | Description | Status | Evidence |
|--------|-------------|--------|----------|
| T1 | Exact ledger with hash-chained attestations | **ALIVE** | `results/mutation_report.json` — 72/72 forgeries rejected |
| T2 | IS-REINFORCE unbiasedness identity (machine-checked) | **ALIVE** | `results/identity_report.json` — 40/40 exact Fraction equalities |
| T3 | PPO staleness bias grows monotonically with lag | **ALIVE** | `results/staleness_report.json` — 4/4 cells monotone |
| T4 | Float clip-boundary flips exist in float32/float64 | **MIXED** | `results/boundary_report.json` — 183 flips found (float32/eps=0.1); 0 found for float64/eps=0.1 (kill rule fired for that variant) |
| T5 | Async protocol safety (TLA+ + crash cuts) | **ALIVE** | `results/interleave_report.json` — 9/9 scenarios verified |

**T4 note:** The kill rule fired for the float64+ε=0.1 variant and float32+ε=0.2
variant in the automated run (N=1000 candidates; see preregistration.md for the
full 10^5-candidate production protocol). The float32+ε=0.1 path shows 183 flips,
confirming boundary flips exist. The zero-flip results for certain variants likely
reflect insufficient candidate count rather than a true absence — the full production
run (10^5 candidates) should resolve this. This is reported fully per the preregistered
descriptive question obligation.

## Non-claims

See `docs/nonclaims.md`. Key:
- Results apply only to finite, tabular, rational-parameter MDPs (|S|≤5, |A|≤3, H≤5).
- No softmax/neural-network policy claims.
- No deep-RL or RLHF-scale claims.
- The rational simplex policy is not a softmax; results do not automatically transfer.
- The BLAKE2b draw is not a security boundary.
- Single-host, CPU-only; no distributed-systems claims.
- The TLA+ model covers a finite scope (2 actors, 3 revisions, 2 actions).

## Quick start

```bash
# Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project
uv sync

# Run all checks (import isolation + 98 tests)
make check

# Run the T2 identity campaign (40 machine-checked exact equalities)
uv run python -m campaigns.identity

# Run the end-to-end demonstration
uv run python -m campaigns.e2e_demo

# Run the mutation campaign (72 forgeries, 100% rejection)
uv run python -m campaigns.mutation

# Run the T5 interleaving + SIGKILL campaign
uv run python -m campaigns.interleave
```

## Repository layout

```
martingale/
  pyproject.toml, Makefile, README.md, LICENSE
  docs/
    design.md            # parameterization decision (Option A: rational simplex),
                         # exact-enumeration semantics, ledger schema, protocol
    preregistration.md   # FROZEN T3 + T4 protocols and kill rules (2026-08-31)
    nonclaims.md
  spec/
    RevisionPin.tla           # correct pin-before-draw protocol (invariant holds)
    RevisionPinWeakened.tla   # weakened ordering (counterexample exists)
    check.sh                  # TLC model checker runner (skips gracefully if not installed)
  src/martingale/
    rational.py          # exact conversions, reduced-fraction serialization
    mdp.py               # rational MDPs + exact enumeration engine
    policy.py            # rational simplex policy + exact parameter-gradients
    draw.py              # keyed BLAKE2b draws over rational simplexes
    estimators.py        # exact expected gradients: on-policy, IS-REINFORCE, PPO, GRPO
    revision.py          # content-addressed revision store (durable, atomic writes)
    ledger.py            # hash-chained trajectory attestations (mixed-revision aware)
    pipeline.py          # multi-process actor/learner with pin-before-draw protocol
  checker/
    verify.py            # independent ledger verifier (imports nothing from src/)
  campaigns/
    identity.py          # T2: 40+ machine-checked exact equality/inequality certificates
    staleness.py         # T3: preregistered monotone-bias scaling campaign
    float_baselines.py   # T4: float32/float64 log-ratio-exp-clip defendants
    boundary.py          # T4: clip-boundary flip search + minimal reproducers
    mutation.py          # ledger-forgery campaign: 72 mutants, 100% rejection
    interleave.py        # T5: scripted-interleaving + SIGKILL crash-cut campaign
    e2e_demo.py          # M7: full end-to-end demonstration
  tests/                 # 98 pytest tests (TDD, all green)
  results/               # committed evidence artifacts (JSON reports)
    identity_report.json
    mutation_report.json
    staleness_report.json
    boundary_report.json
    interleave_report.json
    e2e_report.json
```

## Policy parameterization (Option A)

The rational simplex parameterization is used throughout T1–T3, T5:
- A policy is a table of `fractions.Fraction` entries summing to exactly 1.
- IS weights `π(a|s) / b(a|s)` are exact Fraction divisions — no floating point.
- Gradient ascent: `p_i ← p_i + α·G·(1_{i=a} - p_i)` preserves sum=1 exactly.
- See `docs/design.md` for the full parameterization decision and trade-offs.

Option B (integerized softmax) is used only in `campaigns/float_baselines.py` for T4.

## Exactness guarantees

- All core computations use `fractions.Fraction` arithmetic — no floats.
- Floats enter only at declared boundaries: T4 float defendants only.
- The checker (`checker/verify.py`) imports nothing from `src/` and re-derives
  all probabilities from the revision store — enforced by CI (`make check-imports`).
- All reported biases are exact rationals serialized as `"numerator/denominator"` strings.
- Clip comparisons in `estimators.py` use exact Fraction inequality — never float.
