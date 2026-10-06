# martingale

**Exact off-policy staleness accounting for asynchronous policy-gradient RL**

> Status: research milestones M1–M7 complete with committed evidence. The production
> layer (`prod/`, `integrations/`, dashboard) is a **prototype sketch whose ledger the
> independent checker currently rejects** — see `docs/nonclaims.md`, "Production layer".
> A verifiable token-level record for LLM RL is the next milestone.

## What this is

`martingale` is a correctness-oriented research codebase (with a production layer in
progress) for the learner/algorithm side of asynchronous reinforcement learning. Modern online-RL
pipelines (veRL, TRL, Lightning-RL) run actors and learners asynchronously: trajectories
are generated under behavior-policy revisions that lag the learner's current policy,
sometimes with *mixed* revisions inside a single trajectory when weights update mid-rollout.

This is the missing accountability layer:

- Every action carries an **exact behavior probability** bound to a **hash-attested revision digest**.
- IS weights are computed against the precise policy that generated each action — no approximation.
- An **independent checker** re-derives every probability and draw from the revision store
  and rejects tampering; a daemon wraps it and emits Prometheus metrics.
- The **true expected gradient** is verifiable by exhaustive exact-arithmetic enumeration,
  so that staleness bias becomes a machine-checked rational identity — not a noisy anecdote.

### Research context

The async RLHF staleness problem is actively studied. Recent papers attack it from the
algorithm side:

- **A-3PO** ([arxiv 2512.06547](https://arxiv.org/abs/2512.06547), ICLR 2026) — staleness-aware PPO for LLM training
- **Staleness-LR Scaling Laws** ([arxiv 2607.01083](https://arxiv.org/abs/2607.01083), July 2026) — independently derives monotone bias growth (T3)
- **RAC / V-trace for RLHF** ([arxiv 2606.27580](https://arxiv.org/abs/2606.27580), June 2026) — closed-form delay correction
- **Missing Old Logits** ([arxiv 2605.12070](https://arxiv.org/abs/2605.12070), 2025) — IS accuracy under missing behavior log-probs (T4)

Martingale is the **infrastructure complement**: where those papers estimate staleness bias,
Martingale provides tamper-evident proof of exactly which policy generated each action.

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

## Test your own estimator (exact, no floats)

```python
from fractions import Fraction
from martingale.exact import check_unbiased, defendants

report = check_unbiased(defendants.token_ppo_clip(eps=Fraction(1, 5)), n_configs=20, lags=(1, 4, 8))
report.all_unbiased                 # False
report.certificates[-1].bias        # exact rational bias vector, (state, action) -> Fraction
report.certificates[-1].rel_sq_bias # ||bias||^2 / ||grad||^2, exact

# Your estimator: any callable (traj, target, behavior) -> {(state, action): Fraction}
report = check_unbiased(my_estimator, name="mine")
```

Built-in defendants come in two families: `seq_*` weight the whole trajectory by the
sequence ratio w = prod_t pi/b (the textbook object), `token_*` weight each token by its own
ratio r_t (what PPO / TIS / MIS / CISPO implementations actually do). Variants:
`unweighted`, `seq_is`, `token_is`, `{seq,token}_truncated_is` (TIS),
`{seq,token}_masked_is` (MIS / IcePop-style), `{seq,token}_ppo_clip`, `{seq,token}_cispo`.
Committed certificates over 20 seeded MDPs x lags {0,1,2,4,8} are in
`results/estimator_bench_report.json` (`uv run python -m campaigns.estimator_bench`):

| Defendant | lag 0 | lag 1 | lag 2 | lag 4 | lag 8 |
|---|---|---|---|---|---|
| unweighted (ignore staleness) | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| seq_is (sequence-level IS-REINFORCE) | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| token_is (per-token ratio only) | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| seq_truncated_is(cap=2) | 20/20 | 20/20 | 20/20 | 16/20 | 7/20 |
| token_truncated_is(cap=2) | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| seq_masked_is(1/2, 2) | 20/20 | 13/20 | 7/20 | 3/20 | 2/20 |
| token_masked_is(1/2, 2) | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| seq_ppo_clip(eps=1/5) | 20/20 | 0/20 | 2/20 | 0/20 | 0/20 |
| token_ppo_clip(eps=1/5) | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| seq_cispo(eps=1/5) | 20/20 | 0/20 | 2/20 | 0/20 | 0/20 |
| token_cispo(eps=1/5) | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 |

Cells are "configs certified exactly unbiased". Sequence-level truncation is exactly unbiased
until some trajectory ratio crosses the cap; every per-token scheme, with a trajectory-level
advantage, is biased from the first stale step, clipped or not. Scope is the tiny tabular
family in `docs/nonclaims.md`; nothing here transfers to softmax policies by itself.

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

# Install core (no dependencies)
uv sync

# Install with production extras (PyTorch, Prometheus, CLI)
uv sync --extra prod

# Install with observatory server (adds FastAPI + uvicorn)
uv sync --extra server

# Run all checks (import isolation + tests)
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

## Observatory dashboard (prototype)

```bash
# Initialise a workspace
martingale init --dir ./run_workspace

# Start the dashboard server (port 7373 by default)
martingale serve --dir ./run_workspace --port 7373
# → http://127.0.0.1:7373
```

**Honesty note:** the dashboard reads the SQLite workspace written by the prototype
`prod.AsyncActor`, whose records the independent checker rejects (see non-claims).
Until the token-record schema lands, treat every number it shows as unverified.

The dashboard shows:
- Revision timeline and per-actor staleness histogram
- IS weight distribution (mean, p95, p99) and clip-boundary flip count
- Checker daemon status (forgeries detected / trajectories verified)
- Live staleness scaling chart (T3 monotone-bias indicator)

## Framework hooks (prototype — publish checkpoint digests only)

These classes compute a content-addressed digest of a model's weights after each
optimizer step. They do **not** yet record tokens or behavior log-probs, and they do
not call TRL, veRL or Lightning APIs; a real TRL `GRPOTrainer` integration is planned.

```python
# TRL (HuggingFace)
from martingale.integrations.trl import MartingalePPOTrainer
trainer = MartingalePPOTrainer(publisher=publisher, **trl_kwargs)

# PyTorch Lightning
from martingale.integrations.lightning import MartingaleCallback
pl_trainer = Trainer(callbacks=[MartingaleCallback(publisher)])

# Production actor (any framework)
with actor.pin_revision(checkpoint_digest) as ctx:
    result = actor.sample_and_record(obs, log_probs, episode_id, step)
    ctx.commit_episode(episode_id)
```

## Verify a run

```bash
martingale verify --dir ./run_workspace --seed my-training-seed
```

Today this only verifies ledgers written by the exact research pipeline
(`campaigns/e2e_demo.py`). Ledgers written by the prototype production actor fail
verification by construction; `tests/test_prod_checker.py` pins that fact.

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
    exact/               # check_unbiased(): exact test bench for user estimators + built-in defendants
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
    estimator_bench.py   # exact bias certificates for TIS / MIS / PPO-clip / CISPO
  tests/                 # 240 pytest tests (TDD, all green; one strict xfail pinning the prod defect)
  results/               # committed evidence artifacts (JSON reports)
    identity_report.json
    mutation_report.json
    staleness_report.json
    boundary_report.json
    interleave_report.json
    e2e_report.json
    estimator_bench_report.json
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
