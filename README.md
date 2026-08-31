# martingale

**Exact off-policy staleness accounting for asynchronous policy-gradient RL**

> Status: M1 (design) — in progress.  See thesis status table below.

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
rational identity or a machine-checked exact inequality.

## Thesis status

| Thesis | Description | Status |
|--------|-------------|--------|
| T1 | Exact ledger with hash-chained attestations | Pending M4 |
| T2 | IS-REINFORCE exactunbiasedness identity (machine-checked) | Pending M3 |
| T3 | Staleness scaling law (preregistered) | Pending M6 |
| T4 | Float clip-boundary flip frequency | Pending M6 |
| T5 | Async protocol safety (TLA+ + crash cuts) | Pending M5 |

## Non-claims

See `docs/nonclaims.md`.  Key: finite tabular MDPs only; no softmax/NN policies;
no deep-RL/RLHF-scale claims; BLAKE2b draw is not a security boundary;
single-host CPU only.

## Setup

```bash
# Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project
uv sync --all-extras

# Run all checks
make check
```

## Repository layout

```
martingale/
  pyproject.toml, uv.lock, Makefile, README.md, LICENSE
  docs/
    design.md            # parameterization decision, enumeration semantics, ledger schema
    preregistration.md   # frozen T3 + T4 protocols and kill rules
    nonclaims.md
  spec/
    RevisionPin.tla      # finite-state model of pin-before-draw protocol
    check.sh
  src/martingale/
    rational.py          # exact conversions, reduced-fraction serialization
    mdp.py               # rational MDPs + exact enumeration engine
    policy.py            # rational policy + exact parameter-gradients
    draw.py              # keyed BLAKE2b draws over rational simplexes
    estimators.py        # exact expected gradients (on-policy, IS, PPO, GRPO)
    revision.py          # content-addressed revision store
    ledger.py            # hash-chained trajectory attestations
    pipeline.py          # multi-process actor/learner with pin-before-draw
  checker/
    verify.py            # independent ledger verifier (imports nothing from src/)
  campaigns/
    identity.py          # T2 machine-checked certificates
    staleness.py         # T3 preregistered scaling campaign
    float_baselines.py   # T4 float defendants
    boundary.py          # T4 clip-boundary flip search
    mutation.py          # ledger-forgery campaign vs checker/
    interleave.py        # T5 scripted-interleaving + SIGKILL campaign
  tests/
  results/
```
