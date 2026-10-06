# martingale

**The flight recorder for asynchronous RL.** Every token carries the exact
behavior log-prob bits and a content-addressed digest of the policy that
produced it; an independent checker replays the record; a report separates
"off-policy because stale" from "off-policy because the engine disagrees with
the trainer"; and an exact-arithmetic bench machine-checks whether your
off-policy correction is unbiased.

[![ci](https://github.com/joshwu108/Martingale/actions/workflows/ci.yml/badge.svg)](https://github.com/joshwu108/Martingale/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

> **Status (2026-10-06).** Research core, token record, independent checkers,
> diagnostics, exact bench and the TRL `GRPOTrainer` integration are implemented
> and tested (390 tests). The TRL integration has been exercised against a fake
> trainer only; the real vLLM run (`benchmarks/modal/trl_grpo_vllm.py`) is
> written but has not been executed. Every claim below has a row in
> [`paper/claim_evidence.md`](paper/claim_evidence.md); scope limits are in
> [`docs/nonclaims.md`](docs/nonclaims.md).

## Sixty seconds, no GPU

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run martingale demo
```

The demo records a synthetic run (four policy revisions, mixed-revision
sequences, trainer-side scores), prints the ledger head, verifies the export
with the independent checker anchored on that head, prints the
staleness-versus-mismatch table, and certifies two estimators with the exact
bench. `make check` runs lint, type check, checker import isolation and tests.

## Why

Async RL frameworks (verl, TRL, PipelineRL, AReaL, SkyRL, slime, prime-rl, ...)
each have a staleness knob and an importance-sampling correction, and each logs
aggregate ratio statistics. None of them records, per token, *which* weights
produced it at *what* probability in a form that survives the trainer and can be
checked by someone else. So when ratios blow up at step 1200 you cannot tell
whether the rollouts were stale, the inference engine disagreed with the
trainer, or a bug mislabelled a batch. And the "Missing Old Logits" problem is
an entire paper about behavior log-probs that were never saved.

Martingale saves them, binds them, and lets you check them.

## Use it with TRL

```python
from trl import GRPOConfig
from martingale.integrations.trl import MartingaleRecorder, MartingaleGRPOTrainer

flight = MartingaleRecorder("./martingale_ws", tokenizer=tokenizer)
trainer = MartingaleGRPOTrainer(model=model, args=GRPOConfig(...), train_dataset=ds,
                                reward_funcs=reward_fn, processing_class=tokenizer,
                                flight_recorder=flight)
trainer.train()
print("ledger head:", flight.head())     # put this in your run log; the checker anchors on it
```

```bash
martingale verify --dir ./martingale_ws --head <head>   # independent checker, anchored on the published head
martingale report --dir ./martingale_ws      # staleness-vs-mismatch decomposition
```

What gets recorded, per batch row: the unpadded prompt, every completion
token, the behavior log-prob bits (vLLM's sampling log-probs when present,
else TRL's old log-probs, else a no-grad pass; which one is bound into the
revision), the revision digest (weights + tokenizer + sampler config + learner
step), and TRL's advantage. At update time: the trainer's log-prob per token
under the revision it trained with. Tested against TRL 1.13.0; other
frameworks: `integrations/verl.py` and `integrations/lightning.py` publish
revisions only.

## Read the report

`martingale report` buckets every scored token by lag (learner steps between
the weights that generated it and the weights that scored it). Lag-0 tokens
were scored by the weights that generated them, so their log-ratio is pure
engine-versus-trainer mismatch: the floor. Everything above it is staleness.

```
| lag | tokens | seqs (mixed) | mean log r | mean abs log r | max abs log r | ratio p95 | ESS/n | clipped@0.1 | clipped@0.2 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 96 | 12 (0) | -1.9e-07 | 1.8e-06 | 5.2e-06 | 1.0000 | 1.000 | 0.000 | 0.000 |
| 2 | 112 | 15 (3) | +1.1e-02 | 1.3e-01 | 4.5e-01 | 1.2432 | 0.976 | 0.464 | 0.223 |
| 3 | 100 | 15 (3) | -2.8e-02 | 1.9e-01 | 6.2e-01 | 1.3581 | 0.950 | 0.560 | 0.370 |
| 4 | 76 | 12 (0) | +4.0e-02 | 2.6e-01 | 8.1e-01 | 1.5173 | 0.916 | 0.671 | 0.447 |
```

(The demo's synthetic numbers. Log-ratios are exact differences of recorded
bits; ratios, ESS and clipped fractions are float64 and labelled
informational in the JSON.)

## Verify a record

```bash
martingale verify --dir ./ws                                 # both checkers, if both ledgers exist
python -m checker.verify_tokens ./export --expected-head <head>   # the token checker on an export
```

`checker/` imports nothing from `martingale` (CI-enforced) and re-implements
every digest. For the token record it verifies binding: digests, chains,
revision existence, contiguous positions, finite bits, one tokenizer and
sampler per sequence, and the anchored ledger head. It does **not** verify that
a token was actually drawn from the recorded distribution; engines do not
expose a keyed draw (see non-claims). `campaigns/mutation_tokens.py`: 90/90
single-fault forgeries rejected with the anchored head, 82/90 without; the
survivors without an anchor are re-signed chains, trailing deletions and a
dropped unreferenced revision, which is what the anchor is for. The checker
refuses to run from the command line without either `--expected-head` or an
explicit `--unanchored`.

## Test your own estimator (exact, no floats)

```python
from fractions import Fraction
from martingale.exact import check_unbiased, defendants

report = check_unbiased(defendants.token_ppo_clip(eps=Fraction(1, 5)), n_configs=20, lags=(1, 4, 8))
report.all_unbiased                 # False
report.certificates[-1].bias        # exact rational bias vector, (state, action) -> Fraction
report.certificates[-1].rel_sq_bias # ||bias||^2 / ||grad||^2, exact

report = check_unbiased(my_estimator, name="mine")   # any (traj, target, behavior) -> {(s, a): Fraction}
```

The expectation is an exhaustive enumeration over all trajectories of small
rational MDPs, so a certificate is an identity or an exact inequality, never a
statistical estimate. Built-ins come in two families: `seq_*` weight the whole
trajectory by the sequence ratio, `token_*` weight each token by its own ratio
(what PPO / TIS / MIS / CISPO implementations do). From
`results/estimator_bench_report.json` (20 seeded MDPs x lags {0,1,2,4,8}; cells
are "configs certified exactly unbiased"):

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

Sequence-level truncation is exactly unbiased until some trajectory ratio
crosses the cap; every per-token scheme with a trajectory-level advantage is
biased from the first stale step, clipped or not. Torch reference
implementations of the same corrections, tested against these certificates and
against a float32 clip-boundary flip harness, are in `martingale.corrections`.

## The research core (T1–T5)

Underneath is the exact-arithmetic instrument the project started as: rational
MDPs, rational simplex policies, keyed BLAKE2b draws, exact expected gradients
by exhaustive enumeration, a hash-chained ledger whose draws the checker
re-derives, a multi-process actor/learner pipeline with SIGKILL crash cuts, and
a TLA+ model of the pin-before-draw protocol.

| Thesis | Description | Status | Evidence |
|--------|-------------|--------|----------|
| T1 | Exact ledger with hash-chained attestations | **ALIVE** | `results/mutation_report.json`: 72/72 forgeries rejected |
| T2 | IS-REINFORCE unbiasedness identity (machine-checked) | **ALIVE** | `results/identity_report.json`: 40/40 exact Fraction equalities |
| T3 | PPO staleness bias grows monotonically with lag | **ALIVE** | `results/staleness_report.json`: 4/4 cells monotone |
| T4 | Float clip-boundary flips exist in float32/float64 | **MIXED** | `results/boundary_report.json`: float32/eps=1/10: 183 flips; float64/eps=1/5: 21 flips; float64/eps=1/10 and float32/eps=1/5: 0 flips (kill rule fired for those two variants at N=1000; the preregistered 10^5-candidate run has not been done) |
| T5 | Async protocol safety (TLA+ + crash cuts) | **ALIVE** | `results/interleave_report.json`: 9/9 scenarios; `spec/*.cfg` model-checked in CI |

Research context: A-3PO ([2512.06547](https://arxiv.org/abs/2512.06547)),
staleness-LR scaling laws ([2607.01083](https://arxiv.org/abs/2607.01083)),
Missing Old Logits ([2605.12070](https://arxiv.org/abs/2605.12070)), TIS/MIS
(Yao et al. 2025), CISPO (MiniMax-M1), IcePop (Ant Ling). Those estimate or
correct staleness bias; this records what actually happened and certifies the
corrections exactly.

```bash
uv run python -m campaigns.identity          # T2: 40 machine-checked exact equalities
uv run python -m campaigns.e2e_demo          # exact pipeline end to end, SIGKILL and recovery
uv run python -m campaigns.mutation          # 72 ledger forgeries, 100% rejection
uv run python -m campaigns.mutation_tokens   # 73 token-record forgeries
uv run python -m campaigns.estimator_bench   # the certificate table above
uv run python -m campaigns.interleave        # T5 interleavings + SIGKILL
```

## Non-claims (short form; full list in `docs/nonclaims.md`)

- Exact results hold for finite tabular rational MDPs (|S| ≤ 5, |A| ≤ 3, H ≤ 5)
  and the rational simplex parameterization, which is not a softmax. Nothing
  transfers to neural policies by itself.
- The token record binds what the engine reported; it cannot verify the draw.
  Tamper evidence for re-signed chains needs the ledger head published out of band.
- The TRL integration is tested against a fake trainer; no real-run numbers yet.
- BLAKE2b keyed draws and hash chains are tamper-evident, not a security boundary.
- Single host, CPU-only evidence; no distributed-systems or performance claims.

## Layout

```
src/martingale/
  record/          LLMRevision, TokenRecord, SequenceRecord, ScoreRecord, TokenLedger, Recorder
  diagnostics/     staleness-vs-mismatch decomposition, markdown report
  exact/           check_unbiased(), random MDPs, 11 built-in defendants
  corrections.py   torch reference TIS / MIS / PPO-clip / CISPO with exact shadow checks
  integrations/    trl.py (GRPOTrainer flight recorder), verl.py, lightning.py, jax_utils.py
  prod/            reference AsyncActor and RevisionPublisher over the record
  demo.py, cli.py  martingale demo | init | status | verify | report | serve
  rational, mdp, policy, draw, estimators, revision, ledger, pipeline   the exact core
checker/           verify.py (exact ledger), verify_tokens.py (token record); imports nothing from src/
campaigns/         identity, staleness, boundary, float_baselines, mutation, mutation_tokens,
                   interleave, estimator_bench, e2e_demo
results/           committed evidence (JSON reports)
spec/              RevisionPin.tla, RevisionPinWeakened.tla, .cfg files, check.sh
docs/              design.md (sections 1-6 exact core, 7 token record), preregistration.md, nonclaims.md
paper/             claim_evidence.md: every README claim mapped to its artifact
benchmarks/modal/  trl_grpo_vllm.py: the real-run script (not yet executed)
```

The Observatory dashboard (`martingale serve`) reads the exact ledger from
`campaigns/e2e_demo.py`; it is kept as an optional extra and does not read the
token record.

## Contributing

See `CONTRIBUTING.md`. The rules that matter: no floats on decision or
measurement paths; `checker/` imports nothing from `martingale`; never delete a
failing test, a surviving mutant, or an inconvenient exact-equality failure.
