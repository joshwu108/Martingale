# martingale

**A doctor for RL training runs.** Attach it to your GRPO/PPO trainer and it tells
you, per step, how much of your off-policy signal is **staleness** (rollouts
generated under old weights) and how much is **engine-versus-trainer mismatch**
(vLLM disagreeing with your trainer on the same tokens), which lags are wasted,
and what to change. It works from a per-token record that no framework keeps
today: which weights produced each token, at what probability, and what the
trainer later thought.

[![ci](https://github.com/joshwu108/Martingale/actions/workflows/ci.yml/badge.svg)](https://github.com/joshwu108/Martingale/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

> **Status (2026-10-06).** Record, diagnosis, TRL `GRPOTrainer` integration and an
> exact estimator bench are implemented and tested (`make check`: ruff, mypy, 394
> tests). The TRL integration has been exercised against a fake trainer only; the
> first real vLLM run (`benchmarks/modal/trl_grpo_vllm.py`) is written and not yet
> executed, so every number below is synthetic. Claims map to artifacts in
> [`paper/claim_evidence.md`](paper/claim_evidence.md); limits in
> [`docs/nonclaims.md`](docs/nonclaims.md).

## Sixty seconds, no GPU

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run martingale demo
```

The demo records a synthetic run and prints the diagnosis:

```
# martingale doctor

- Of the off-policy signal in stale tokens, 100% is staleness and 0% is engine-vs-trainer
  mismatch (lag-0 floor: mean |log r| = 1.05e-06).
- Staleness dominates: lower num_iterations / the async level, or use a correction that is
  unbiased at the sequence level (see martingale.exact).
- 8 sequences span a weight update (in-flight sync); their tokens carry two revisions and
  are bucketed by their own lag.

| lag | tokens | seqs (mixed) | mean log r | mean abs log r | max abs log r | ratio p95 | ESS/n | clipped@0.1 | clipped@0.2 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 104 | 14 (2) | +1.052e-07 | 1.049e-06 | 2.027e-06 | 1.0000 | 1.000 | 0.000 | 0.000 |
| 2 | 100 | 15 (5) | -1.971e-02 | 1.541e-01 | 2.944e-01 | 1.2936 | 0.971 | 0.690 | 0.310 |
| 3 | 96 | 15 (6) | -6.704e-02 | 2.029e-01 | 4.483e-01 | 1.4968 | 0.944 | 0.708 | 0.510 |
| 4 | 84 | 12 (3) | +4.371e-02 | 3.248e-01 | 5.962e-01 | 1.7080 | 0.896 | 0.857 | 0.762 |
```

Lag is the number of optimizer steps between the weights that generated a token
and the weights that scored it. Lag-0 tokens were scored by the weights that
produced them, so their disagreement is pure engine mismatch: that is the floor,
and everything above it is staleness. Log-ratios are exact differences of the
recorded float bits; ratios, effective sample size and clipped fractions are
float64 and labelled informational in the JSON.

## The problem

Every async RL framework (verl, TRL, PipelineRL, AReaL, SkyRL, slime, prime-rl)
has a staleness knob and an importance-sampling correction, and each logs an
aggregate ratio statistic. When ratios blow up at step 1200 none of them can tell
you whether the rollouts were stale, the inference engine disagreed with the
trainer, or a weight sync silently failed. The last year of papers (TIS/MIS, the
FP16 fix, VeXact, CISPO, GSPO, IcePop, A-3PO, staleness scaling laws, Missing Old
Logits) is about exactly this, and the tooling is still scripts.

## Use it with TRL

```python
from trl import GRPOConfig
from martingale.integrations.trl import MartingaleRecorder, MartingaleGRPOTrainer

flight = MartingaleRecorder("./martingale_ws", tokenizer=tokenizer)
trainer = MartingaleGRPOTrainer(model=model, args=GRPOConfig(...), train_dataset=ds,
                                reward_funcs=reward_fn, processing_class=tokenizer,
                                flight_recorder=flight)
trainer.train()
print("ledger head:", flight.head())     # one line for your run log
```

```bash
martingale doctor --dir ./martingale_ws          # the diagnosis above, from your run
martingale doctor --dir ./martingale_ws --json out.json --markdown out.md
```

The trainer also runs the doctor **live**: after every optimizer step it logs
`martingale/staleness_share`, `martingale/floor_mean_abs_log_ratio`,
`martingale/ess_fraction_lag{k}` and `martingale/alarms` into TRL's metrics (so
Weights & Biases shows them), and raises on an error-level alarm unless you pass
`halt_on_error=False`. Alarms, with thresholds in `AlarmConfig`:

| Alarm | What it means | Level |
|---|---|---|
| `weights_unchanged` | a generation ran under the same trainer weights as the previous one although optimizer steps happened: frozen model, lr 0, broken optimizer (exact, from the record) | error |
| `stale_server` | the lag-0 engine floor jumped 10x right after a new generation: tokens the trainer thinks are fresh behave as stale; the inference server is likely serving old weights (a proxy, the record holds the trainer's digest, not the server's) | error |
| `negative_lag` | tokens scored by an older step than generated them: resume mislabel or server ahead of trainer | error |
| `unmatched` | more than 1% of loss rows could not be matched to a recorded sequence | error |
| `floor_jump` | lag-0 floor 3x its trailing median: engine config, dtype or server settings changed | warn |
| `ess_collapse` | effective sample size under 30% (error under 10%) in a lag bucket holding over 10% of the step's tokens | warn / error |
| `span` | over 5% of new sequences span more than two weight revisions | warn |

Each checkpoint gets a `martingale_head.txt` so a resumed run anchors on the right head.

Per batch row it records the unpadded prompt, every completion token, the
behavior log-prob bits (vLLM's sampling log-probs when present, else TRL's
old log-probs, else a no-grad pass; which one is bound into the revision), a
digest of the generating weights plus tokenizer and sampler config, and TRL's
advantage. At update time it records the trainer's log-prob per token under the
weights it trained with. Tested against TRL 1.13.0; `integrations/verl.py` and
`integrations/lightning.py` publish revisions only, a verl adapter is next.

## Test your own correction (exact, no floats)

```python
from fractions import Fraction
from martingale.exact import check_unbiased, defendants

report = check_unbiased(defendants.token_ppo_clip(eps=Fraction(1, 5)), n_configs=20, lags=(1, 4, 8))
report.all_unbiased                 # False
report.certificates[-1].bias        # exact rational bias vector, (state, action) -> Fraction
report = check_unbiased(my_estimator, name="mine")   # any (traj, target, behavior) -> {(s, a): Fraction}
```

Exhaustive enumeration over small rational MDPs, so a certificate is an identity
or an exact inequality. Built-ins: `seq_*` (sequence ratio) and `token_*` (per-token
ratio, what PPO / TIS / MIS / CISPO implementations do). From
`results/estimator_bench_report.json`, cells are "configs certified unbiased"
over 20 MDPs:

| Defendant | lag 0 | lag 1 | lag 2 | lag 4 | lag 8 |
|---|---|---|---|---|---|
| unweighted (ignore staleness) | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| seq_is (sequence-level IS) | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| token_is (per-token ratio only) | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| seq_truncated_is(cap=2) | 20/20 | 20/20 | 20/20 | 16/20 | 7/20 |
| token_truncated_is(cap=2) | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| seq_masked_is(1/2, 2) | 20/20 | 13/20 | 7/20 | 3/20 | 2/20 |
| token_masked_is(1/2, 2) | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| seq_ppo_clip(eps=1/5) | 20/20 | 0/20 | 2/20 | 0/20 | 0/20 |
| token_ppo_clip(eps=1/5) | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| seq_cispo(eps=1/5) | 20/20 | 0/20 | 2/20 | 0/20 | 0/20 |
| token_cispo(eps=1/5) | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 |

Every per-token scheme with a trajectory-level advantage is biased from the first
stale step, clipped or not. Torch reference implementations checked against these
certificates are in `martingale.corrections`. Scope: tiny tabular MDPs; nothing
transfers to neural policies by itself.

## Why you can trust the numbers

The record is append-only and hash-chained, and `martingale verify --dir ws --head <head>`
runs an independent checker (`checker/`, imports nothing from `martingale`) that
re-derives every digest. 90 of 90 single-fault forgeries are rejected when the
head printed by the trainer is supplied (`campaigns/mutation_tokens.py`). The
checker verifies binding, not the draw: an engine does not expose a keyed draw,
so "this token was sampled from that distribution" is not a claim (non-claims).
Underneath is the exact-arithmetic research core the project started as (rational
MDPs, exact expected gradients, keyed draws, a TLA+ model of pin-before-draw with
SIGKILL crash cuts); its five theses and their reports are in
[`docs/design.md`](docs/design.md) and `results/`.

## Non-claims (short form; full list in `docs/nonclaims.md`)

- No real-run numbers yet; the TRL integration is tested against a fake trainer.
- The lag-0 floor includes whatever the engine's log-prob mode is (vLLM
  processed vs raw log-probs are a config property not yet captured).
- The record binds what the engine reported; it cannot verify the draw.
- Exact results hold for finite tabular rational MDPs and the rational simplex
  parameterization only. Single host, CPU evidence; no performance claims.
- Diagnosis thresholds (floor > 1e-2, ESS/n < 0.5, staleness share > 50%) are
  heuristics stated in the code, not findings.

## Layout

```
src/martingale/
  record/          revisions, token/sequence/score records, TokenLedger, Recorder
  diagnostics/     decompose(), attribute(), diagnosis(), markdown report
  exact/           check_unbiased(), random MDPs, 11 built-in defendants
  corrections.py   torch TIS / MIS / PPO-clip / CISPO with exact shadow checks
  integrations/    trl.py (GRPOTrainer flight recorder), verl.py, lightning.py, jax_utils.py
  prod/            reference AsyncActor and RevisionPublisher over the record
  cli.py, demo.py  martingale demo | doctor | verify | init | status | serve
  rational, mdp, policy, draw, estimators, revision, ledger, pipeline   the exact core
checker/           verify.py (exact ledger), verify_tokens.py (token record)
campaigns/         identity, staleness, boundary, float_baselines, mutation, mutation_tokens,
                   interleave, estimator_bench, e2e_demo
results/           committed evidence (JSON reports)
spec/              TLA+ models, .cfg files, check.sh (run in CI)
docs/              design.md, preregistration.md, nonclaims.md
paper/             claim_evidence.md
benchmarks/modal/  trl_grpo_vllm.py: the first real run (not yet executed)
```

`martingale serve` is the old Observatory dashboard over the exact ledger; it
will be rebuilt as the rollout inspector over the token record.

## Contributing

See `CONTRIBUTING.md`. No floats on decision paths; `checker/` imports nothing
from `martingale`; never delete a failing test or a surviving mutant.
