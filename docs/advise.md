# `martingale doctor --advise`

A suggestion with the numbers behind it, read from the per-lag table. Martingale does not
apply it, tune anything, or watch whether you followed it.

```bash
martingale doctor --dir ./martingale_ws --advise --lr 5e-6 --updates-per-pass 2 [--target-interval 8] [--ess-min 0.5]
```

Python: `advise_decomposition(decompose(ledger), ...)` for a finished record,
`advise(history)` over the `StepDiagnosis` list the live monitor produces (the last cumulative
table, plus each step's own table so one bad step is not averaged away). Both are pure and
call `advise_table(by_lag, cfg, ...)`.

## What it computes

**Sync interval.** S is the number of optimizer updates one generation feeds, so its tokens
are scored at lags 0..S-1. The advice is the largest S whose lags all keep ESS/n >= `ess_min`
in every bucket holding *over* 10% of the tokens (a bucket at exactly 10% does not count;
the share is an exact Fraction). The first qualifying lag that fails sets S to that lag.

| status | meaning |
|---|---|
| `within_observed` | every qualifying lag passes; S = largest qualifying lag + 1. Nothing is said about lags the run never produced. |
| `reduce` | lag F fails; S = F. |
| `engine_floor` | lag 0 fails: engine and trainer disagree before any staleness. No staleness knob fixes that. |
| `withheld` | the lag-0 floor is withheld (temperature scaling differs, `comparability.py`): ratios at every lag compare two different functions of the logits. No advice. |
| `no_data` | no scored tokens at lag >= 0, or no bucket qualifies. |

Only fresh rows are read when the record has provenance: the sync knobs set a fresh row's
lag; a replayed row's lag is the buffer's age. Negative lags are excluded and counted.

**Knobs.** With `--updates-per-pass U` (TRL `steps_per_generation`; verl
`train_batch_size / ppo_mini_batch_size`): a synchronous recipe with K passes runs lags
0..U·K-1, so `num_iterations` / `ppo_epochs` <= S // U; one pass with async level a (a extra
optimizer steps between generation and the first update) runs lags 0..U+a-1, so a <= S - U.
Frameworks count async differently; check yours against this definition.

**Learning-rate ceiling (a heuristic).** Song et al., *Scaling Laws for Collapse in
Asynchronous GRPO*, [arXiv:2607.01083](https://arxiv.org/abs/2607.01083), find that the largest
stable learning rate scales roughly as 1/S and bound the staleness-induced update bias by
O(S·η) under local regularity conditions. When lag F fails, the advice holds S·η at the
budget that passed, S_pass·η, and prints η·S_pass/S at the interval the run used, next to
η·(F+1)/S, the same law at the budget that failed. That step assumes the failure is staleness
bias, which a lower lr shrinks; a failure from the engine floor or from sampling noise does
not respond to lr, and the record cannot tell which. With no failing lag the record supports
the run's own lr and nothing higher. `--target-interval` (needs `--lr`) prints η·S_pass/S_target,
marked "not measured" unless the target is S_pass. Intervals, shares and learning rates are
exact (an lr given as a decimal string is a Fraction); ESS/n is the record's declared float.

## Calibration on the committed records

`uv run python benchmarks/modal/advise_calibration.py` prints both tables;
`tests/test_advise_calibration.py` pins the default row. Five records, four runs:
`t1.0` and `t1.0_fp32_probe` share the ledger head `fc9ee53f8526` (the same chain stored twice).
All TRL runs: Qwen2.5-0.5B, 16 steps, `num_iterations=2`, `steps_per_generation=2` (S = 4),
lr 5e-6 (from `trl_grpo_vllm.py`; the TRL configs do not record it); verl: 8 steps, `ppo_epochs=2`, 8/4 mini-batches (S = 4), lr 1e-6.

Fresh rows, lag: tokens (share) ESS/n, max |log r| in nats:

| record | lags |
|---|---|
| t1.0 | 0: 120 (25%) 0.398, 9.3; 1: 120 (25%) 0.932, 9.7; 2: 120 (25%) 0.919, 13; 3: 120 (25%) 0.904, 12 |
| t1.0_fp32_probe | same chain as t1.0 |
| t1.0_probe | 0: 124 (25%) 0.998, 0.25; 1: 127 (25%) 0.965, 4.5; 2: 124 (25%) 0.913, 11; 3: 127 (25%) 0.923, 8.9 |
| t1.0_replay1of4_replay | 0: 104 (26%) 0.999, 0.25; 1: 93 (24%) 0.961, 8.8; 2: 104 (26%) 0.351, 11; 3: 93 (24%) 0.956, 11 |
| verl t4 | 0: 910 (26%) 1.000, 0.024; 1: 930 (26%) 0.984, 4.9; 2: 912 (26%) 0.985, 11; 3: 789 (22%) 0.983, 13 |

Advice by threshold (S = passing/observed, U = 2 everywhere):

| record | ESS >= 0.3 | **ESS >= 0.5 (default)** | ESS >= 0.9 | ESS >= 0.95 |
|---|---|---|---|---|
| t1.0 | S=4/4, iters<=2, lr<=5e-6 | engine_floor | engine_floor | engine_floor |
| t1.0_fp32_probe | S=4/4, iters<=2, lr<=5e-6 | engine_floor | engine_floor | engine_floor |
| t1.0_probe | S=4/4, iters<=2, lr<=5e-6 | S=4/4, iters<=2, lr<=5e-6 | S=4/4, iters<=2, lr<=5e-6 | reduce: S=2/4, iters<=1, lr<=2.5e-6 |
| t1.0_replay1of4_replay | S=4/4, iters<=2, lr<=5e-6 | reduce: S=2/4, iters<=1, lr<=2.5e-6 | reduce: S=2/4, iters<=1, lr<=2.5e-6 | reduce: S=2/4, iters<=1, lr<=2.5e-6 |
| verl t4 | S=4/4, iters<=2, lr<=1e-6 | S=4/4, iters<=2, lr<=1e-6 | S=4/4, iters<=2, lr<=1e-6 | S=4/4, iters<=2, lr<=1e-6 |

Why 0.5. Sorted, the qualifying buckets of all five records split into two groups with a gap
between them: 0.904 and up (lowest: lag 3 of `t1.0`; the two runs with no low bucket,
`t1.0_probe` and verl, bottom out at 0.913), and two buckets at 0.351 and 0.398. The split
comes from the gap in the values, not from a label given beforehand. Only one of the two low
buckets has a known cause: lag 0 of `t1.0`, the fp32-trainer versus bf16-engine precision gap
(`docs/findings/2026-10-06-trl-colocate-sync.md`). Lag 2 of the replay run has none. Any
threshold in (0.398, 0.904] splits the groups the same way; 0.5 is round and leaves margin on
both sides. The live alarm's 0.3 (`ess_collapse`) misses both, which is
right for an alarm that interrupts a run and too loose for planning headroom.

What the default would have said:

- **t1.0 / fp32_probe:** "lag 0 already has ESS/n 0.398 over 120 tokens: fix the floor
  first." Correct for that run: the cause was trainer precision, not staleness.
- **probe:** S = 4 is supported, nothing measured beyond it; no lag failed, so the run's
  lr of 5e-6 stands and no headroom is shown.
- **replay:** lag 2 at 0.351 over 104 tokens: S = 2 (`num_iterations` 1 with
  `steps_per_generation` 2), or, if the failure is staleness bias, keep S = 4 at lr <= 2.5e-6
  by the scaling law (the budget that failed gives 3.75e-6). Lags 1 and 3 of the same run pass
  at 0.96, and lag 2 of the probe run, same recipe, passes at 0.913. That pattern fits sampling
  noise in a 104-token ESS, but it is one bucket and the noise reading is not tested. The
  `reduce` status applies the threshold; it does not say why the bucket is low.
- **verl:** S = 4 supported at every threshold tried; the run's lr of 1e-6 stands.

None of these runs varied S or η, so no record tests the scaling law itself; see
`docs/nonclaims.md`, "Knob advisor".
