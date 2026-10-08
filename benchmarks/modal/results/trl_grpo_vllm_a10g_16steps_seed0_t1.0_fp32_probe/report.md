# martingale doctor

- Of the off-policy signal in stale tokens, 29% is staleness and 71% is engine-vs-trainer mismatch (lag-0 floor: mean |log r| = 4.53e-01).
- 6 lag-0 tokens the engine was >=50% sure of are scored >2 nats lower by the trainer. bf16 rounding cannot do that: the engine generated from DIFFERENT WEIGHTS (or a different input) than the trainer scored with. Check that the weight sync into the inference engine actually applies (compare the engine's log-probs against a reference forward of the trainer's checkpoint: `martingale recompute`).

sequences: 64 (mixed-revision: 0, unscored: 0), scored tokens: 480

Engine-mismatch floor (lag 0): mean |log r| = 4.534e-01, max |log r| = 9.257e+00, ESS/n = 0.398

| lag | tokens | seqs (mixed) | mean log r | mean abs log r | max abs log r | ratio p95 | ESS/n | clipped@0.1 | clipped@0.2 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 120 | 32 (0) | -4.072e-01 | 4.534e-01 | 9.257e+00 | 1.0007 | 0.398 | 0.075 | 0.075 |
| 1 | 120 | 32 (0) | -4.977e-01 | 5.020e-01 | 9.742e+00 | 1.0093 | 0.932 | 0.092 | 0.092 |
| 2 | 120 | 32 (0) | -6.358e-01 | 6.383e-01 | 1.308e+01 | 1.0093 | 0.919 | 0.092 | 0.092 |
| 3 | 120 | 32 (0) | -7.718e-01 | 7.752e-01 | 1.179e+01 | 1.0026 | 0.904 | 0.100 | 0.100 |

Exact values (reduced fractions) are in the JSON report; the table is float64 and informational.
