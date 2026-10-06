# martingale doctor

- Of the off-policy signal in stale tokens, 29% is staleness and 71% is engine-vs-trainer mismatch (lag-0 floor: mean |log r| = 4.53e-01).
- The engine floor is large (mean |log r| = 4.53e-01) but the median ratio is 1.0000: the mismatch sits in a few low-probability tokens, the known bf16-vs-fp32 tail effect, worse at temperature 1 and as the policy sharpens. Not a stale server (that would move the median). FP16, a bit-exact engine, or masking tail ratios (MIS) shrinks it.

sequences: 64 (mixed-revision: 0, unscored: 0), scored tokens: 480

Engine-mismatch floor (lag 0): mean |log r| = 4.534e-01, max |log r| = 9.257e+00, ESS/n = 0.398

| lag | tokens | seqs (mixed) | mean log r | mean abs log r | max abs log r | ratio p95 | ESS/n | clipped@0.1 | clipped@0.2 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 120 | 32 (0) | -4.072e-01 | 4.534e-01 | 9.257e+00 | 1.0007 | 0.398 | 0.075 | 0.075 |
| 1 | 120 | 32 (0) | -4.977e-01 | 5.020e-01 | 9.742e+00 | 1.0093 | 0.932 | 0.092 | 0.092 |
| 2 | 120 | 32 (0) | -6.358e-01 | 6.383e-01 | 1.308e+01 | 1.0093 | 0.919 | 0.092 | 0.092 |
| 3 | 120 | 32 (0) | -7.718e-01 | 7.752e-01 | 1.179e+01 | 1.0026 | 0.904 | 0.100 | 0.100 |

Exact values (reduced fractions) are in the JSON report; the table is float64 and informational.
