# martingale doctor

- Of the off-policy signal in stale tokens, 10% is staleness and 90% is engine-vs-trainer mismatch (lag-0 floor: mean |log r| = 4.53e-01).
- The engine floor is large (>1e-2 mean |log r|): the inference engine and the trainer disagree even on fresh tokens. Check dtype (bf16 vs fp32 lm_head), sampler settings, and whether the server's weights were actually updated; FP16 or a bit-exact engine shrinks this.

sequences: 64 (mixed-revision: 0, unscored: 0), scored tokens: 240

Engine-mismatch floor (lag 0): mean |log r| = 4.534e-01, max |log r| = 9.257e+00, ESS/n = 0.398

| lag | tokens | seqs (mixed) | mean log r | mean abs log r | max abs log r | ratio p95 | ESS/n | clipped@0.1 | clipped@0.2 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 120 | 32 (0) | -4.072e-01 | 4.534e-01 | 9.257e+00 | 1.0007 | 0.398 | 0.075 | 0.075 |
| 1 | 120 | 32 (0) | -4.977e-01 | 5.020e-01 | 9.742e+00 | 1.0093 | 0.932 | 0.092 | 0.092 |

Exact values (reduced fractions) are in the JSON report; the table is float64 and informational.
