# martingale doctor

- Of the off-policy signal in stale tokens, 96% is staleness and 4% is engine-vs-trainer mismatch (lag-0 floor: mean |log r| = 1.02e-02).
- Staleness dominates: lower num_iterations / the async level, or use a correction that is unbiased at the sequence level (see martingale.exact).
- The engine floor is large (mean |log r| = 1.02e-02) but the median ratio is 1.0000 and no confident token disagrees: the mismatch sits in low-probability tokens, the bf16-vs-fp32 tail effect, worse at temperature 1 and as the policy sharpens. FP16, a bit-exact engine, or masking tail ratios (MIS) shrinks it.

sequences: 64 (mixed-revision: 0, unscored: 0), scored tokens: 502

Engine-mismatch floor (lag 0): mean |log r| = 1.016e-02, max |log r| = 2.480e-01, ESS/n = 0.998

| lag | tokens | seqs (mixed) | mean log r | mean abs log r | max abs log r | ratio p95 | ESS/n | clipped@0.1 | clipped@0.2 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 124 | 32 (0) | -2.630e-03 | 1.016e-02 | 2.480e-01 | 1.0032 | 0.998 | 0.032 | 0.024 |
| 1 | 127 | 32 (0) | -6.336e-02 | 9.747e-02 | 4.532e+00 | 1.0381 | 0.965 | 0.047 | 0.047 |
| 2 | 124 | 32 (0) | -3.496e-01 | 3.823e-01 | 1.068e+01 | 1.0446 | 0.913 | 0.145 | 0.121 |
| 3 | 127 | 32 (0) | -1.918e-01 | 2.416e-01 | 8.900e+00 | 1.0521 | 0.923 | 0.236 | 0.110 |

Exact values (reduced fractions) are in the JSON report; the table is float64 and informational.
