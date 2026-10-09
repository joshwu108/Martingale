# martingale doctor

- Of the off-policy signal in stale tokens, 100% is staleness and 0% is engine-vs-trainer mismatch (lag-0 floor: mean |log r| = 2.02e-04).
- Staleness dominates: lower num_iterations / the async level, or use a correction that is unbiased at the sequence level (see martingale.exact).

sequences: 512 (mixed-revision: 0, unscored: 0), scored tokens: 3541

Engine-mismatch floor (lag 0): mean |log r| = 2.017e-04, max |log r| = 2.428e-02, ESS/n = 1.000

| lag | tokens | seqs (mixed) | mean log r | mean abs log r | max abs log r | ratio p95 | ESS/n | clipped@0.1 | clipped@0.2 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 910 | 256 (0) | -1.783e-05 | 2.017e-04 | 2.428e-02 | 1.0000 | 1.000 | 0.000 | 0.000 |
| 1 | 930 | 266 (0) | -2.942e-02 | 3.812e-02 | 4.911e+00 | 1.0008 | 0.984 | 0.017 | 0.015 |
| 2 | 912 | 256 (0) | -6.153e-02 | 6.973e-02 | 1.080e+01 | 1.0015 | 0.985 | 0.029 | 0.019 |
| 3 | 789 | 224 (0) | -7.971e-02 | 8.729e-02 | 1.336e+01 | 1.0002 | 0.983 | 0.023 | 0.020 |

Exact values (reduced fractions) are in the JSON report; the table is float64 and informational.
