# martingale doctor

- Of the off-policy signal in stale tokens, 0% is staleness and 100% is engine-vs-trainer mismatch (lag-0 floor: mean |log r| = 6.11e-05).

sequences: 64 (mixed-revision: 0, unscored: 0), scored tokens: 232

Engine-mismatch floor (lag 0): mean |log r| = 6.112e-05, max |log r| = 1.053e-03, ESS/n = 1.000

| lag | tokens | seqs (mixed) | mean log r | mean abs log r | max abs log r | ratio p95 | ESS/n | clipped@0.1 | clipped@0.2 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 112 | 32 (0) | +5.487e-05 | 6.112e-05 | 1.053e-03 | 1.0005 | 1.000 | 0.000 | 0.000 |
| 1 | 120 | 32 (0) | +1.879e-05 | 2.833e-05 | 1.053e-03 | 1.0000 | 1.000 | 0.000 | 0.000 |

Exact values (reduced fractions) are in the JSON report; the table is float64 and informational.
