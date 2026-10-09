# martingale doctor

- Of the off-policy signal in stale tokens, 99% is staleness and 1% is engine-vs-trainer mismatch (lag-0 floor: mean |log r| = 4.14e-03).
- Staleness dominates: lower num_iterations / the async level, or use a correction that is unbiased at the sequence level (see martingale.exact).
- lag 2: effective sample size is 35% of the tokens; 5% would be clipped at eps=0.2. Rollouts this stale are mostly wasted.
- Replayed rows: 12 sequences (100 scored tokens, 20% of scored tokens) at lags 4..15; lowest token-level ESS/n 0.80 at lag 7. The buffer's declared weights give a sequence-level ESS/n of 0.83 over 12 rows; the trainer's scores measure 0.83.

sequences: 76 (mixed-revision: 0, unscored: 12), scored tokens: 494

Engine-mismatch floor (lag 0): mean |log r| = 4.143e-03, max |log r| = 2.480e-01, ESS/n = 0.999

| lag | tokens | seqs (mixed) | mean log r | mean abs log r | max abs log r | ratio p95 | ESS/n | clipped@0.1 | clipped@0.2 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 104 | 27 (0) | +1.286e-03 | 4.143e-03 | 2.480e-01 | 1.0035 | 0.999 | 0.019 | 0.010 |
| 1 | 93 | 25 (0) | -3.026e-01 | 3.048e-01 | 8.817e+00 | 1.0093 | 0.961 | 0.043 | 0.043 |
| 2 | 104 | 27 (0) | -3.503e-01 | 4.078e-01 | 1.068e+01 | 1.0108 | 0.351 | 0.077 | 0.048 |
| 3 | 93 | 25 (0) | -3.978e-01 | 4.004e-01 | 1.087e+01 | 1.0093 | 0.956 | 0.097 | 0.054 |
| 4 | 7 | 2 (0) | +1.806e-03 | 1.954e-03 | 9.297e-03 | 1.0093 | 1.000 | 0.000 | 0.000 |
| 5 | 25 | 5 (0) | -7.404e-01 | 7.416e-01 | 1.032e+01 | 1.0048 | 0.862 | 0.200 | 0.200 |
| 6 | 7 | 2 (0) | -3.210e-02 | 3.299e-02 | 1.688e-01 | 1.0025 | 0.997 | 0.143 | 0.000 |
| 7 | 25 | 5 (0) | -1.149e+00 | 1.150e+00 | 9.979e+00 | 1.0049 | 0.803 | 0.280 | 0.280 |
| 8 | 7 | 2 (0) | -1.207e-04 | 3.500e-03 | 1.003e-02 | 1.0089 | 1.000 | 0.000 | 0.000 |
| 9 | 4 | 1 (0) | -3.772e-04 | 1.546e-03 | 3.555e-03 | 1.0023 | 1.000 | 0.000 | 0.000 |
| 10 | 7 | 2 (0) | +9.643e-04 | 2.399e-03 | 8.805e-03 | 1.0088 | 1.000 | 0.000 | 0.000 |
| 11 | 4 | 1 (0) | +1.397e-04 | 1.004e-03 | 2.269e-03 | 1.0023 | 1.000 | 0.000 | 0.000 |
| 12 | 3 | 1 (0) | +3.055e-03 | 3.125e-03 | 8.693e-03 | 1.0087 | 1.000 | 0.000 | 0.000 |
| 13 | 4 | 1 (0) | +2.857e-04 | 9.568e-04 | 2.283e-03 | 1.0023 | 1.000 | 0.000 | 0.000 |
| 14 | 3 | 1 (0) | +3.191e-03 | 3.191e-03 | 8.931e-03 | 1.0090 | 1.000 | 0.000 | 0.000 |
| 15 | 4 | 1 (0) | -4.193e-05 | 1.437e-03 | 2.943e-03 | 1.0023 | 1.000 | 0.000 | 0.000 |

Fresh vs replayed rows:

| provenance | lag | tokens | seqs | mean abs log r | ESS/n | floor source |
|---|---|---|---|---|---|---|
| fresh | 0 | 104 | 27 | 4.143e-03 | 0.999 | own lag-0 |
| fresh | 1 | 93 | 25 | 3.048e-01 | 0.961 | own lag-0 |
| fresh | 2 | 104 | 27 | 4.078e-01 | 0.351 | own lag-0 |
| fresh | 3 | 93 | 25 | 4.004e-01 | 0.956 | own lag-0 |
| replayed | 4 | 7 | 2 | 1.954e-03 | 1.000 | fresh lag-0 |
| replayed | 5 | 25 | 5 | 7.416e-01 | 0.862 | fresh lag-0 |
| replayed | 6 | 7 | 2 | 3.299e-02 | 0.997 | fresh lag-0 |
| replayed | 7 | 25 | 5 | 1.150e+00 | 0.803 | fresh lag-0 |
| replayed | 8 | 7 | 2 | 3.500e-03 | 1.000 | fresh lag-0 |
| replayed | 9 | 4 | 1 | 1.546e-03 | 1.000 | fresh lag-0 |
| replayed | 10 | 7 | 2 | 2.399e-03 | 1.000 | fresh lag-0 |
| replayed | 11 | 4 | 1 | 1.004e-03 | 1.000 | fresh lag-0 |
| replayed | 12 | 3 | 1 | 3.125e-03 | 1.000 | fresh lag-0 |
| replayed | 13 | 4 | 1 | 9.568e-04 | 1.000 | fresh lag-0 |
| replayed | 14 | 3 | 1 | 3.191e-03 | 1.000 | fresh lag-0 |
| replayed | 15 | 4 | 1 | 1.437e-03 | 1.000 | fresh lag-0 |

Replayed sequences: 12 (12 scored, 0 without a fresh origin in this record).
Sequence weights over 12 rows (0 not representable in float64): declared (importance weight x rescale) ESS/n = 0.825; measured from the trainer's scores ESS/n = 0.832; dispersion of log(declared) - log(measured) after per-step centring: 0.725 nats over 11 rows.

Exact values (reduced fractions) are in the JSON report; the table is float64 and informational.
