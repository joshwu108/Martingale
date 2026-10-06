---
name: Estimator certificate request
about: Request that an estimator be certified by the exact bench
labels: estimator
---

**Estimator name**

**Definition as a per-trajectory weight function.** Given one trajectory
`(state, action, next_state, reward)*`, target policy `pi`, behavior policy `b`,
what gradient contribution `{(state, action): value}` does it assign? Give the
weight as a function of the ratios and return, using only exact rational
operations (no roots, exp, or log).

**Paper or reference** (link, equation number)

**Expected unbiasedness** under `E_b[...]` versus the true gradient: unbiased
always / unbiased under a condition (state it) / expected biased.

**Anything not exactly representable?** (e.g. length-normalised ratios)
