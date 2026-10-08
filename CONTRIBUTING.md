# Contributing

## Setup

```bash
uv sync                  # core CLI and its small Click dependency
uv sync --extra prod     # optional: torch, prometheus, CLI
make check               # checker import isolation + full test suite
```

## Rules

- **Exactness.** No floats on decision or measurement paths. Use
  `fractions.Fraction`. Floats are allowed only at declared boundaries
  (currently `campaigns/float_baselines.py`) and any float-derived number must
  be labelled informational.
- **Checker isolation.** `checker/` imports nothing from `martingale`.
  `scripts/check_imports.py` enforces it (`make check-imports`, and CI).
- **File size.** Keep files around 400 lines; split by feature, not by type.
- **TDD.** Write the failing test first, then the minimal implementation.
- **Never delete** a failing test, a surviving mutant, or an inconvenient
  exact-equality failure. Fix the code, or pin it as a strict xfail with a
  reason (see `tests/test_prod_checker.py`).
- **Preregistered thresholds** in `docs/preregistration.md` are frozen.
  Changing one requires an explicit maintainer decision.
- **Claims.** Every README claim needs a row in `paper/claim_evidence.md`
  pointing at an existing artifact. If scope changes, update
  `docs/nonclaims.md`.

## Adding a defendant

A defendant is an estimator `(traj, target, behavior) -> {(state, action): Fraction}`.
In `src/martingale/exact/defendants.py`:

1. Write the per-step weight as a `WeightFn` `(ratio, G) -> Fraction` and wrap it
   with `_seq_level(...)` (sequence ratio) or `_token_level(...)` (per-token
   ratio); register it with `_named(est, "name(params)")`.
2. Use Fraction arithmetic only. Raise `SupportError` rather than approximating.
3. Add tests in `tests/test_exact_bench.py` for the expected unbiased/biased
   pattern, then add it to `campaigns/estimator_bench.py` with its expectation.
4. Regenerate `results/estimator_bench_report.json` and update the README table.

## Regenerating results

```bash
uv run python -m campaigns.identity         # results/identity_report.json
uv run python -m campaigns.staleness        # results/staleness_report.json
uv run python -m campaigns.mutation         # results/mutation_report.json
uv run python -m campaigns.interleave       # results/interleave_report.json
uv run python -m campaigns.boundary         # results/boundary_report.json
uv run python -m campaigns.e2e_demo         # results/e2e_report.json
uv run python -m campaigns.estimator_bench  # results/estimator_bench_report.json
```

Commit regenerated JSON together with the change that caused it.

## Commits

`<type>: <description>` with types feat, fix, refactor, docs, test, chore,
perf, ci. Run `make check` before committing.
