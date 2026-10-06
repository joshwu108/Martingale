"""
campaigns/estimator_bench.py — exact bias certificates for the built-in defendants.

Runs martingale.exact.check_unbiased over every built-in estimator and writes
results/estimator_bench_report.json. Each certificate is an exact rational
identity (bias == 0) or exact inequality (bias vector given as reduced
fractions). Floats appear only in fields labelled informational.

    uv run python -m campaigns.estimator_bench
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

from martingale.exact import check_unbiased, defendants

RESULTS_DIR = Path(__file__).parent.parent / "results"
CAMPAIGN_SEED = b"martingale-estimator-bench-v1"
N_CONFIGS = 20
LAGS = (0, 1, 2, 4, 8)

DEFENDANTS = {
    "unweighted": defendants.unweighted,
    "seq_is": defendants.seq_is,
    "token_is": defendants.token_is,
    "seq_truncated_is(cap=2)": defendants.seq_truncated_is(Fraction(2)),
    "token_truncated_is(cap=2)": defendants.token_truncated_is(Fraction(2)),
    "seq_masked_is(1/2,2)": defendants.seq_masked_is(Fraction(1, 2), Fraction(2)),
    "token_masked_is(1/2,2)": defendants.token_masked_is(Fraction(1, 2), Fraction(2)),
    "seq_ppo_clip(eps=1/5)": defendants.seq_ppo_clip(Fraction(1, 5)),
    "token_ppo_clip(eps=1/5)": defendants.token_ppo_clip(Fraction(1, 5)),
    "seq_cispo(eps=1/5)": defendants.seq_cispo(Fraction(1, 5), Fraction(1, 5)),
    "token_cispo(eps=1/5)": defendants.token_cispo(Fraction(1, 5), Fraction(1, 5)),
}


def run(n_configs: int = N_CONFIGS, verbose: bool = True) -> dict:
    reports = {}
    for name, est in DEFENDANTS.items():
        report = check_unbiased(est, n_configs=n_configs, lags=LAGS, seed=CAMPAIGN_SEED, name=name)
        by_lag = {}
        for lag in LAGS:
            certs = [c for c in report.certificates if c.lag == lag]
            by_lag[lag] = sum(1 for c in certs if c.is_unbiased)
        reports[name] = {"report": report.to_dict(), "unbiased_by_lag": by_lag}
        if verbose:
            cells = " ".join(f"lag{lag}:{by_lag[lag]}/{n_configs}" for lag in LAGS)
            print(f"{name:28s} unbiased {cells}")

    # Expected shape of the evidence: sequence-level IS unbiased everywhere; every
    # other defendant unbiased at lag 0 and biased somewhere at lag >= 1.
    expectations = {}
    for name, r in reports.items():
        bl = r["unbiased_by_lag"]
        if name == "seq_is":
            expectations[name] = all(v == n_configs for v in bl.values())
        else:
            expectations[name] = bl[0] == n_configs and any(bl[l] < n_configs for l in LAGS if l > 0)

    out = {
        "campaign": "estimator_bench", "seed": CAMPAIGN_SEED.decode(),
        "n_configs": n_configs, "lags": list(LAGS),
        "expectations_met": expectations,
        "defendants": {name: r["report"] for name, r in reports.items()},
    }
    return out


def main() -> None:
    out = run()
    RESULTS_DIR.mkdir(exist_ok=True)
    target = RESULTS_DIR / "estimator_bench_report.json"
    target.write_text(json.dumps(out, indent=1))
    print(f"wrote {target}")
    if not all(out["expectations_met"].values()):
        print("EXPECTATION FAILED:", out["expectations_met"])
        sys.exit(1)


if __name__ == "__main__":
    main()
