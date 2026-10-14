"""benchmarks/modal/advise_calibration.py — the advice each committed record would have produced.

Prints the markdown tables in docs/advise.md: per-lag ESS/n of the fresh rows, then
the advice at a sweep of ESS thresholds with the run's own learning rate and updates
per pass. Reads the records read-only; writes nothing.

    uv run python benchmarks/modal/advise_calibration.py
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from martingale.diagnostics import decompose
from martingale.diagnostics.advise import AdviseConfig, advise_decomposition
from martingale.record import TokenLedger

RESULTS = Path(__file__).resolve().parent / "results"
THRESHOLDS = (0.3, 0.5, 0.9, 0.95)
TRL_LR = "5e-6"   # trl_grpo_vllm.py passes learning_rate=5e-6; the TRL configs do not record it


def run_knobs(config: dict) -> tuple[Fraction, int]:
    """(learning rate, optimizer updates per pass over one generation) from a run's config.json."""
    if "verl" in config:
        return Fraction(str(config["lr"])), config["train_batch_size"] // config["ppo_mini_batch_size"]
    return Fraction(TRL_LR), config["steps_per_generation"]


def main() -> None:
    records = sorted(p.parent for p in RESULTS.glob("*/tokens.db"))
    decomps = {}
    for ws in records:
        ledger = TokenLedger(ws / "tokens.db", readonly=True)
        try:
            decomps[ws.name] = decompose(ledger)
        finally:
            ledger.close()
    print("| record | head | lag: tokens (share) ESS/n, max abs log r |\n|---|---|---|")
    for ws in records:
        a = advise_decomposition(decomps[ws.name])
        cells = "; ".join(f"{r.lag}: {r.n_tokens} ({float(r.token_share):.0%}) {r.ess_fraction:.3f}, "
                          f"{float(Fraction(r.max_abs_log_ratio)):.2g}" for r in a.rows)
        print(f"| {ws.name} | {(ws / 'head.txt').read_text().strip()[:12]} | {cells} |")
    print("\n| record | lr | U | " + " | ".join(f"ESS >= {t}" for t in THRESHOLDS) + " |")
    print("|---|---|---|" + "---|" * len(THRESHOLDS))
    for ws in records:
        lr, per_pass = run_knobs(json.loads((ws / "config.json").read_text()))
        cells = []
        for t in THRESHOLDS:
            a = advise_decomposition(decomps[ws.name], AdviseConfig(ess_min=t), lr=lr, updates_per_pass=per_pass)
            if a.passing_interval is None:
                cells.append(a.status)
            else:
                cells.append(f"{a.status}: S={a.passing_interval}/{a.observed_interval}, "
                             f"iters<={a.knobs['max_iterations']}, lr<={float(a.lr_ceiling):.3g}")
        print(f"| {ws.name} | {float(lr):.0e} | {per_pass} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
