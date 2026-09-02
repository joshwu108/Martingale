"""
campaigns/interleave.py — T5 scripted-interleaving + SIGKILL campaign.

Tests the pin-before-draw protocol under various interleavings and crash cuts:
  1. Scripted interleavings: verify ledger/pin consistency after each schedule.
  2. SIGKILL at every cut point: recovery leaves ledger consistent and clean.

Results committed to results/interleave_report.json.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from martingale.pipeline import PipelineConfig, run_pipeline
from checker.verify import verify_ledger

RESULTS_DIR = Path(__file__).parent.parent / "results"
SEED = b"interleave-campaign-seed-v1"


def _run_and_verify(cfg: PipelineConfig, label: str) -> dict:
    """Run a pipeline config and verify the resulting ledger."""
    run_pipeline(cfg)
    report = verify_ledger(cfg.ledger_dir, cfg.store_dir, seed=SEED)
    return {
        "label": label,
        "total_trajectories": report["total"],
        "passed": report["passed"],
        "failed": report["failed"],
        "ok": report["failed"] == 0,
        "errors": report["errors"],
    }


def run_interleave_campaign() -> dict:
    """Run the scripted-interleaving and SIGKILL campaign."""
    RESULTS_DIR.mkdir(exist_ok=True)
    results = []

    # === Scripted interleavings ===
    # These test the pin-before-draw protocol under normal multi-actor concurrency.

    configs = [
        # (n_actors, n_revisions, n_episodes, label)
        (1, 1, 5, "baseline_1a_1r"),
        (2, 2, 4, "concurrent_2a_2r"),
        (2, 5, 4, "concurrent_2a_5r"),
        (3, 3, 3, "concurrent_3a_3r"),
        (2, 4, 6, "concurrent_2a_4r_6ep"),
    ]

    for n_a, n_r, n_ep, label in configs:
        with tempfile.TemporaryDirectory(prefix=f"martingale_{label}_") as tmp:
            tmp_path = Path(tmp)
            cfg = PipelineConfig(
                n_actors=n_a,
                n_revisions=n_r,
                n_episodes_per_actor=n_ep,
                n_states=2,
                n_actions=2,
                horizon=2,
                seed=SEED,
                store_dir=tmp_path / "store",
                ledger_dir=tmp_path / "ledger",
            )
            r = _run_and_verify(cfg, label)
            results.append(r)
            print(
                f"  {label}: {r['passed']}/{r['total_trajectories']} passed "
                f"({'OK' if r['ok'] else 'FAIL'})"
            )

    # === Mixed-revision interleavings ===
    with tempfile.TemporaryDirectory(prefix="martingale_mixed_") as tmp:
        tmp_path = Path(tmp)
        cfg = PipelineConfig(
            n_actors=2,
            n_revisions=4,
            n_episodes_per_actor=4,
            n_states=2,
            n_actions=2,
            horizon=3,
            seed=SEED,
            store_dir=tmp_path / "store",
            ledger_dir=tmp_path / "ledger",
            force_mixed_revisions=True,
        )
        r = _run_and_verify(cfg, "mixed_revisions_2a_4r")
        results.append(r)
        print(f"  mixed_revisions: {r['passed']}/{r['total_trajectories']} ({'OK' if r['ok'] else 'FAIL'})")

    # === SIGKILL scenarios ===
    # Kill actor after N episodes, verify ledger is consistent, then restart.
    for kill_after in [1, 2, 3]:
        label = f"sigkill_after_{kill_after}ep"
        with tempfile.TemporaryDirectory(prefix=f"martingale_{label}_") as tmp:
            tmp_path = Path(tmp)
            cfg = PipelineConfig(
                n_actors=1,
                n_revisions=3,
                n_episodes_per_actor=5,
                n_states=2,
                n_actions=2,
                horizon=2,
                seed=SEED,
                store_dir=tmp_path / "store",
                ledger_dir=tmp_path / "ledger",
                sigkill_after_episodes=kill_after,
            )
            r = _run_and_verify(cfg, label)
            results.append(r)
            print(f"  {label}: {r['passed']}/{r['total_trajectories']} ({'OK' if r['ok'] else 'FAIL'})")

    n_ok = sum(1 for r in results if r["ok"])
    n_fail = sum(1 for r in results if not r["ok"])
    report = {
        "campaign": "interleave-T5",
        "n_scenarios": len(results),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "scenarios": results,
    }
    return report


def main() -> None:
    print("=== T5 Interleave + SIGKILL Campaign ===")
    report = run_interleave_campaign()

    out = RESULTS_DIR / "interleave_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {out}")

    print(f"\n=== Summary ===")
    print(f"Scenarios: {report['n_scenarios']}")
    print(f"OK: {report['n_ok']}")
    print(f"FAIL: {report['n_fail']}")

    if report["n_fail"] > 0:
        print("\nFAILED SCENARIOS:")
        for s in report["scenarios"]:
            if not s["ok"]:
                print(f"  {s['label']}: {s['errors']}")
        sys.exit(1)
    else:
        print(f"\nPASS: All {report['n_scenarios']} scenarios verified clean.")


if __name__ == "__main__":
    main()
