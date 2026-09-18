"""
checker_daemon.py — Long-running ledger audit sidecar.

Wraps checker/verify.py in a daemon that:
  - Polls the SQLite ledger for new trajectories
  - Verifies each trajectory using the independent checker
  - Exports Prometheus metrics: trajectories_verified, forgeries_detected
  - Optionally halts training on first forgery (configurable)

Usage:
    daemon = CheckerDaemon(ledger, store, CheckerConfig(seed=b"...", halt_on_forgery=True))
    metrics = daemon.scan_once()

    # Long-running loop:
    daemon.run(interval_seconds=30)
"""
from __future__ import annotations

import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram

from martingale.store.sqlite import SQLiteLedger, SQLiteRevisionStore

# Prometheus metrics (module-level singletons)
_TRAJ_CHECKED = Counter(
    "martingale_trajectories_checked_total",
    "Total trajectories verified by checker daemon",
)
_FORGERIES = Counter(
    "martingale_forgeries_detected_total",
    "Total forgeries detected by checker daemon",
)
_SCAN_DURATION = Histogram(
    "martingale_scan_duration_ms",
    "Duration of each checker scan in milliseconds",
    buckets=[1, 5, 10, 50, 100, 500, 1000, 5000],
)
_LAST_SCAN_OK = Gauge(
    "martingale_last_scan_ok",
    "1 if the last scan found no forgeries, 0 otherwise",
)


@dataclass
class CheckerConfig:
    seed: bytes
    halt_on_forgery: bool = True
    incremental: bool = False  # only check new trajectories since last scan


@dataclass
class DaemonMetrics:
    total_checked: int
    forgeries_detected: int
    last_scan_ok: bool
    scan_duration_ms: float


class CheckerDaemon:
    """
    Ledger audit daemon. Uses checker/verify.py (no martingale imports) to verify
    trajectories stored in a SQLiteLedger.
    """

    def __init__(
        self,
        ledger: SQLiteLedger,
        store: SQLiteRevisionStore,
        config: CheckerConfig,
    ) -> None:
        self._ledger = ledger
        self._store = store
        self._config = config
        self._checked_keys: set[tuple[int, int]] = set()  # (actor_id, episode_id)

        # Import checker (no martingale imports — isolation enforced)
        _root = Path(__file__).parent.parent.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from checker.verify import verify_trajectory
        self._verify_trajectory = verify_trajectory

    def scan_once(self) -> DaemonMetrics:
        """
        Scan trajectories, verify each, update Prometheus metrics.
        Returns DaemonMetrics for this scan.
        """
        t0 = time.perf_counter()
        n_checked = 0
        n_forgeries = 0

        # Export ledger to file-based format for checker compatibility
        with tempfile.TemporaryDirectory(prefix="martingale_checker_") as tmp:
            tmp_path = Path(tmp)
            ledger_dir = tmp_path / "ledger"
            store_dir = tmp_path / "store"
            self._ledger.export_for_checker(ledger_dir, store_dir)

            for path in sorted(ledger_dir.glob("actor*.json")):
                import json
                with open(path) as f:
                    traj = json.load(f)

                key = (traj["actor_id"], traj["episode_id"])
                if self._config.incremental and key in self._checked_keys:
                    continue

                errors = self._verify_trajectory(traj, store_dir, self._config.seed)
                n_checked += 1
                self._checked_keys.add(key)

                if errors:
                    n_forgeries += 1
                    _FORGERIES.inc()
                    if self._config.halt_on_forgery:
                        raise RuntimeError(
                            f"forgery detected in trajectory "
                            f"actor={traj['actor_id']} episode={traj['episode_id']}: "
                            f"{errors[0]}"
                        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        ok = n_forgeries == 0

        _TRAJ_CHECKED.inc(n_checked)
        _SCAN_DURATION.observe(elapsed_ms)
        _LAST_SCAN_OK.set(1 if ok else 0)

        return DaemonMetrics(
            total_checked=n_checked,
            forgeries_detected=n_forgeries,
            last_scan_ok=ok,
            scan_duration_ms=elapsed_ms,
        )

    def run(self, interval_seconds: float = 30.0) -> None:
        """Run continuous scanning loop (blocking)."""
        while True:
            self.scan_once()
            time.sleep(interval_seconds)
