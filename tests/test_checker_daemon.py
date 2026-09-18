"""Tests for checker_daemon.py — long-running ledger audit sidecar."""
import time
import pytest
from fractions import Fraction
from pathlib import Path
from martingale.checker_daemon import CheckerDaemon, CheckerConfig, DaemonMetrics
from martingale.draw import draw_action
from martingale.ledger import ActionRecord, GENESIS_DIGEST
from martingale.store.sqlite import SQLiteRevisionStore, SQLiteLedger


SEED = b"checker-daemon-test"


def _write_valid_traj(ledger, store, actor_id=0, episode_id=0):
    probs = {0: Fraction(1, 2), 1: Fraction(1, 2)}
    rev = store.publish({0: probs})
    draw = draw_action(probs, seed=SEED, actor_id=actor_id, episode=episode_id, step=0)
    rec = ActionRecord(
        revision_digest=rev.digest, state=0, action=draw.action,
        behavior_prob=probs[draw.action], draw=draw,
        env_next_state=0, env_reward=Fraction(1),
        prev_digest=GENESIS_DIGEST,
    )
    ledger.append_trajectory(actor_id=actor_id, episode_id=episode_id, records=[rec])
    return rev.digest


class TestCheckerDaemon:
    def test_scan_valid_ledger_all_pass(self, tmp_path):
        store = SQLiteRevisionStore(tmp_path / "rev.db")
        ledger = SQLiteLedger(tmp_path / "ledger.db", store)
        _write_valid_traj(ledger, store)
        cfg = CheckerConfig(seed=SEED, halt_on_forgery=False)
        daemon = CheckerDaemon(ledger, store, cfg)
        metrics = daemon.scan_once()
        assert metrics.total_checked == 1
        assert metrics.forgeries_detected == 0
        assert metrics.last_scan_ok is True

    def test_scan_detects_zero_forgeries_on_clean_ledger(self, tmp_path):
        store = SQLiteRevisionStore(tmp_path / "rev.db")
        ledger = SQLiteLedger(tmp_path / "ledger.db", store)
        for ep in range(5):
            _write_valid_traj(ledger, store, episode_id=ep)
        cfg = CheckerConfig(seed=SEED, halt_on_forgery=False)
        daemon = CheckerDaemon(ledger, store, cfg)
        metrics = daemon.scan_once()
        assert metrics.total_checked == 5
        assert metrics.forgeries_detected == 0

    def test_metrics_structure(self, tmp_path):
        store = SQLiteRevisionStore(tmp_path / "rev.db")
        ledger = SQLiteLedger(tmp_path / "ledger.db", store)
        _write_valid_traj(ledger, store)
        daemon = CheckerDaemon(
            ledger, store, CheckerConfig(seed=SEED, halt_on_forgery=False)
        )
        metrics = daemon.scan_once()
        assert isinstance(metrics, DaemonMetrics)
        assert hasattr(metrics, "total_checked")
        assert hasattr(metrics, "forgeries_detected")
        assert hasattr(metrics, "last_scan_ok")
        assert hasattr(metrics, "scan_duration_ms")

    def test_prometheus_metrics_exported(self, tmp_path):
        """Prometheus counters are updated after a scan."""
        store = SQLiteRevisionStore(tmp_path / "rev.db")
        ledger = SQLiteLedger(tmp_path / "ledger.db", store)
        _write_valid_traj(ledger, store)
        daemon = CheckerDaemon(
            ledger, store, CheckerConfig(seed=SEED, halt_on_forgery=False)
        )
        daemon.scan_once()
        # Prometheus registry contains our counters
        from prometheus_client import REGISTRY
        names = {m.name for m in REGISTRY.collect()}
        assert any("martingale" in n for n in names)

    def test_halt_on_forgery_raises(self, tmp_path):
        """With halt_on_forgery=True, a tampered trajectory raises an exception."""
        import json
        store = SQLiteRevisionStore(tmp_path / "rev.db")
        ledger = SQLiteLedger(tmp_path / "ledger.db", store)
        _write_valid_traj(ledger, store)

        # Tamper the trajectory in the DB
        con = ledger._con
        row = con.execute("SELECT records_json FROM trajectories").fetchone()
        records = json.loads(row[0])
        records[0]["action"] = 1 - records[0]["action"]  # flip action
        con.execute("UPDATE trajectories SET records_json=?", (json.dumps(records),))
        con.commit()

        cfg = CheckerConfig(seed=SEED, halt_on_forgery=True)
        daemon = CheckerDaemon(ledger, store, cfg)
        with pytest.raises(RuntimeError, match="forgery"):
            daemon.scan_once()

    def test_incremental_scan_only_new(self, tmp_path):
        """Daemon tracks which trajectories were already checked."""
        store = SQLiteRevisionStore(tmp_path / "rev.db")
        ledger = SQLiteLedger(tmp_path / "ledger.db", store)
        _write_valid_traj(ledger, store, episode_id=0)
        cfg = CheckerConfig(seed=SEED, halt_on_forgery=False, incremental=True)
        daemon = CheckerDaemon(ledger, store, cfg)

        m1 = daemon.scan_once()
        assert m1.total_checked == 1

        # Add a new trajectory
        _write_valid_traj(ledger, store, episode_id=1)
        m2 = daemon.scan_once()
        assert m2.total_checked == 1  # only the new one checked
