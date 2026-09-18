"""Tests for cli.py — martingale CLI commands."""
import json
import pytest
from click.testing import CliRunner
from fractions import Fraction
from martingale.cli import cli


class TestCLI:
    def test_init_creates_directories(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--dir", str(tmp_path / "workspace")])
        assert result.exit_code == 0
        assert (tmp_path / "workspace" / "revisions.db").exists()
        assert (tmp_path / "workspace" / "ledger.db").exists()

    def test_status_empty(self, tmp_path):
        runner = CliRunner()
        runner.invoke(cli, ["init", "--dir", str(tmp_path)])
        result = runner.invoke(cli, ["status", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "revisions" in result.output.lower()

    def test_status_shows_counts(self, tmp_path):
        from martingale.store.sqlite import SQLiteRevisionStore, SQLiteLedger
        from martingale.draw import draw_action
        from martingale.ledger import ActionRecord, GENESIS_DIGEST
        # Populate store and ledger
        store = SQLiteRevisionStore(tmp_path / "revisions.db")
        ledger = SQLiteLedger(tmp_path / "ledger.db", store)
        probs = {0: Fraction(1, 2), 1: Fraction(1, 2)}
        rev = store.publish({0: probs})
        draw = draw_action(probs, seed=b"cli", actor_id=0, episode=0, step=0)
        rec = ActionRecord(
            revision_digest=rev.digest, state=0, action=draw.action,
            behavior_prob=probs[draw.action], draw=draw,
            env_next_state=0, env_reward=Fraction(1),
            prev_digest=GENESIS_DIGEST,
        )
        ledger.append_trajectory(actor_id=0, episode_id=0, records=[rec])

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "1" in result.output  # at least 1 revision and 1 trajectory

    def test_verify_clean_ledger(self, tmp_path):
        from martingale.store.sqlite import SQLiteRevisionStore, SQLiteLedger
        from martingale.draw import draw_action
        from martingale.ledger import ActionRecord, GENESIS_DIGEST
        store = SQLiteRevisionStore(tmp_path / "revisions.db")
        ledger = SQLiteLedger(tmp_path / "ledger.db", store)
        probs = {0: Fraction(1, 2), 1: Fraction(1, 2)}
        rev = store.publish({0: probs})
        draw = draw_action(probs, seed=b"cli-verify", actor_id=0, episode=0, step=0)
        rec = ActionRecord(
            revision_digest=rev.digest, state=0, action=draw.action,
            behavior_prob=probs[draw.action], draw=draw,
            env_next_state=0, env_reward=Fraction(1),
            prev_digest=GENESIS_DIGEST,
        )
        ledger.append_trajectory(actor_id=0, episode_id=0, records=[rec])

        runner = CliRunner()
        result = runner.invoke(cli, [
            "verify", "--dir", str(tmp_path), "--seed", "cli-verify"
        ])
        assert result.exit_code == 0
        assert "passed" in result.output.lower()

    def test_verify_exits_nonzero_on_forgery(self, tmp_path):
        import sqlite3
        from martingale.store.sqlite import SQLiteRevisionStore, SQLiteLedger
        from martingale.draw import draw_action
        from martingale.ledger import ActionRecord, GENESIS_DIGEST
        store = SQLiteRevisionStore(tmp_path / "revisions.db")
        ledger = SQLiteLedger(tmp_path / "ledger.db", store)
        probs = {0: Fraction(1, 2), 1: Fraction(1, 2)}
        rev = store.publish({0: probs})
        draw = draw_action(probs, seed=b"cli-forge", actor_id=0, episode=0, step=0)
        rec = ActionRecord(
            revision_digest=rev.digest, state=0, action=draw.action,
            behavior_prob=probs[draw.action], draw=draw,
            env_next_state=0, env_reward=Fraction(1),
            prev_digest=GENESIS_DIGEST,
        )
        ledger.append_trajectory(actor_id=0, episode_id=0, records=[rec])

        # Tamper the record
        con = sqlite3.connect(str(tmp_path / "ledger.db"))
        row = con.execute("SELECT records_json FROM trajectories").fetchone()
        records = json.loads(row[0])
        records[0]["action"] = 1 - records[0]["action"]
        con.execute("UPDATE trajectories SET records_json=?", (json.dumps(records),))
        con.commit()
        con.close()

        runner = CliRunner()
        result = runner.invoke(cli, [
            "verify", "--dir", str(tmp_path), "--seed", "cli-forge"
        ])
        assert result.exit_code != 0
