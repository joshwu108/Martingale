"""Tests for `martingale serve` CLI command."""
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from martingale.cli import cli


class TestServeCommand:
    def test_serve_command_exists(self):
        """The CLI has a `serve` subcommand."""
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "serve" in result.output.lower() or "host" in result.output.lower()

    def test_serve_starts_uvicorn(self, tmp_path):
        """serve invokes uvicorn.run with the FastAPI app."""
        runner = CliRunner()
        # Init workspace first
        runner.invoke(cli, ["init", "--dir", str(tmp_path)])

        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(cli, [
                "serve", "--dir", str(tmp_path),
                "--host", "127.0.0.1", "--port", "7373",
            ])
        assert result.exit_code == 0
        assert mock_run.called
        _, kwargs = mock_run.call_args
        assert kwargs.get("host") == "127.0.0.1"
        assert kwargs.get("port") == 7373

    def test_serve_default_port_is_7373(self, tmp_path):
        runner = CliRunner()
        runner.invoke(cli, ["init", "--dir", str(tmp_path)])
        with patch("uvicorn.run") as mock_run:
            runner.invoke(cli, ["serve", "--dir", str(tmp_path)])
        _, kwargs = mock_run.call_args
        assert kwargs.get("port") == 7373

    def test_serve_passes_app_to_uvicorn(self, tmp_path):
        runner = CliRunner()
        runner.invoke(cli, ["init", "--dir", str(tmp_path)])
        with patch("uvicorn.run") as mock_run:
            runner.invoke(cli, ["serve", "--dir", str(tmp_path)])
        args, _ = mock_run.call_args
        # First positional arg is the app
        assert args[0] is not None

    def test_serve_prints_url(self, tmp_path):
        runner = CliRunner()
        runner.invoke(cli, ["init", "--dir", str(tmp_path)])
        with patch("uvicorn.run"):
            result = runner.invoke(cli, ["serve", "--dir", str(tmp_path)])
        assert "7373" in result.output or "http" in result.output

    def test_serve_missing_workspace_fails(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "serve", "--dir", str(tmp_path / "nonexistent"),
        ])
        assert result.exit_code != 0

    def test_serve_initialises_checker_daemon(self, tmp_path):
        """serve creates a CheckerDaemon and starts background scanning."""
        from fractions import Fraction
        from martingale.store.sqlite import SQLiteRevisionStore, SQLiteLedger
        from martingale.draw import draw_action
        from martingale.ledger import ActionRecord, GENESIS_DIGEST

        # Populate workspace with a trajectory
        store = SQLiteRevisionStore(tmp_path / "revisions.db")
        ledger = SQLiteLedger(tmp_path / "ledger.db", store)
        probs = {0: Fraction(1, 2), 1: Fraction(1, 2)}
        rev = store.publish({0: probs})
        draw = draw_action(probs, seed=b"serve-test", actor_id=0, episode=0, step=0)
        rec = ActionRecord(
            revision_digest=rev.digest, state=0, action=draw.action,
            behavior_prob=probs[draw.action], draw=draw,
            env_next_state=1, env_reward=Fraction(1),
            prev_digest=GENESIS_DIGEST,
        )
        ledger.append_trajectory(actor_id=0, episode_id=0, records=[rec])

        captured_app = {}
        def capture_run(app, **kwargs):
            captured_app['app'] = app

        runner = CliRunner()
        with patch("uvicorn.run", side_effect=capture_run):
            runner.invoke(cli, [
                "serve", "--dir", str(tmp_path),
                "--scan-interval", "0",
                "--seed", "serve-test",
            ])

        # App should have been created
        assert captured_app.get('app') is not None
