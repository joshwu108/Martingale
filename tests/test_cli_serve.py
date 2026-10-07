"""`martingale serve` starts the rollout inspector over tokens.db; the old Observatory flags are still accepted."""
from unittest.mock import patch

from click.testing import CliRunner

from martingale.cli import cli


class TestServeCommand:
    def test_serve_command_exists(self):
        result = CliRunner().invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--tokenizer" in result.output and "--head" in result.output

    def test_serve_starts_uvicorn(self, tmp_path):
        runner = CliRunner()
        runner.invoke(cli, ["init", "--dir", str(tmp_path)])
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(cli, ["serve", "--dir", str(tmp_path), "--host", "127.0.0.1", "--port", "7373"])
        assert result.exit_code == 0, result.output
        _, kwargs = mock_run.call_args
        assert kwargs.get("host") == "127.0.0.1" and kwargs.get("port") == 7373

    def test_serve_default_port_is_7373(self, tmp_path):
        runner = CliRunner()
        runner.invoke(cli, ["init", "--dir", str(tmp_path)])
        with patch("uvicorn.run") as mock_run:
            runner.invoke(cli, ["serve", "--dir", str(tmp_path)])
        assert mock_run.call_args.kwargs.get("port") == 7373

    def test_serve_passes_app_to_uvicorn(self, tmp_path):
        runner = CliRunner()
        runner.invoke(cli, ["init", "--dir", str(tmp_path)])
        with patch("uvicorn.run") as mock_run:
            runner.invoke(cli, ["serve", "--dir", str(tmp_path)])
        app = mock_run.call_args.args[0]
        assert set(app.openapi()["paths"]) >= {"/", "/api/doctor", "/api/steps", "/api/sequences",
                                                "/api/sequence/{digest}", "/api/record"}

    def test_serve_prints_url(self, tmp_path):
        runner = CliRunner()
        runner.invoke(cli, ["init", "--dir", str(tmp_path)])
        with patch("uvicorn.run"):
            result = runner.invoke(cli, ["serve", "--dir", str(tmp_path)])
        assert "http://127.0.0.1:7373" in result.output

    def test_serve_missing_workspace_fails(self, tmp_path):
        result = CliRunner().invoke(cli, ["serve", "--dir", str(tmp_path / "nonexistent")])
        assert result.exit_code != 0

    def test_serve_without_token_record_fails(self, tmp_path):
        (tmp_path / "revisions.db").write_bytes(b"")
        result = CliRunner().invoke(cli, ["serve", "--dir", str(tmp_path)])
        assert result.exit_code != 0 and "tokens.db" in result.output

    def test_old_observatory_flags_are_still_accepted(self, tmp_path):
        runner = CliRunner()
        runner.invoke(cli, ["init", "--dir", str(tmp_path)])
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(cli, ["serve", "--dir", str(tmp_path), "--scan-interval", "0",
                                         "--seed", "serve-test", "--no-halt-on-forgery"])
        assert result.exit_code == 0, result.output
        assert mock_run.called

    def test_tokenizer_and_head_reach_the_app(self, tmp_path):
        runner = CliRunner()
        runner.invoke(cli, ["init", "--dir", str(tmp_path)])
        with patch("uvicorn.run"), patch("martingale.server.create_app") as mk:
            result = runner.invoke(cli, ["serve", "--dir", str(tmp_path), "--tokenizer", "org/model", "--head", "a" * 64])
        assert result.exit_code == 0, result.output
        kwargs = mk.call_args.kwargs
        assert kwargs["tokenizer"] == "org/model" and kwargs["head"] == "a" * 64

    def test_serve_mentions_ids_only_when_transformers_is_missing(self, tmp_path, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "transformers", None)
        runner = CliRunner()
        runner.invoke(cli, ["init", "--dir", str(tmp_path)])
        with patch("uvicorn.run"):
            result = runner.invoke(cli, ["serve", "--dir", str(tmp_path), "--tokenizer", "org/model"])
        assert result.exit_code == 0, result.output
        assert "ids only" in result.output
