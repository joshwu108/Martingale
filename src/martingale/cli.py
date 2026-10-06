"""
cli.py — martingale CLI.

Commands:
  init    — scaffold workspace (SQLite revision store + ledger)
  status  — show revision count, trajectory count
  verify  — run independent checker over entire ledger
"""
from __future__ import annotations

import sys
from pathlib import Path

import click


@click.group()
def cli():
    """martingale — async-RL provenance and IS-weight infrastructure."""


@cli.command()
@click.option("--dir", "workspace", default=".", show_default=True,
              help="Workspace directory to initialise.")
def init(workspace: str):
    """Scaffold a martingale workspace (revision store + ledger)."""
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)

    from martingale.store.sqlite import SQLiteLedger, SQLiteRevisionStore
    SQLiteRevisionStore(ws / "revisions.db")
    store = SQLiteRevisionStore(ws / "revisions.db")
    SQLiteLedger(ws / "ledger.db", store)
    from martingale.record import Recorder
    Recorder(ws)  # creates tokens.db (LLM revisions, sequences, scores)

    click.echo(f"Initialised martingale workspace at {ws.resolve()}")
    click.echo(f"  revisions : {ws / 'revisions.db'}   (exact rational policies)")
    click.echo(f"  ledger    : {ws / 'ledger.db'}      (exact trajectories)")
    click.echo(f"  tokens    : {ws / 'tokens.db'}      (LLM revisions + token record)")


@cli.command()
@click.option("--dir", "workspace", default=".", show_default=True,
              help="Workspace directory.")
def status(workspace: str):
    """Show revision count, trajectory count, and lag statistics."""
    ws = Path(workspace)
    rev_db = ws / "revisions.db"
    led_db = ws / "ledger.db"

    if not rev_db.exists():
        click.echo(f"No revision store found at {rev_db}. Run: martingale init")
        sys.exit(1)

    from martingale.store.sqlite import SQLiteLedger, SQLiteRevisionStore
    store = SQLiteRevisionStore(rev_db)
    ledger = SQLiteLedger(led_db, store)

    n_revisions = sum(1 for _ in store.all_digests())
    n_trajectories = sum(1 for _ in ledger.all_trajectories())

    click.echo(f"martingale workspace: {ws.resolve()}")
    click.echo(f"  revisions   : {n_revisions}")
    click.echo(f"  trajectories: {n_trajectories}")
    if (ws / "tokens.db").exists():
        from martingale.record import TokenLedger
        tl = TokenLedger(ws / "tokens.db")
        click.echo(f"  llm revisions: {sum(1 for _ in tl.all_revisions())}")
        click.echo(f"  sequences    : {tl.count_sequences()}")


@cli.command()
@click.option("--dir", "workspace", default=".", show_default=True,
              help="Workspace directory.")
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Bind host for the HTTP server.")
@click.option("--port", default=7373, show_default=True,
              help="Bind port for the HTTP server.")
@click.option("--halt-on-forgery/--no-halt-on-forgery", default=True, show_default=True,
              help="Halt training if a forgery is detected.")
@click.option("--scan-interval", default=30.0, show_default=True,
              help="Checker daemon scan interval in seconds (0 = run once and stop).")
@click.option("--seed", "daemon_seed", default="martingale", show_default=True,
              help="Draw seed for checker daemon (UTF-8 encoded).")
def serve(workspace: str, host: str, port: int, halt_on_forgery: bool,
          scan_interval: float, daemon_seed: str):
    """Start the Martingale Observatory dashboard server."""
    import threading

    import uvicorn
    ws = Path(workspace)
    rev_db = ws / "revisions.db"

    if not rev_db.exists():
        click.echo(f"No revision store found at {rev_db}. Run: martingale init --dir {ws}", err=True)
        raise SystemExit(1)

    from martingale.checker_daemon import CheckerConfig, CheckerDaemon
    from martingale.server import create_app
    from martingale.store.sqlite import SQLiteLedger, SQLiteRevisionStore

    store = SQLiteRevisionStore(rev_db)
    ledger = SQLiteLedger(ws / "ledger.db", store)
    seed_bytes = daemon_seed.encode("utf-8")
    daemon = CheckerDaemon(ledger, store, CheckerConfig(
        seed=seed_bytes, halt_on_forgery=halt_on_forgery
    ))

    # Run one immediate scan, then schedule background thread
    daemon.scan_once()
    if scan_interval > 0:
        def _bg():
            import time
            while True:
                time.sleep(scan_interval)
                daemon.scan_once()
        t = threading.Thread(target=_bg, daemon=True)
        t.start()

    app = create_app(store, ledger, daemon)
    click.echo(f"Martingale Observatory → http://{host}:{port}")
    click.echo(f"  workspace : {ws.resolve()}")
    click.echo(f"  revisions : {rev_db}")
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.option("--dir", "workspace", default=".", show_default=True,
              help="Workspace directory.")
@click.option("--seed", default="martingale", show_default=True,
              help="Draw seed (string, encoded as UTF-8).")
@click.option("--head", "expected_head", default=None,
              help="Ledger head published by the run (flight.head()). Without it re-signed chains and trailing "
                   "deletions are undetectable; the token check then prints an UNANCHORED warning.")
def verify(workspace: str, seed: str, expected_head: str | None):
    """Run the independent checkers over the exact ledger and the token record."""
    import tempfile
    ws = Path(workspace)
    failed = False

    if (ws / "tokens.db").exists():
        failed = _verify_tokens(ws, expected_head) or failed
    if not (ws / "ledger.db").exists():
        if failed:
            sys.exit(1)
        return

    from martingale.store.sqlite import SQLiteLedger, SQLiteRevisionStore
    store = SQLiteRevisionStore(ws / "revisions.db")
    ledger = SQLiteLedger(ws / "ledger.db", store)

    seed_bytes = seed.encode("utf-8")

    # Export to file-based format for checker
    with tempfile.TemporaryDirectory(prefix="martingale_verify_") as tmp:
        tmp_path = Path(tmp)
        ledger_dir = tmp_path / "ledger"
        store_dir = tmp_path / "store"
        ledger.export_for_checker(ledger_dir, store_dir)

        from checker.verify import verify_ledger
        report = verify_ledger(ledger_dir, store_dir, seed=seed_bytes)

    click.echo("Verification complete:")
    click.echo(f"  total     : {report['total']}")
    click.echo(f"  passed    : {report['passed']}")
    click.echo(f"  failed    : {report['failed']}")

    if report["failed"] > 0:
        click.echo(f"\nFORGERIES DETECTED ({report['failed']}):", err=True)
        for e in report["errors"]:
            click.echo(f"  {e['file']}: {e['errors'][0]}", err=True)
        sys.exit(1)
    else:
        click.echo(f"\nAll {report['passed']} trajectories verified clean.")
    if failed:
        sys.exit(1)


def _verify_tokens(ws: Path, expected_head: str | None) -> bool:
    """Run checker/verify_tokens.py on the token record. Returns True on failure."""
    import tempfile

    from checker.verify_tokens import verify_export
    from martingale.record import TokenLedger

    tl = TokenLedger(ws / "tokens.db")
    with tempfile.TemporaryDirectory(prefix="martingale_verify_tokens_") as tmp:
        report = verify_export(tl.export_for_checker(Path(tmp) / "export"), expected_head=expected_head)
    click.echo("Token record verification" + (" (anchored):" if report["anchored"]
               else " (UNANCHORED: pass --head to detect re-signed chains):"))
    click.echo(f"  revisions : {report['n_revisions']}")
    click.echo(f"  sequences : {report['n_sequences']}")
    click.echo(f"  tokens    : {report['n_tokens']}")
    for actor, head in sorted(report["heads"].items()):
        click.echo(f"  head[{actor}]  : {head}")
    if report["errors"]:
        click.echo(f"\nTOKEN RECORD FORGERIES DETECTED ({len(report['errors'])}):", err=True)
        for name, errs in report["errors"].items():
            click.echo(f"  {name}: {errs[0]}", err=True)
        return True
    click.echo("  all sequences verified clean.")
    return False


@cli.command()
@click.option("--dir", "workspace", default=".", show_default=True, help="Workspace directory.")
@click.option("--eps", multiple=True, type=float, default=(0.1, 0.2), show_default=True,
              help="Clip half-widths to report clipped fractions for (repeatable).")
@click.option("--json", "json_out", type=click.Path(), default=None, help="Write the full JSON report here.")
@click.option("--markdown", "md_out", type=click.Path(), default=None, help="Write the markdown table here.")
def report(workspace: str, eps: tuple[float, ...], json_out: str | None, md_out: str | None):
    """Staleness-vs-mismatch decomposition of the token record (lag 0 = engine floor)."""
    import json as _json

    from martingale.diagnostics import decompose, render_markdown
    from martingale.record import TokenLedger

    ws = Path(workspace)
    if not (ws / "tokens.db").exists():
        click.echo(f"No token record at {ws / 'tokens.db'}", err=True)
        sys.exit(1)
    d = decompose(TokenLedger(ws / "tokens.db"), eps=tuple(eps))
    md = render_markdown(d)
    click.echo(md, nl=False)
    if json_out:
        Path(json_out).write_text(_json.dumps(d, indent=1))
        click.echo(f"wrote {json_out}")
    if md_out:
        Path(md_out).write_text(md)
        click.echo(f"wrote {md_out}")


@cli.command()
@click.option("--dir", "workspace", default=None, help="Workspace to write (default: a temporary directory).")
def demo(workspace: str | None):
    """Record, verify and report a synthetic run in under a minute (no GPU, no torch)."""
    import tempfile

    from martingale.demo import run_demo

    if workspace is None:
        with tempfile.TemporaryDirectory(prefix="martingale_demo_") as tmp:
            run_demo(Path(tmp), echo=click.echo)
    else:
        run_demo(Path(workspace), echo=click.echo)
        click.echo(f"workspace kept at {Path(workspace).resolve()}: try `martingale verify --dir` and `martingale report --dir`")
