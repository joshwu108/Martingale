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

    from martingale.store.sqlite import SQLiteRevisionStore, SQLiteLedger
    SQLiteRevisionStore(ws / "revisions.db")
    store = SQLiteRevisionStore(ws / "revisions.db")
    SQLiteLedger(ws / "ledger.db", store)

    click.echo(f"Initialised martingale workspace at {ws.resolve()}")
    click.echo(f"  revisions : {ws / 'revisions.db'}")
    click.echo(f"  ledger    : {ws / 'ledger.db'}")


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

    from martingale.store.sqlite import SQLiteRevisionStore, SQLiteLedger
    store = SQLiteRevisionStore(rev_db)
    ledger = SQLiteLedger(led_db, store)

    n_revisions = sum(1 for _ in store.all_digests())
    n_trajectories = sum(1 for _ in ledger.all_trajectories())

    click.echo(f"martingale workspace: {ws.resolve()}")
    click.echo(f"  revisions   : {n_revisions}")
    click.echo(f"  trajectories: {n_trajectories}")


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

    from martingale.store.sqlite import SQLiteRevisionStore, SQLiteLedger
    from martingale.checker_daemon import CheckerDaemon, CheckerConfig
    from martingale.server import create_app

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
def verify(workspace: str, seed: str):
    """Run the independent checker over the entire ledger."""
    import json
    import tempfile
    ws = Path(workspace)

    from martingale.store.sqlite import SQLiteRevisionStore, SQLiteLedger
    store = SQLiteRevisionStore(ws / "revisions.db")
    ledger = SQLiteLedger(ws / "ledger.db", store)

    seed_bytes = seed.encode("utf-8")

    # Export to file-based format for checker
    with tempfile.TemporaryDirectory(prefix="martingale_verify_") as tmp:
        tmp_path = Path(tmp)
        ledger_dir = tmp_path / "ledger"
        store_dir = tmp_path / "store"
        ledger.export_for_checker(ledger_dir, store_dir)

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from checker.verify import verify_ledger
        report = verify_ledger(ledger_dir, store_dir, seed=seed_bytes)

    click.echo(f"Verification complete:")
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
