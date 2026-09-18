"""
server.py — Martingale Observatory REST server.

Exposes the SQLite revision store and ledger over HTTP.
Serves a live dashboard at /.

Usage:
    from martingale.server import create_app
    app = create_app(store, ledger, daemon)
    # → pass to uvicorn, or use with TestClient in tests
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse

from martingale.checker_daemon import CheckerDaemon
from martingale.store.sqlite import SQLiteRevisionStore, SQLiteLedger


def create_app(
    store: SQLiteRevisionStore,
    ledger: SQLiteLedger,
    daemon: Optional[CheckerDaemon] = None,
) -> FastAPI:
    """
    Create the FastAPI application bound to the given store/ledger/daemon.

    The app is stateless per-request; all state lives in the SQLite databases.
    Passing daemon=None disables forgery reporting (returns zeros).
    """
    app = FastAPI(title="Martingale Observatory", version="0.1.0")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _seq_map() -> dict[str, int]:
        """digest → sequence number (0 = oldest)."""
        return {d: seq for seq, d, _ in store.revisions_ordered()}

    def _latest_seq() -> int:
        rows = store.revisions_ordered()
        return rows[-1][0] if rows else 0

    def _latest_digest() -> Optional[str]:
        rows = store.revisions_ordered()
        return rows[-1][1] if rows else None

    def _trajectory_lag(traj, seq_map: dict, latest: int) -> int:
        step_seqs = [seq_map.get(r.revision_digest, 0) for r in traj.action_records]
        return latest - min(step_seqs) if step_seqs else 0

    # ── /api/status ───────────────────────────────────────────────────────────

    @app.get("/api/status")
    def status():
        n_revs = len(store.revisions_ordered())
        n_trajs = sum(1 for _ in ledger.all_trajectories())
        forgeries = 0
        if daemon is not None and daemon._last_scan_metrics is not None:
            forgeries = daemon._last_scan_metrics.forgeries_detected
        return {
            "revisions": n_revs,
            "trajectories": n_trajs,
            "forgeries_in_last_scan": forgeries,
        }

    # ── /api/revisions ────────────────────────────────────────────────────────

    @app.get("/api/revisions")
    def list_revisions():
        return [
            {"seq": seq, "digest": digest, "created_at": created_at}
            for seq, digest, created_at in store.revisions_ordered()
        ]

    @app.get("/api/revisions/{digest}")
    def get_revision(digest: str):
        try:
            rev = store.get(digest)
        except KeyError:
            raise HTTPException(status_code=404, detail="Revision not found")
        seq_map = _seq_map()
        return {
            "digest": digest,
            "seq": seq_map.get(digest, -1),
            "table": {
                str(s): {str(a): str(p) for a, p in ap.items()}
                for s, ap in rev.table.items()
            },
        }

    # ── /api/trajectories ─────────────────────────────────────────────────────

    @app.get("/api/trajectories")
    def list_trajectories(limit: int = 50, offset: int = 0):
        seq_map = _seq_map()
        latest = _latest_seq()
        trajs = list(ledger.all_trajectories())
        page = trajs[offset: offset + limit]
        return [
            {
                "actor_id": t.actor_id,
                "episode_id": t.episode_id,
                "n_steps": len(t.action_records),
                "lag": _trajectory_lag(t, seq_map, latest),
            }
            for t in page
        ]

    @app.get("/api/trajectories/{actor_id}/{episode_id}")
    def get_trajectory(actor_id: int, episode_id: int):
        try:
            traj = ledger.load_trajectory(actor_id, episode_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Trajectory not found")
        return traj.to_dict()

    # ── /api/staleness/histogram ──────────────────────────────────────────────

    @app.get("/api/staleness/histogram")
    def staleness_histogram():
        seq_map = _seq_map()
        latest = _latest_seq()
        lag_counts: dict[int, int] = {}
        for t in ledger.all_trajectories():
            lag = _trajectory_lag(t, seq_map, latest)
            lag_counts[lag] = lag_counts.get(lag, 0) + 1
        return {
            "buckets": [
                {"lag": k, "count": v}
                for k, v in sorted(lag_counts.items())
            ]
        }

    # ── /api/staleness/scaling ────────────────────────────────────────────────

    @app.get("/api/staleness/scaling")
    def staleness_scaling():
        """
        Per-lag IS weight variance (T3 live analog).

        Returns [{lag, is_weight_variance, count}] for each observed lag value.
        IS weights are computed as target_prob / behavior_prob using the latest revision.
        """
        seq_map = _seq_map()
        latest = _latest_seq()
        latest_d = _latest_digest()

        # {lag: [w1, w2, ...]}
        lag_weights: dict[int, list[float]] = {}

        latest_rev = None
        if latest_d:
            try:
                latest_rev = store.get(latest_d)
            except KeyError:
                pass

        for t in ledger.all_trajectories():
            lag = _trajectory_lag(t, seq_map, latest)
            lag_weights.setdefault(lag, [])
            if latest_rev is not None:
                for r in t.action_records:
                    target = latest_rev.table.get(r.state, {}).get(r.action)
                    if target is not None and r.behavior_prob != 0:
                        w = float(target / r.behavior_prob)
                        lag_weights[lag].append(w)

        data = []
        for lag, ws in sorted(lag_weights.items()):
            if ws:
                mean_w = sum(ws) / len(ws)
                variance = sum((w - mean_w) ** 2 for w in ws) / len(ws)
            else:
                variance = 0.0
            data.append({
                "lag": lag,
                "is_weight_variance": variance,
                "count": len(ws),
            })

        return {"data": data}

    # ── /api/is-weights/stats ─────────────────────────────────────────────────

    @app.get("/api/is-weights/stats")
    def is_weight_stats():
        latest_d = _latest_digest()
        if latest_d is None:
            return {"mean": 1.0, "std": 0.0, "p50": 1.0, "p95": 1.0, "p99": 1.0, "n_samples": 0}

        try:
            latest_rev = store.get(latest_d)
        except KeyError:
            return {"mean": 1.0, "std": 0.0, "p50": 1.0, "p95": 1.0, "p99": 1.0, "n_samples": 0}

        weights = []
        for t in ledger.all_trajectories():
            for r in t.action_records:
                target = latest_rev.table.get(r.state, {}).get(r.action)
                if target is not None and r.behavior_prob != 0:
                    weights.append(float(target / r.behavior_prob))

        if not weights:
            return {"mean": 1.0, "std": 0.0, "p50": 1.0, "p95": 1.0, "p99": 1.0, "n_samples": 0}

        weights_sorted = sorted(weights)
        n = len(weights_sorted)
        avg = sum(weights_sorted) / n
        variance = sum((w - avg) ** 2 for w in weights_sorted) / n
        std = math.sqrt(variance)

        def _pct(pct: float) -> float:
            idx = min(int(n * pct), n - 1)
            return weights_sorted[idx]

        return {
            "mean": avg,
            "std": std,
            "p50": _pct(0.50),
            "p95": _pct(0.95),
            "p99": _pct(0.99),
            "n_samples": n,
        }

    # ── /api/forgeries ────────────────────────────────────────────────────────

    @app.get("/api/forgeries")
    def forgeries():
        if daemon is None:
            return {"total_checked": 0, "forgeries": 0}
        return {
            "total_checked": daemon._cumulative_checked,
            "forgeries": daemon._cumulative_forgeries,
        }

    # ── /metrics (Prometheus) ─────────────────────────────────────────────────

    @app.get("/metrics")
    def metrics():
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return PlainTextResponse(
            generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )

    # ── / (dashboard) ─────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        html_path = Path(__file__).parent / "static" / "dashboard.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text())
        return HTMLResponse(
            "<html><body><h1>Martingale Observatory</h1>"
            "<p>Dashboard not found at static/dashboard.html</p></body></html>"
        )

    return app
