"""Tests for the Martingale Observatory REST server.

All tests use FastAPI's TestClient (sync). The server is stateless per-request;
state lives in the SQLite store/ledger fixtures.
"""
import pytest
from fractions import Fraction
from starlette.testclient import TestClient

from martingale.store.sqlite import SQLiteRevisionStore, SQLiteLedger
from martingale.draw import draw_action
from martingale.ledger import ActionRecord, GENESIS_DIGEST
from martingale.checker_daemon import CheckerDaemon, CheckerConfig


SEED = b"test-server"


def _write_traj(ledger, store, probs=None, actor_id=0, episode_id=0, n_steps=1):
    """Write a valid trajectory to the ledger. Returns the revision."""
    if probs is None:
        probs = {0: Fraction(1, 2), 1: Fraction(1, 2)}
    rev = store.publish({0: probs})
    records = []
    prev = GENESIS_DIGEST
    for step in range(n_steps):
        draw = draw_action(probs, seed=SEED, actor_id=actor_id, episode=episode_id, step=step)
        rec = ActionRecord(
            revision_digest=rev.digest,
            state=0,
            action=draw.action,
            behavior_prob=probs[draw.action],
            draw=draw,
            env_next_state=1,
            env_reward=Fraction(1),
            prev_digest=prev,
        )
        records.append(rec)
        prev = rec.digest
    ledger.append_trajectory(actor_id=actor_id, episode_id=episode_id, records=records)
    return rev


@pytest.fixture
def client(tmp_path):
    from martingale.server import create_app
    store = SQLiteRevisionStore(tmp_path / "rev.db")
    ledger = SQLiteLedger(tmp_path / "ledger.db", store)
    _write_traj(ledger, store, episode_id=0)
    _write_traj(ledger, store, episode_id=1)
    daemon = CheckerDaemon(ledger, store, CheckerConfig(seed=SEED, halt_on_forgery=False))
    daemon.scan_once()
    app = create_app(store, ledger, daemon)
    return TestClient(app)


@pytest.fixture
def empty_client(tmp_path):
    """Client with an empty store (no revisions, no trajectories)."""
    from martingale.server import create_app
    store = SQLiteRevisionStore(tmp_path / "rev.db")
    ledger = SQLiteLedger(tmp_path / "ledger.db", store)
    app = create_app(store, ledger, daemon=None)
    return TestClient(app)


# ──────────────────────────────────────────────────────────────────────────────
# /api/status
# ──────────────────────────────────────────────────────────────────────────────

class TestStatus:
    def test_returns_200(self, client):
        assert client.get("/api/status").status_code == 200

    def test_revision_count(self, client):
        data = client.get("/api/status").json()
        assert data["revisions"] >= 1

    def test_trajectory_count(self, client):
        data = client.get("/api/status").json()
        assert data["trajectories"] == 2

    def test_has_forgeries_field(self, client):
        data = client.get("/api/status").json()
        assert "forgeries_in_last_scan" in data
        assert isinstance(data["forgeries_in_last_scan"], int)

    def test_empty_workspace(self, empty_client):
        data = empty_client.get("/api/status").json()
        assert data["revisions"] == 0
        assert data["trajectories"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# /api/revisions
# ──────────────────────────────────────────────────────────────────────────────

class TestRevisions:
    def test_returns_list(self, client):
        resp = client.get("/api/revisions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_has_digest_and_seq(self, client):
        revs = client.get("/api/revisions").json()
        assert len(revs) >= 1
        assert "digest" in revs[0]
        assert "seq" in revs[0]

    def test_seq_starts_at_zero(self, client):
        revs = client.get("/api/revisions").json()
        seqs = [r["seq"] for r in revs]
        assert min(seqs) == 0

    def test_seqs_are_consecutive(self, client):
        revs = client.get("/api/revisions").json()
        seqs = sorted(r["seq"] for r in revs)
        assert seqs == list(range(len(seqs)))

    def test_get_single_revision(self, client):
        revs = client.get("/api/revisions").json()
        digest = revs[0]["digest"]
        resp = client.get(f"/api/revisions/{digest}")
        assert resp.status_code == 200
        assert resp.json()["digest"] == digest

    def test_single_revision_has_table(self, client):
        revs = client.get("/api/revisions").json()
        digest = revs[0]["digest"]
        data = client.get(f"/api/revisions/{digest}").json()
        assert "table" in data

    def test_unknown_revision_returns_404(self, client):
        resp = client.get("/api/revisions/" + "a" * 64)
        assert resp.status_code == 404


# ──────────────────────────────────────────────────────────────────────────────
# /api/trajectories
# ──────────────────────────────────────────────────────────────────────────────

class TestTrajectories:
    def test_returns_list(self, client):
        resp = client.get("/api/trajectories")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_has_expected_fields(self, client):
        t = client.get("/api/trajectories").json()[0]
        for field in ("actor_id", "episode_id", "n_steps", "lag"):
            assert field in t, f"Missing field: {field}"

    def test_n_steps_correct(self, client):
        trajs = client.get("/api/trajectories").json()
        for t in trajs:
            assert t["n_steps"] == 1

    def test_lag_is_non_negative(self, client):
        for t in client.get("/api/trajectories").json():
            assert t["lag"] >= 0

    def test_limit_param(self, client):
        assert len(client.get("/api/trajectories?limit=1").json()) == 1

    def test_offset_param(self, client):
        all_trajs = client.get("/api/trajectories").json()
        offset_trajs = client.get("/api/trajectories?offset=1").json()
        assert len(offset_trajs) == len(all_trajs) - 1

    def test_get_single_trajectory(self, client):
        resp = client.get("/api/trajectories/0/0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["actor_id"] == 0
        assert data["episode_id"] == 0
        assert "action_records" in data

    def test_single_trajectory_action_records_have_revision(self, client):
        data = client.get("/api/trajectories/0/0").json()
        rec = data["action_records"][0]
        assert "revision_digest" in rec

    def test_unknown_trajectory_returns_404(self, client):
        assert client.get("/api/trajectories/99/99").status_code == 404


# ──────────────────────────────────────────────────────────────────────────────
# /api/staleness/histogram
# ──────────────────────────────────────────────────────────────────────────────

class TestStalenessHistogram:
    def test_returns_200(self, client):
        assert client.get("/api/staleness/histogram").status_code == 200

    def test_has_buckets_list(self, client):
        data = client.get("/api/staleness/histogram").json()
        assert "buckets" in data
        assert isinstance(data["buckets"], list)

    def test_buckets_have_lag_and_count(self, client):
        for b in client.get("/api/staleness/histogram").json()["buckets"]:
            assert "lag" in b
            assert "count" in b
            assert b["count"] >= 0

    def test_total_count_equals_n_trajectories(self, client):
        data = client.get("/api/staleness/histogram").json()
        n_trajs = client.get("/api/status").json()["trajectories"]
        total = sum(b["count"] for b in data["buckets"])
        assert total == n_trajs

    def test_empty_workspace(self, empty_client):
        data = empty_client.get("/api/staleness/histogram").json()
        assert data["buckets"] == []


# ──────────────────────────────────────────────────────────────────────────────
# /api/staleness/scaling
# ──────────────────────────────────────────────────────────────────────────────

class TestStalenessScaling:
    def test_returns_200(self, client):
        assert client.get("/api/staleness/scaling").status_code == 200

    def test_has_data_points(self, client):
        data = client.get("/api/staleness/scaling").json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_data_points_have_lag_and_is_variance(self, client):
        for pt in client.get("/api/staleness/scaling").json()["data"]:
            assert "lag" in pt
            assert "is_weight_variance" in pt
            assert "count" in pt


# ──────────────────────────────────────────────────────────────────────────────
# /api/is-weights/stats
# ──────────────────────────────────────────────────────────────────────────────

class TestISWeightStats:
    def test_returns_200(self, client):
        assert client.get("/api/is-weights/stats").status_code == 200

    def test_has_statistical_fields(self, client):
        data = client.get("/api/is-weights/stats").json()
        for field in ("mean", "p50", "p95", "p99", "n_samples"):
            assert field in data, f"Missing field: {field}"

    def test_values_are_positive(self, client):
        data = client.get("/api/is-weights/stats").json()
        assert data["mean"] > 0
        assert data["p50"] > 0

    def test_empty_workspace(self, empty_client):
        data = empty_client.get("/api/is-weights/stats").json()
        assert data["n_samples"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# /api/forgeries
# ──────────────────────────────────────────────────────────────────────────────

class TestForgeries:
    def test_returns_200(self, client):
        assert client.get("/api/forgeries").status_code == 200

    def test_has_total_checked_and_forgeries(self, client):
        data = client.get("/api/forgeries").json()
        assert "total_checked" in data
        assert "forgeries" in data

    def test_checked_equals_trajectories_after_scan(self, client):
        data = client.get("/api/forgeries").json()
        n_trajs = client.get("/api/status").json()["trajectories"]
        assert data["total_checked"] == n_trajs

    def test_no_daemon_returns_zeros(self, empty_client):
        data = empty_client.get("/api/forgeries").json()
        assert data["forgeries"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# /metrics (Prometheus)
# ──────────────────────────────────────────────────────────────────────────────

class TestPrometheusMetrics:
    def test_returns_200(self, client):
        assert client.get("/metrics").status_code == 200

    def test_contains_martingale_metrics(self, client):
        text = client.get("/metrics").text
        assert "martingale" in text

    def test_content_type_is_text(self, client):
        resp = client.get("/metrics")
        assert resp.headers["content-type"].startswith("text/plain")


# ──────────────────────────────────────────────────────────────────────────────
# / (dashboard HTML)
# ──────────────────────────────────────────────────────────────────────────────

class TestDashboard:
    def test_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_content_type_is_html(self, client):
        resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]

    def test_html_has_staleness_chart(self, client):
        assert "staleness-chart" in client.get("/").text

    def test_html_has_is_weights_chart(self, client):
        assert "is-weights-chart" in client.get("/").text

    def test_html_has_revision_timeline(self, client):
        assert "revision-timeline" in client.get("/").text

    def test_html_has_trajectories_table(self, client):
        assert "trajectories-table" in client.get("/").text

    def test_html_has_staleness_scaling_chart(self, client):
        assert "staleness-scaling-chart" in client.get("/").text
