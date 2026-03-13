# tests/test_api.py
"""Tests for the ATHENA FastAPI application (athena.api.app)."""

import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="FastAPI not installed (pip install athena[api])")
pytest.importorskip("sse_starlette", reason="sse-starlette not installed (pip install athena[api])")

from athena.api.models import PipelineOptions, PipelineResult, RunStatus
from athena.api import registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_RUN_BODY = {
    "case_data": {
        "case_id": "test-001",
        "parties": [
            {"id": "p1", "role": "appellant"},
            {"id": "p2", "role": "respondent"},
        ],
        "facts": {"undisputed": [], "disputed": []},
        "evidence": [],
        "legal_texts": [],
        "key_precedents": [],
        "seed_arguments": {"by_party": {"p1": [], "p2": []}},
    },
    "sim_config": {
        "judge_profiles": [{"id": "j1"}],
        "party_profiles": {"p1": [{"id": "pp1"}], "p2": [{"id": "pp2"}]},
        "runs_per_combination": 1,
    },
}


def _fake_pipeline_result() -> PipelineResult:
    return PipelineResult(
        case_id="test-001",
        results=[{"judge_decision": {"verdict": {}}}],
        aggregated={"outcomes": {}, "arguments": {}},
        memo="# Memo",
        table_md="| table |",
        tree_txt="tree",
        stats={"calls": 1, "total_tokens": 100, "total_time": 10, "avg_tok_s": 10.0},
        started_at=datetime(2026, 1, 1),
        completed_at=datetime(2026, 1, 1, 0, 1),
    )


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the in-memory run registry between tests."""
    registry.reset()
    yield
    registry.reset()


@pytest.fixture()
def client():
    """Create a TestClient for the ATHENA API app."""
    from fastapi.testclient import TestClient
    from athena.api.app import create_app

    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_status(self, client):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=MagicMock(get=AsyncMock(return_value=mock_resp)))
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_ctx

            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "status" in data
            assert data["status"] == "ok"
            assert "process" in data
            assert data["process"] == "alive"
            assert "omlx" in data


# ---------------------------------------------------------------------------
# POST /runs
# ---------------------------------------------------------------------------


class TestCreateRun:
    def test_returns_202(self, client):
        # Mock the pipeline so the background task completes instantly
        with patch(
            "athena.api.app._run_pipeline_async",
            new_callable=lambda: lambda: AsyncMock(),
        ):
            resp = client.post("/runs", json=VALID_RUN_BODY)
            assert resp.status_code == 202
            data = resp.json()
            assert "run_id" in data
            assert data["status"] == "pending"

    def test_invalid_request_422(self, client):
        resp = client.post("/runs", json={"bad": "data"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /runs/{run_id}
# ---------------------------------------------------------------------------


class TestGetRun:
    def test_not_found_404(self, client):
        resp = client.get("/runs/nonexistent")
        assert resp.status_code == 404

    def test_lifecycle(self, client):
        # Manually create a run and mark it completed
        run_id = registry.create_run()
        registry.mark_running(run_id)
        registry.mark_completed(run_id, _fake_pipeline_result())

        resp = client.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert data["status"] == "completed"
        assert data["result"] is not None
        assert data["result"]["case_id"] == "test-001"

    def test_failed_run(self, client):
        run_id = registry.create_run()
        registry.mark_running(run_id)
        registry.mark_failed(run_id, "Something went wrong")

        resp = client.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert data["error"] == "Something went wrong"


# ---------------------------------------------------------------------------
# GET /runs
# ---------------------------------------------------------------------------


class TestListRuns:
    def test_empty(self, client):
        resp = client.get("/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_with_entries(self, client):
        run_id = registry.create_run()
        registry.mark_running(run_id)

        resp = client.get("/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["run_id"] == run_id
        assert data[0]["status"] == "running"

    def test_strips_results_by_default(self, client):
        run_id = registry.create_run()
        registry.mark_completed(run_id, _fake_pipeline_result())

        resp = client.get("/runs")
        data = resp.json()
        assert len(data) == 1
        # Result should be stripped by default
        assert data[0]["result"] is None

    def test_includes_results_when_requested(self, client):
        run_id = registry.create_run()
        registry.mark_completed(run_id, _fake_pipeline_result())

        resp = client.get("/runs?include_results=true")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["result"] is not None


# ---------------------------------------------------------------------------
# Full pipeline integration (mocked)
# ---------------------------------------------------------------------------


class TestFullRunLifecycle:
    """Test POST /runs followed by GET /runs/{id} with mocked pipeline."""

    def test_post_then_get(self, client):
        with patch("athena.api.pipeline.prepare_case_data", side_effect=lambda d: d), \
             patch("athena.api.pipeline.prepare_sim_config", side_effect=lambda d: d), \
             patch("athena.api.pipeline.run_pipeline", return_value=_fake_pipeline_result()):

            resp = client.post("/runs", json=VALID_RUN_BODY)
            assert resp.status_code == 202
            run_id = resp.json()["run_id"]

            # Give the background task time to complete
            time.sleep(0.5)

            resp = client.get(f"/runs/{run_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] in ("completed", "running", "pending")

    def test_multiple_runs(self, client):
        with patch("athena.api.pipeline.prepare_case_data", side_effect=lambda d: d), \
             patch("athena.api.pipeline.prepare_sim_config", side_effect=lambda d: d), \
             patch("athena.api.pipeline.run_pipeline", return_value=_fake_pipeline_result()):

            # Create two runs
            resp1 = client.post("/runs", json=VALID_RUN_BODY)
            resp2 = client.post("/runs", json=VALID_RUN_BODY)
            assert resp1.status_code == 202
            assert resp2.status_code == 202

            id1 = resp1.json()["run_id"]
            id2 = resp2.json()["run_id"]
            assert id1 != id2

            time.sleep(0.5)

            resp = client.get("/runs")
            data = resp.json()
            assert len(data) == 2
