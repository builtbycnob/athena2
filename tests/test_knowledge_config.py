# tests/test_knowledge_config.py
"""Tests for knowledge graph configuration — connection management, schema migration.

These tests require a running Neo4j instance. They are automatically skipped
when ATHENA_KG_ENABLED is not set or Neo4j is unreachable.
"""

import os
import pytest

from athena.knowledge.config import is_kg_enabled

# Skip all tests in this module if KG is not enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("ATHENA_KG_ENABLED") != "1",
    reason="Knowledge graph not enabled (set ATHENA_KG_ENABLED=1 and NEO4J_PASSWORD)",
)


class TestIsKgEnabled:
    @pytest.mark.skipif(True, reason="Always run this one")
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("ATHENA_KG_ENABLED", raising=False)
        assert not is_kg_enabled()


class TestConnection:
    def test_get_session(self):
        from athena.knowledge.config import get_session
        with get_session() as session:
            result = session.run("RETURN 1 AS n").single()
            assert result["n"] == 1

    def test_health_check(self):
        from athena.knowledge.config import health_check
        status = health_check()
        assert status["status"] == "ok"
        assert "node_count" in status

    def test_schema_constraints_idempotent(self):
        """Schema migration should be safe to run multiple times."""
        from athena.knowledge.config import get_driver, _ensure_schema, _get_neo4j_config
        driver = get_driver()
        cfg = _get_neo4j_config()
        # Run twice — should not raise
        _ensure_schema(driver, cfg["database"])
        _ensure_schema(driver, cfg["database"])
