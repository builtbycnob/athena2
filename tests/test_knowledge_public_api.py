# tests/test_knowledge_public_api.py
"""Tests for the knowledge module public API — no-op when KG disabled."""

import os
import pytest


class TestPublicAPINoOp:
    """Verify all public API functions are no-ops when KG disabled."""

    def test_ingest_case_noop(self, monkeypatch):
        monkeypatch.delenv("ATHENA_KG_ENABLED", raising=False)
        from athena.knowledge import ingest_case
        result = ingest_case({"case_id": "test"})
        assert result == {"nodes": 0, "edges": 0}

    def test_store_run_result_noop(self, monkeypatch):
        monkeypatch.delenv("ATHENA_KG_ENABLED", raising=False)
        from athena.knowledge import store_run_result
        result = store_run_result("test", {})
        assert result == {"nodes": 0, "edges": 0}

    def test_store_aggregation_noop(self, monkeypatch):
        monkeypatch.delenv("ATHENA_KG_ENABLED", raising=False)
        from athena.knowledge import store_aggregation
        result = store_aggregation("test", {})
        assert result == {"updated_args": 0, "updated_precs": 0}

    def test_store_game_theory_noop(self, monkeypatch):
        monkeypatch.delenv("ATHENA_KG_ENABLED", raising=False)
        from athena.knowledge import store_game_theory
        result = store_game_theory("test", None)
        assert result == {"nodes": 0, "edges": 0}

    def test_get_enrichment_noop(self, monkeypatch):
        monkeypatch.delenv("ATHENA_KG_ENABLED", raising=False)
        from athena.knowledge import get_enrichment
        result = get_enrichment("test", "judge1")
        assert result is None

    def test_get_post_analysis_noop(self, monkeypatch):
        monkeypatch.delenv("ATHENA_KG_ENABLED", raising=False)
        from athena.knowledge import get_post_analysis
        result = get_post_analysis("test")
        assert result is None


class TestIsKgEnabled:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("ATHENA_KG_ENABLED", raising=False)
        from athena.knowledge.config import is_kg_enabled
        assert not is_kg_enabled()

    def test_enabled_when_set(self, monkeypatch):
        monkeypatch.setenv("ATHENA_KG_ENABLED", "1")
        from athena.knowledge.config import is_kg_enabled
        assert is_kg_enabled()

    def test_not_enabled_with_wrong_value(self, monkeypatch):
        monkeypatch.setenv("ATHENA_KG_ENABLED", "yes")
        from athena.knowledge.config import is_kg_enabled
        assert not is_kg_enabled()
