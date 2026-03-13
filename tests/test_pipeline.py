# tests/test_pipeline.py
"""Tests for the extracted pipeline module (athena.api.pipeline)."""

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from athena.api.models import PipelineOptions, PipelineResult, ProgressEvent
from athena.api.pipeline import (
    prepare_case_data,
    prepare_sim_config,
    run_pipeline,
    write_pipeline_outputs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_CASE = {
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
}

MINIMAL_SIM_CONFIG = {
    "case_ref": "test-001",
    "language": "it",
    "judge_profiles": [
        {"id": "j1", "jurisprudential_orientation": "follows_cassazione", "formalism": "high"},
    ],
    "appellant_profiles": [
        {"id": "pp1", "style": "aggressive"},
    ],
    "temperature": {"appellant": 0.7, "respondent": 0.3, "judge": 0.2},
    "runs_per_combination": 1,
}


def _make_pipeline_result(**overrides) -> PipelineResult:
    defaults = dict(
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
    defaults.update(overrides)
    return PipelineResult(**defaults)


# ---------------------------------------------------------------------------
# prepare_case_data
# ---------------------------------------------------------------------------


class TestPrepareCaseData:
    """Tests for prepare_case_data."""

    @patch("athena.cli.migrate_case_v1", side_effect=lambda d: d)
    def test_unwraps_case_key(self, _mock_migrate):
        raw = {"case": {"case_id": "c1", "parties": []}}
        result = prepare_case_data(raw)
        assert result["case_id"] == "c1"

    @patch("athena.cli.migrate_case_v1", side_effect=lambda d: d)
    def test_id_to_case_id(self, _mock_migrate):
        raw = {"id": "old-id", "parties": []}
        result = prepare_case_data(raw)
        assert result["case_id"] == "old-id"
        assert "id" not in result

    @patch("athena.cli.migrate_case_v1", side_effect=lambda d: d)
    def test_promotes_key_precedents(self, _mock_migrate):
        raw = {
            "case_id": "c1",
            "jurisdiction": {"country": "CH", "key_precedents": [{"id": "kp1"}]},
            "parties": [],
        }
        result = prepare_case_data(raw)
        assert result["key_precedents"] == [{"id": "kp1"}]

    @patch("athena.cli.migrate_case_v1")
    def test_migrates_v1(self, mock_migrate):
        mock_migrate.return_value = {"case_id": "migrated", "parties": []}
        raw = {"case_id": "c1", "parties": []}
        result = prepare_case_data(raw)
        mock_migrate.assert_called_once()
        assert result["case_id"] == "migrated"


# ---------------------------------------------------------------------------
# prepare_sim_config
# ---------------------------------------------------------------------------


class TestPrepareSimConfig:
    def test_unwraps_simulation_key(self):
        raw = {"simulation": MINIMAL_SIM_CONFIG}
        result = prepare_sim_config(raw)
        assert "judge_profiles" in result
        assert "party_profiles" in result

    def test_validates_and_roundtrips(self):
        result = prepare_sim_config(MINIMAL_SIM_CONFIG)
        assert result["runs_per_combination"] == 1

    def test_invalid_config_raises(self):
        from pydantic import ValidationError

        with pytest.raises((ValidationError, TypeError)):
            prepare_sim_config({"runs_per_combination": "not-an-int"})


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """Tests for run_pipeline with fully mocked internal calls."""

    PATCHES = {
        "athena.simulation.orchestrator.run_monte_carlo": [
            {"judge_decision": {"verdict": {}}}
        ],
        "athena.simulation.aggregator.aggregate_results": {
            "outcomes": {},
            "arguments": {},
        },
        "athena.output.table.format_probability_table": "| table |",
        "athena.output.decision_tree.generate_decision_tree": "tree",
        "athena.output.memo.generate_strategic_memo": "# Memo",
        "athena.agents.llm.get_stats": {
            "calls": 1,
            "total_tokens": 100,
            "total_time": 10,
            "avg_tok_s": 10.0,
        },
    }

    def _apply_patches(self):
        """Return a list of (patcher, mock) pairs — caller must stop them."""
        active = []
        for target, return_value in self.PATCHES.items():
            p = patch(target, return_value=return_value)
            m = p.start()
            active.append(p)
        return active

    def test_basic(self):
        patchers = self._apply_patches()
        try:
            sim = prepare_sim_config(MINIMAL_SIM_CONFIG)
            result = run_pipeline(
                MINIMAL_CASE,
                sim,
                PipelineOptions(skip_meta_agents=True, skip_game_theory=True),
            )
            assert isinstance(result, PipelineResult)
            assert result.case_id == "test-001"
            assert result.memo == "# Memo"
            assert result.table_md == "| table |"
            assert result.tree_txt == "tree"
            assert result.stats["calls"] == 1
            assert result.started_at is not None
            assert result.completed_at is not None
        finally:
            for p in patchers:
                p.stop()

    def test_progress_callback(self):
        patchers = self._apply_patches()
        try:
            events: list[ProgressEvent] = []
            sim = prepare_sim_config(MINIMAL_SIM_CONFIG)
            run_pipeline(
                MINIMAL_CASE,
                sim,
                PipelineOptions(skip_meta_agents=True, skip_game_theory=True),
                progress_callback=lambda e: events.append(e),
            )
            stages = [e.stage for e in events]
            assert "simulation" in stages
            assert "aggregation" in stages
            assert "outputs" in stages
            assert "done" in stages
        finally:
            for p in patchers:
                p.stop()

    def test_meta_agent_failure_continues(self):
        patchers = self._apply_patches()
        # Also patch meta-agents to raise
        meta_patch = patch(
            "athena.agents.meta_agents.run_red_team",
            side_effect=RuntimeError("boom"),
        )
        irac_patch = patch(
            "athena.agents.meta_agents.run_irac_extraction",
            side_effect=RuntimeError("boom"),
        )
        meta_patch.start()
        irac_patch.start()
        patchers.extend([meta_patch, irac_patch])
        try:
            sim = prepare_sim_config(MINIMAL_SIM_CONFIG)
            result = run_pipeline(
                MINIMAL_CASE,
                sim,
                PipelineOptions(skip_meta_agents=False, skip_game_theory=True),
            )
            # Pipeline should still succeed
            assert isinstance(result, PipelineResult)
            assert result.red_team is None
            assert result.irac is None
        finally:
            for p in patchers:
                p.stop()


# ---------------------------------------------------------------------------
# write_pipeline_outputs
# ---------------------------------------------------------------------------


class TestWritePipelineOutputs:
    def test_writes_core_files(self):
        result = _make_pipeline_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            written = write_pipeline_outputs(result, tmpdir)
            filenames = [os.path.basename(p) for p in written]
            assert "probability_table.md" in filenames
            assert "decision_tree.txt" in filenames
            assert "strategic_memo.md" in filenames
            assert "raw_results.json" in filenames
            for path in written:
                assert os.path.isfile(path)

    def test_writes_optional_files_when_present(self):
        result = _make_pipeline_result(
            game_analysis={"batna": 100},
            gt_summary_md="## GT Summary",
            red_team={"vulnerabilities": []},
            game_theorist={"analysis": "ok"},
            irac={"irac_analyses": [{"issue": "x"}]},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            written = write_pipeline_outputs(result, tmpdir)
            filenames = [os.path.basename(p) for p in written]
            assert "game_theory.json" in filenames
            assert "game_theory_summary.md" in filenames
            assert "red_team.json" in filenames
            assert "game_theorist_agent.json" in filenames
            assert "irac_analysis.json" in filenames

    def test_skips_optional_files_when_absent(self):
        result = _make_pipeline_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            written = write_pipeline_outputs(result, tmpdir)
            filenames = [os.path.basename(p) for p in written]
            assert "game_theory.json" not in filenames
            assert "red_team.json" not in filenames
            assert "irac_analysis.json" not in filenames
