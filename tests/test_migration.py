# tests/test_migration.py
"""Tests for v1 → v2 migration of case and simulation configs."""

import pytest
from athena.cli import migrate_case_v1
from athena.schemas.simulation import migrate_simulation_v1
from athena.schemas.case import CaseFile
from athena.schemas.simulation import SimulationConfig


class TestMigrateCaseV1:
    def test_migrates_seed_arguments(self):
        old = {
            "parties": [
                {"id": "opponente", "role": "appellant"},
                {"id": "comune_milano", "role": "respondent"},
            ],
            "seed_arguments": {
                "appellant": [{"id": "SEED1", "claim": "test", "direction": "d", "references_facts": []}],
                "respondent": [{"id": "SEED2", "claim": "test2", "direction": "d2", "references_facts": []}],
            },
            "facts": {"undisputed": [], "disputed": []},
        }
        result = migrate_case_v1(old)
        assert "by_party" in result["seed_arguments"]
        assert "opponente" in result["seed_arguments"]["by_party"]
        assert "comune_milano" in result["seed_arguments"]["by_party"]
        assert result["seed_arguments"]["by_party"]["opponente"][0]["id"] == "SEED1"

    def test_migrates_disputed_facts(self):
        old = {
            "parties": [
                {"id": "opponente", "role": "appellant"},
                {"id": "comune_milano", "role": "respondent"},
            ],
            "seed_arguments": {"by_party": {}},
            "facts": {
                "undisputed": [],
                "disputed": [
                    {
                        "id": "D1",
                        "description": "test",
                        "appellant_position": "pos_a",
                        "respondent_position": "pos_r",
                        "depends_on_facts": [],
                    }
                ],
            },
        }
        result = migrate_case_v1(old)
        df = result["facts"]["disputed"][0]
        assert "positions" in df
        assert df["positions"]["opponente"] == "pos_a"
        assert df["positions"]["comune_milano"] == "pos_r"
        assert "appellant_position" not in df

    def test_already_new_format_unchanged(self):
        new = {
            "parties": [],
            "seed_arguments": {"by_party": {"x": []}},
            "facts": {"undisputed": [], "disputed": []},
        }
        result = migrate_case_v1(new)
        assert result["seed_arguments"]["by_party"] == {"x": []}

    def test_full_case_loads_after_migration(self):
        """Old-format case data loads into CaseFile after migration."""
        old = _make_old_case_data()
        migrated = migrate_case_v1(old)
        case = CaseFile(**migrated)
        assert case.case_id == "test-001"
        assert "opponente" in case.seed_arguments.by_party


class TestMigrateSimulationV1:
    def test_migrates_appellant_profiles(self):
        old = {
            "case_ref": "test",
            "language": "it",
            "judge_profiles": [
                {"id": "j1", "jurisprudential_orientation": "follows_cassazione", "formalism": "high"},
            ],
            "appellant_profiles": [
                {"id": "agg", "style": "aggressive"},
            ],
            "temperature": {"appellant": 0.5, "respondent": 0.4, "judge": 0.3},
            "runs_per_combination": 1,
        }
        result = migrate_simulation_v1(old)
        assert "party_profiles" in result
        assert "opponente" in result["party_profiles"]
        assert result["party_profiles"]["opponente"][0]["parameters"]["style"] == "aggressive"
        assert "temperatures" in result
        assert "appellant_profiles" not in result

    def test_migrates_judge_profiles(self):
        old = {
            "case_ref": "test",
            "language": "it",
            "judge_profiles": [
                {"id": "j1", "jurisprudential_orientation": "follows_cassazione", "formalism": "high"},
            ],
            "appellant_profiles": [{"id": "a1", "style": "s1"}],
            "temperature": {"appellant": 0.5, "respondent": 0.4, "judge": 0.3},
            "runs_per_combination": 1,
        }
        result = migrate_simulation_v1(old)
        jp = result["judge_profiles"][0]
        assert jp["role_type"] == "adjudicator"
        assert jp["parameters"]["jurisprudential_orientation"] == "follows_cassazione"

    def test_already_new_format_unchanged(self):
        new = {
            "case_ref": "test",
            "language": "it",
            "judge_profiles": [{"id": "j1", "party_id": "judge", "role_type": "adjudicator", "parameters": {}}],
            "party_profiles": {"p1": []},
            "temperatures": {},
            "runs_per_combination": 1,
        }
        result = migrate_simulation_v1(new)
        assert result is new

    def test_full_sim_loads_after_migration(self):
        old = {
            "case_ref": "test",
            "language": "it",
            "judge_profiles": [
                {"id": "j1", "jurisprudential_orientation": "follows_cassazione", "formalism": "high"},
            ],
            "appellant_profiles": [{"id": "a1", "style": "s1"}],
            "temperature": {"appellant": 0.5, "respondent": 0.4, "judge": 0.3},
            "runs_per_combination": 5,
        }
        migrated = migrate_simulation_v1(old)
        config = SimulationConfig(**migrated)
        assert config.total_runs == 5


def _make_old_case_data():
    """Build old-format case data for testing migration."""
    return {
        "case_id": "test-001",
        "jurisdiction": {
            "country": "IT",
            "court": "giudice_di_pace",
            "venue": "Milano",
            "applicable_law": [],
            "key_precedents": [],
            "procedural_rules": {"rite": "test", "phases": [], "allowed_moves": {}},
        },
        "parties": [
            {"id": "opponente", "role": "appellant", "type": "persona_fisica",
             "objectives": {"primary": "test", "subordinate": "test"}},
            {"id": "comune_milano", "role": "respondent", "type": "pa",
             "objectives": {"primary": "test", "subordinate": "test"}},
        ],
        "stakes": {
            "current_sanction": {"norm": "test", "fine_range": [100, 200], "points_deducted": 0},
            "alternative_sanction": {"norm": "test", "fine_range": [50, 100], "points_deducted": 0},
            "litigation_cost_estimate": 500,
        },
        "evidence": [],
        "facts": {
            "undisputed": [],
            "disputed": [
                {
                    "id": "D1",
                    "description": "test",
                    "appellant_position": "pos_a",
                    "respondent_position": "pos_r",
                    "depends_on_facts": [],
                }
            ],
        },
        "legal_texts": [],
        "seed_arguments": {
            "appellant": [{"id": "SA1", "claim": "c", "direction": "d", "references_facts": []}],
            "respondent": [{"id": "SR1", "claim": "c", "direction": "d", "references_facts": []}],
        },
        "key_precedents": [],
        "timeline": [],
    }
