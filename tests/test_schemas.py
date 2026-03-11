import pytest
from athena.schemas.case import CaseFile
from athena.schemas.simulation import SimulationConfig
from athena.schemas.state import RunParams, ValidationResult


class TestCaseFile:
    def test_loads_valid_case(self, sample_case_data):
        case = CaseFile(**sample_case_data)
        assert case.case_id == "gdp-milano-17928-2025"
        assert len(case.parties) == 2
        assert len(case.evidence) == 2

    def test_rejects_missing_case_id(self, sample_case_data):
        del sample_case_data["case_id"]
        with pytest.raises(Exception):
            CaseFile(**sample_case_data)

    def test_extract_all_valid_ids(self, sample_case_data):
        case = CaseFile(**sample_case_data)
        ids = case.extract_all_ids()
        assert "F1" in ids
        assert "DOC1" in ids
        assert "D1" in ids
        assert "art_143_cds" in ids
        assert "cass_16515_2005" in ids
        assert "NONEXISTENT" not in ids

    def test_n_party_seed_arguments(self, sample_case_data):
        case = CaseFile(**sample_case_data)
        assert "opponente" in case.seed_arguments.by_party
        assert "comune_milano" in case.seed_arguments.by_party
        assert case.seed_arguments.by_party["opponente"][0].id == "SEED_ARG1"

    def test_n_party_disputed_facts(self, sample_case_data):
        case = CaseFile(**sample_case_data)
        df = case.facts.disputed[0]
        assert "opponente" in df.positions
        assert "comune_milano" in df.positions


class TestSimulationConfig:
    def test_loads_valid_config(self):
        config = SimulationConfig(
            case_ref="gdp-milano-17928-2025",
            language="it",
            judge_profiles=[
                {
                    "id": "formalista_pro_cass",
                    "party_id": "judge",
                    "role_type": "adjudicator",
                    "parameters": {
                        "jurisprudential_orientation": "follows_cassazione",
                        "formalism": "high",
                    },
                }
            ],
            party_profiles={
                "opponente": [
                    {
                        "id": "aggressivo",
                        "party_id": "opponente",
                        "role_type": "advocate",
                        "parameters": {"style": "Attacca frontalmente."},
                    }
                ],
            },
            temperatures={"appellant": 0.5, "respondent": 0.4, "judge": 0.3},
            runs_per_combination=5,
        )
        assert config.total_runs == 5  # 1 judge × 1 style × 5

    def test_total_runs_calculation(self):
        config = SimulationConfig(
            case_ref="test",
            language="it",
            judge_profiles=[
                {"id": "a", "party_id": "judge", "role_type": "adjudicator",
                 "parameters": {"jurisprudential_orientation": "follows_cassazione", "formalism": "high"}},
                {"id": "b", "party_id": "judge", "role_type": "adjudicator",
                 "parameters": {"jurisprudential_orientation": "distinguishes_cassazione", "formalism": "low"}},
            ],
            party_profiles={
                "opponente": [
                    {"id": "x", "party_id": "opponente", "role_type": "advocate", "parameters": {"style": "s1"}},
                    {"id": "y", "party_id": "opponente", "role_type": "advocate", "parameters": {"style": "s2"}},
                    {"id": "z", "party_id": "opponente", "role_type": "advocate", "parameters": {"style": "s3"}},
                ],
            },
            temperatures={"appellant": 0.5, "respondent": 0.4, "judge": 0.3},
            runs_per_combination=5,
        )
        assert config.total_runs == 30  # 2 × 3 × 5

    def test_multi_party_total_runs(self):
        """N-party: 2 judges × 2 appellants × 3 co-defendants × 2 runs = 24."""
        config = SimulationConfig(
            case_ref="test",
            language="it",
            judge_profiles=[
                {"id": "j1", "party_id": "judge", "role_type": "adjudicator", "parameters": {}},
                {"id": "j2", "party_id": "judge", "role_type": "adjudicator", "parameters": {}},
            ],
            party_profiles={
                "appellant": [
                    {"id": "a1", "party_id": "appellant", "role_type": "advocate", "parameters": {}},
                    {"id": "a2", "party_id": "appellant", "role_type": "advocate", "parameters": {}},
                ],
                "co_defendant": [
                    {"id": "c1", "party_id": "co_defendant", "role_type": "advocate", "parameters": {}},
                    {"id": "c2", "party_id": "co_defendant", "role_type": "advocate", "parameters": {}},
                    {"id": "c3", "party_id": "co_defendant", "role_type": "advocate", "parameters": {}},
                ],
            },
            temperatures={"appellant": 0.5, "co_defendant": 0.4, "judge": 0.3},
            runs_per_combination=2,
        )
        assert config.total_runs == 24  # 2 × 2 × 3 × 2


class TestValidationResult:
    def test_valid_result(self):
        v = ValidationResult(valid=True, errors=[], warnings=["test warning"])
        assert v.valid is True

    def test_invalid_with_errors(self):
        v = ValidationResult(valid=False, errors=["missing ID"], warnings=[])
        assert v.valid is False
