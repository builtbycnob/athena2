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


class TestSimulationConfig:
    def test_loads_valid_config(self):
        config = SimulationConfig(
            case_ref="gdp-milano-17928-2025",
            language="it",
            judge_profiles=[
                {
                    "id": "formalista_pro_cass",
                    "jurisprudential_orientation": "follows_cassazione",
                    "formalism": "high",
                }
            ],
            appellant_profiles=[
                {"id": "aggressivo", "style": "Attacca frontalmente."}
            ],
            temperature={"appellant": 0.5, "respondent": 0.4, "judge": 0.3},
            runs_per_combination=5,
        )
        assert config.total_runs == 5  # 1 judge × 1 style × 5

    def test_total_runs_calculation(self):
        config = SimulationConfig(
            case_ref="test",
            language="it",
            judge_profiles=[{"id": "a", "jurisprudential_orientation": "follows_cassazione", "formalism": "high"},
                           {"id": "b", "jurisprudential_orientation": "distinguishes_cassazione", "formalism": "low"}],
            appellant_profiles=[{"id": "x", "style": "s1"}, {"id": "y", "style": "s2"}, {"id": "z", "style": "s3"}],
            temperature={"appellant": 0.5, "respondent": 0.4, "judge": 0.3},
            runs_per_combination=5,
        )
        assert config.total_runs == 30  # 2 × 3 × 5


class TestValidationResult:
    def test_valid_result(self):
        v = ValidationResult(valid=True, errors=[], warnings=["test warning"])
        assert v.valid is True

    def test_invalid_with_errors(self):
        v = ValidationResult(valid=False, errors=["missing ID"], warnings=[])
        assert v.valid is False
