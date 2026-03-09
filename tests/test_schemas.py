import pytest
from athena.schemas.case import CaseFile
from athena.schemas.simulation import SimulationConfig
from athena.schemas.agents import (
    AppellantBrief,
    RespondentBrief,
    JudgeDecision,
)
from athena.schemas.state import SimulationState, RunParams, ValidationResult


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


class TestAppellantBrief:
    def test_valid_brief(self, sample_appellant_brief):
        brief = AppellantBrief(**sample_appellant_brief)
        assert len(brief.filed_brief.arguments) == 1
        assert brief.filed_brief.arguments[0].id == "ARG1"

    def test_rejects_empty_arguments(self):
        with pytest.raises(Exception):
            AppellantBrief(
                filed_brief={"arguments": [], "requests": {"primary": "x", "subordinate": "y"}},
                internal_analysis={
                    "strength_self_assessments": {},
                    "key_vulnerabilities": [],
                    "strongest_point": "",
                    "gaps": [],
                },
            )


class TestRespondentBrief:
    def test_valid_brief(self, sample_respondent_brief):
        brief = RespondentBrief(**sample_respondent_brief)
        assert len(brief.filed_brief.responses_to_opponent) == 1

    def test_response_references_argument(self, sample_respondent_brief):
        brief = RespondentBrief(**sample_respondent_brief)
        assert brief.filed_brief.responses_to_opponent[0].to_argument == "ARG1"


class TestJudgeDecision:
    def test_valid_decision(self):
        decision = JudgeDecision(
            preliminary_objections_ruling=[],
            case_reaches_merits=True,
            argument_evaluation=[
                {
                    "argument_id": "ARG1",
                    "party": "appellant",
                    "persuasiveness": 0.7,
                    "strengths": "Argomento testuale forte",
                    "weaknesses": "Cassazione contraria",
                    "determinative": True,
                }
            ],
            precedent_analysis={
                "cass_16515_2005": {
                    "followed": False,
                    "distinguished": True,
                    "reasoning": "Il caso è distinguibile.",
                }
            },
            verdict={
                "qualification_correct": False,
                "qualification_reasoning": "La qualificazione è errata.",
                "if_incorrect": {
                    "consequence": "reclassification",
                    "consequence_reasoning": "Va riqualificata.",
                    "applied_norm": "artt. 6-7 CdS",
                    "sanction_determined": 87,
                    "points_deducted": 0,
                },
                "costs_ruling": "a carico del Comune",
            },
            reasoning="Motivazione completa della sentenza...",
            gaps=[],
        )
        assert decision.verdict.qualification_correct is False
        assert decision.verdict.if_incorrect.consequence == "reclassification"


class TestValidationResult:
    def test_valid_result(self):
        v = ValidationResult(valid=True, errors=[], warnings=["test warning"])
        assert v.valid is True

    def test_invalid_with_errors(self):
        v = ValidationResult(valid=False, errors=["missing ID"], warnings=[])
        assert v.valid is False
