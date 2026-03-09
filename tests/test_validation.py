# tests/test_validation.py
import pytest
from athena.simulation.validation import validate_agent_output
from athena.schemas.case import CaseFile


class TestValidateAppellant:
    def test_valid_output_passes(self, sample_case_data, sample_appellant_brief):
        case = CaseFile(**sample_case_data)
        result = validate_agent_output(
            output=sample_appellant_brief,
            agent_role="appellant",
            case=case,
        )
        assert result.valid is True
        assert len(result.errors) == 0

    def test_phantom_id_fails(self, sample_case_data, sample_appellant_brief):
        sample_appellant_brief["filed_brief"]["arguments"][0]["facts_referenced"] = ["F999"]
        case = CaseFile(**sample_case_data)
        result = validate_agent_output(
            output=sample_appellant_brief,
            agent_role="appellant",
            case=case,
        )
        assert result.valid is False
        assert any("F999" in e for e in result.errors)

    def test_missing_filed_brief_fails(self, sample_case_data):
        case = CaseFile(**sample_case_data)
        result = validate_agent_output(
            output={"internal_analysis": {}},
            agent_role="appellant",
            case=case,
        )
        assert result.valid is False

    def test_all_high_self_assessment_warns(self, sample_case_data, sample_appellant_brief):
        sample_appellant_brief["internal_analysis"]["strength_self_assessments"] = {"ARG1": 0.95}
        case = CaseFile(**sample_case_data)
        result = validate_agent_output(
            output=sample_appellant_brief,
            agent_role="appellant",
            case=case,
        )
        assert result.valid is True
        assert any("self_assessment" in w.lower() or "0.8" in w for w in result.warnings)


class TestValidateRespondent:
    def test_missed_argument_fails(
        self, sample_case_data, sample_appellant_brief, sample_respondent_brief
    ):
        # Add a second argument to appellant that respondent doesn't address
        sample_appellant_brief["filed_brief"]["arguments"].append({
            "id": "ARG2",
            "type": "new",
            "derived_from": None,
            "claim": "Test",
            "legal_reasoning": "Test",
            "norm_text_cited": ["art_143_cds"],
            "facts_referenced": ["F1"],
            "evidence_cited": ["DOC1"],
            "precedents_addressed": [],
            "supports": None,
        })
        case = CaseFile(**sample_case_data)
        result = validate_agent_output(
            output=sample_respondent_brief,
            agent_role="respondent",
            case=case,
            appellant_brief=sample_appellant_brief,
        )
        assert result.valid is False
        assert any("ARG2" in e for e in result.errors)


class TestValidateJudge:
    def test_missed_evaluation_fails(
        self, sample_case_data, sample_appellant_brief, sample_respondent_brief
    ):
        judge_output = {
            "preliminary_objections_ruling": [],
            "case_reaches_merits": True,
            "argument_evaluation": [],  # Empty — should fail
            "precedent_analysis": {},
            "verdict": {
                "qualification_correct": True,
                "qualification_reasoning": "Test",
                "costs_ruling": "test",
            },
            "reasoning": "Test",
            "gaps": [],
        }
        case = CaseFile(**sample_case_data)
        result = validate_agent_output(
            output=judge_output,
            agent_role="judge",
            case=case,
            appellant_brief=sample_appellant_brief,
            respondent_brief=sample_respondent_brief,
        )
        assert result.valid is False
        assert any("non valutati" in e.lower() or "not evaluated" in e.lower() for e in result.errors)
