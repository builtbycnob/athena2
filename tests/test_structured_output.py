# tests/test_structured_output.py
"""Tests for JSON Schema definitions used in oMLX structured output."""

import pytest
import jsonschema

from athena.schemas.structured_output import (
    AGENT_SCHEMAS,
    APPELLANT_SCHEMA,
    RESPONDENT_SCHEMA,
    JUDGE_SCHEMA,
)


class TestSchemasWellFormed:
    """Validate that each schema is valid JSON Schema."""

    @pytest.mark.parametrize("name", ["appellant", "respondent", "judge"])
    def test_schema_is_valid_json_schema(self, name):
        schema = AGENT_SCHEMAS[name]
        # jsonschema.validators.validator_for resolves the meta-schema
        validator_cls = jsonschema.validators.validator_for(schema)
        validator_cls.check_schema(schema)

    def test_agent_schemas_has_expected_entries(self):
        expected = {"appellant", "respondent", "judge",
                    "advocate_filing", "advocate_response", "adjudicator"}
        assert set(AGENT_SCHEMAS.keys()) == expected


class TestAppellantSchema:
    def test_validates_sample_brief(self, sample_appellant_brief):
        jsonschema.validate(sample_appellant_brief, APPELLANT_SCHEMA)

    def test_rejects_missing_filed_brief(self):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"internal_analysis": {}}, APPELLANT_SCHEMA)

    def test_rejects_wrong_argument_type_enum(self, sample_appellant_brief):
        bad = _deep_copy(sample_appellant_brief)
        bad["filed_brief"]["arguments"][0]["type"] = "invalid"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, APPELLANT_SCHEMA)

    def test_rejects_empty_arguments(self, sample_appellant_brief):
        bad = _deep_copy(sample_appellant_brief)
        bad["filed_brief"]["arguments"] = []
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, APPELLANT_SCHEMA)

    def test_rejects_wrong_strategy_enum(self, sample_appellant_brief):
        bad = _deep_copy(sample_appellant_brief)
        bad["filed_brief"]["arguments"][0]["precedents_addressed"][0]["strategy"] = "nope"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, APPELLANT_SCHEMA)


class TestRespondentSchema:
    def test_validates_sample_brief(self, sample_respondent_brief):
        jsonschema.validate(sample_respondent_brief, RESPONDENT_SCHEMA)

    def test_rejects_wrong_counter_strategy(self, sample_respondent_brief):
        bad = _deep_copy(sample_respondent_brief)
        bad["filed_brief"]["responses_to_opponent"][0]["counter_strategy"] = "nope"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, RESPONDENT_SCHEMA)

    def test_rejects_missing_requests(self, sample_respondent_brief):
        bad = _deep_copy(sample_respondent_brief)
        del bad["filed_brief"]["requests"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, RESPONDENT_SCHEMA)


class TestJudgeSchema:
    def test_validates_sample_decision(self):
        decision = {
            "preliminary_objections_ruling": [],
            "case_reaches_merits": True,
            "argument_evaluation": [
                {
                    "argument_id": "ARG1",
                    "party": "appellant",
                    "persuasiveness": 0.7,
                    "strengths": "Forte",
                    "weaknesses": "Debole",
                    "determinative": True,
                }
            ],
            "precedent_analysis": {
                "cass_16515_2005": {
                    "followed": False,
                    "distinguished": True,
                    "reasoning": "Distinguibile.",
                }
            },
            "verdict": {
                "qualification_correct": False,
                "qualification_reasoning": "Errata.",
                "if_incorrect": {
                    "consequence": "reclassification",
                    "consequence_reasoning": "Va riqualificata.",
                    "applied_norm": "artt. 6-7 CdS",
                    "sanction_determined": 87,
                    "points_deducted": 0,
                },
                "costs_ruling": "a carico del Comune",
            },
            "reasoning": "Motivazione.",
            "gaps": [],
        }
        jsonschema.validate(decision, JUDGE_SCHEMA)

    def test_verdict_if_incorrect_nullable(self):
        decision = {
            "preliminary_objections_ruling": [],
            "case_reaches_merits": True,
            "argument_evaluation": [
                {
                    "argument_id": "ARG1",
                    "party": "respondent",
                    "persuasiveness": 0.5,
                    "strengths": "OK",
                    "weaknesses": "Debole",
                    "determinative": False,
                }
            ],
            "precedent_analysis": {},
            "verdict": {
                "qualification_correct": True,
                "qualification_reasoning": "Corretta.",
                "if_incorrect": None,
                "costs_ruling": "compensate",
            },
            "reasoning": "Motivazione.",
            "gaps": [],
        }
        jsonschema.validate(decision, JUDGE_SCHEMA)

    def test_rejects_invalid_party_enum(self):
        bad = {
            "preliminary_objections_ruling": [],
            "case_reaches_merits": True,
            "argument_evaluation": [
                {
                    "argument_id": "ARG1",
                    "party": "witness",
                    "persuasiveness": 0.5,
                    "strengths": "OK",
                    "weaknesses": "Debole",
                    "determinative": False,
                }
            ],
            "precedent_analysis": {},
            "verdict": {
                "qualification_correct": True,
                "qualification_reasoning": "OK.",
                "if_incorrect": None,
                "costs_ruling": "compensate",
            },
            "reasoning": "Motivazione.",
            "gaps": [],
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, JUDGE_SCHEMA)

    def test_rejects_persuasiveness_out_of_range(self):
        bad = {
            "preliminary_objections_ruling": [],
            "case_reaches_merits": True,
            "argument_evaluation": [
                {
                    "argument_id": "ARG1",
                    "party": "appellant",
                    "persuasiveness": 1.5,
                    "strengths": "OK",
                    "weaknesses": "Debole",
                    "determinative": False,
                }
            ],
            "precedent_analysis": {},
            "verdict": {
                "qualification_correct": True,
                "qualification_reasoning": "OK.",
                "if_incorrect": None,
                "costs_ruling": "compensate",
            },
            "reasoning": "Motivazione.",
            "gaps": [],
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, JUDGE_SCHEMA)


def _deep_copy(d):
    """Simple deep copy for nested dicts/lists."""
    import copy
    return copy.deepcopy(d)
