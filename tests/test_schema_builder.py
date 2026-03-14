# tests/test_schema_builder.py
"""Tests for dynamic JSON schema builder (XGrammar enum constraints)."""

import copy
import pytest

from athena.schemas.schema_builder import build_schema_for_agent
from athena.schemas.structured_output import AGENT_SCHEMAS


# --- Test fixtures ---

SAMPLE_CASE = {
    "facts": {
        "undisputed": [{"id": "F1"}, {"id": "F2"}],
        "disputed": [{"id": "F3"}],
    },
    "evidence": [{"id": "DOC1"}, {"id": "DOC2"}],
    "legal_texts": [{"id": "norm_1"}, {"id": "norm_2"}],
    "key_precedents": [{"id": "prec_1"}, {"id": "prec_2"}],
    "parties": [
        {"id": "appellant", "role": "appellant"},
        {"id": "respondent", "role": "respondent"},
    ],
}

SAMPLE_BRIEFS = {
    "appellant": {
        "filed_brief": {
            "arguments": [{"id": "ARG1"}, {"id": "ARG2"}],
            "affirmative_defenses": [],
        }
    },
    "respondent": {
        "filed_brief": {
            "arguments": [{"id": "RARG1"}],
            "affirmative_defenses": [{"id": "DEF1"}],
        }
    },
}


# --- Appellant schema tests ---


def test_appellant_schema_injects_fact_enums():
    schema = build_schema_for_agent("appellant", SAMPLE_CASE)
    args_items = schema["properties"]["filed_brief"]["properties"]["arguments"]["items"]
    fr = args_items["properties"]["facts_referenced"]["items"]
    assert "enum" in fr
    assert sorted(fr["enum"]) == ["F1", "F2", "F3"]


def test_appellant_schema_injects_evidence_enums():
    schema = build_schema_for_agent("appellant", SAMPLE_CASE)
    args_items = schema["properties"]["filed_brief"]["properties"]["arguments"]["items"]
    ec = args_items["properties"]["evidence_cited"]["items"]
    assert "enum" in ec
    assert sorted(ec["enum"]) == ["DOC1", "DOC2"]


def test_appellant_schema_injects_norm_enums():
    schema = build_schema_for_agent("appellant", SAMPLE_CASE)
    args_items = schema["properties"]["filed_brief"]["properties"]["arguments"]["items"]
    nc = args_items["properties"]["norm_text_cited"]["items"]
    assert "enum" in nc
    assert sorted(nc["enum"]) == ["norm_1", "norm_2"]


def test_appellant_schema_injects_precedent_enums():
    schema = build_schema_for_agent("appellant", SAMPLE_CASE)
    args_items = schema["properties"]["filed_brief"]["properties"]["arguments"]["items"]
    pa = args_items["properties"]["precedents_addressed"]["items"]
    assert "enum" in pa["properties"]["id"]
    assert sorted(pa["properties"]["id"]["enum"]) == ["prec_1", "prec_2"]


# --- Respondent schema tests ---


def test_respondent_schema_injects_argument_enums():
    schema = build_schema_for_agent("respondent", SAMPLE_CASE, prior_briefs=SAMPLE_BRIEFS)
    fb = schema["properties"]["filed_brief"]["properties"]
    rto = fb["responses_to_opponent"]["items"]
    assert "enum" in rto["properties"]["to_argument"]
    assert "ARG1" in rto["properties"]["to_argument"]["enum"]
    assert "ARG2" in rto["properties"]["to_argument"]["enum"]


# --- Judge schema tests ---


def test_judge_schema_injects_argument_id_enums():
    schema = build_schema_for_agent("judge", SAMPLE_CASE, prior_briefs=SAMPLE_BRIEFS)
    ae = schema["properties"]["argument_evaluation"]["items"]
    assert "enum" in ae["properties"]["argument_id"]
    expected = sorted(["ARG1", "ARG2", "RARG1", "DEF1"])
    assert sorted(ae["properties"]["argument_id"]["enum"]) == expected


def test_judge_dynamic_minItems():
    schema = build_schema_for_agent("judge", SAMPLE_CASE, prior_briefs=SAMPLE_BRIEFS)
    ae = schema["properties"]["argument_evaluation"]
    assert ae["minItems"] == 4  # ARG1, ARG2, RARG1, DEF1


# --- Step 2 schema tests ---


def test_step2_schema_injects_error_ids():
    schema = build_schema_for_agent(
        "judge_ch_step2", SAMPLE_CASE, step1_error_count=3
    )
    ea = schema["properties"]["error_assessment"]["items"]
    assert "enum" in ea["properties"]["error_id"]
    assert ea["properties"]["error_id"]["enum"] == [0, 1, 2]


# --- Edge cases ---


def test_empty_case_no_crash():
    empty_case = {"facts": {}, "evidence": [], "legal_texts": [], "key_precedents": []}
    schema = build_schema_for_agent("appellant", empty_case)
    args_items = schema["properties"]["filed_brief"]["properties"]["arguments"]["items"]
    # No enums injected when lists are empty
    assert "enum" not in args_items["properties"]["facts_referenced"]["items"]


def test_deep_copy_no_mutation():
    original = copy.deepcopy(AGENT_SCHEMAS)
    build_schema_for_agent("appellant", SAMPLE_CASE, prior_briefs=SAMPLE_BRIEFS)
    # Original schemas must be unchanged
    assert AGENT_SCHEMAS == original


def test_schema_is_valid_json_schema():
    """Built schemas should be valid JSON Schema (basic structural check)."""
    schema = build_schema_for_agent("judge_ch_step1", SAMPLE_CASE, prior_briefs=SAMPLE_BRIEFS)
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "required" in schema
