# src/athena/schemas/structured_output.py
"""JSON Schema definitions for oMLX structured output (response_format).

Hand-crafted schemas with maxLength, enum, minItems/maxItems constraints
to prevent infinite generation and enforce valid outputs at decode time.
maxLength values calibrated from baseline-001 outputs (p95 x 1.3, rounded).
"""

_PRECEDENT_ADDRESS = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "maxLength": 20},
        "strategy": {"type": "string", "enum": ["distinguish", "criticize", "limit_scope"]},
        "reasoning": {"type": "string", "maxLength": 1000},
    },
    "required": ["id", "strategy", "reasoning"],
    "additionalProperties": False,
}

_ARGUMENT = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "maxLength": 10},
        "type": {"type": "string", "enum": ["derived", "new"]},
        "derived_from": {"type": ["string", "null"], "maxLength": 10},
        "claim": {"type": "string", "maxLength": 500},
        "legal_reasoning": {"type": "string", "maxLength": 2000},
        "norm_text_cited": {
            "type": "array",
            "items": {"type": "string", "maxLength": 20},
            "minItems": 0,
            "maxItems": 10,
        },
        "facts_referenced": {
            "type": "array",
            "items": {"type": "string", "maxLength": 20},
            "minItems": 0,
            "maxItems": 10,
        },
        "evidence_cited": {
            "type": "array",
            "items": {"type": "string", "maxLength": 20},
            "minItems": 0,
            "maxItems": 10,
        },
        "precedents_addressed": {
            "type": "array",
            "items": _PRECEDENT_ADDRESS,
            "minItems": 0,
            "maxItems": 5,
        },
        "supports": {"type": ["string", "null"], "maxLength": 10},
    },
    "required": [
        "id", "type", "derived_from", "claim", "legal_reasoning",
        "norm_text_cited", "facts_referenced", "evidence_cited",
        "precedents_addressed", "supports",
    ],
    "additionalProperties": False,
}

APPELLANT_SCHEMA = {
    "type": "object",
    "properties": {
        "filed_brief": {
            "type": "object",
            "properties": {
                "arguments": {
                    "type": "array",
                    "items": _ARGUMENT,
                    "minItems": 1,
                    "maxItems": 8,
                },
                "requests": {
                    "type": "object",
                    "properties": {
                        "primary": {"type": "string", "maxLength": 500},
                        "subordinate": {"type": "string", "maxLength": 500},
                    },
                    "required": ["primary", "subordinate"],
                    "additionalProperties": False,
                },
            },
            "required": ["arguments", "requests"],
            "additionalProperties": False,
        },
        "internal_analysis": {
            "type": "object",
            "properties": {
                "strength_self_assessments": {
                    "type": "object",
                    "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "key_vulnerabilities": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 500},
                    "minItems": 1,
                    "maxItems": 5,
                },
                "strongest_point": {"type": "string", "maxLength": 500},
                "gaps": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 500},
                    "minItems": 0,
                    "maxItems": 5,
                },
            },
            "required": ["strength_self_assessments", "key_vulnerabilities", "strongest_point", "gaps"],
            "additionalProperties": False,
        },
    },
    "required": ["filed_brief", "internal_analysis"],
    "additionalProperties": False,
}

# --- Respondent ---

_PRECEDENT_CITATION = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "maxLength": 20},
        "relevance": {"type": "string", "maxLength": 500},
    },
    "required": ["id", "relevance"],
    "additionalProperties": False,
}

_PRELIMINARY_OBJECTION = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "maxLength": 10},
        "type": {
            "type": "string",
            "enum": ["tardivita", "inammissibilita", "incompetenza", "difetto_legittimazione"],
        },
        "claim": {"type": "string", "maxLength": 500},
        "legal_basis": {
            "type": "array",
            "items": {"type": "string", "maxLength": 20},
            "minItems": 1,
            "maxItems": 10,
        },
        "reasoning": {"type": "string", "maxLength": 1000},
    },
    "required": ["id", "type", "claim", "legal_basis", "reasoning"],
    "additionalProperties": False,
}

_RESPONSE_TO_OPPONENT = {
    "type": "object",
    "properties": {
        "to_argument": {"type": "string", "maxLength": 10},
        "counter_strategy": {
            "type": "string",
            "enum": ["rebut", "distinguish", "concede_partially"],
        },
        "counter_reasoning": {"type": "string", "maxLength": 1500},
        "norm_text_cited": {
            "type": "array",
            "items": {"type": "string", "maxLength": 20},
            "minItems": 0,
            "maxItems": 10,
        },
        "precedents_cited": {
            "type": "array",
            "items": _PRECEDENT_CITATION,
            "minItems": 0,
            "maxItems": 5,
        },
    },
    "required": ["to_argument", "counter_strategy", "counter_reasoning", "norm_text_cited", "precedents_cited"],
    "additionalProperties": False,
}

_AFFIRMATIVE_DEFENSE = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "maxLength": 10},
        "type": {"type": "string", "enum": ["derived", "new"]},
        "derived_from": {"type": ["string", "null"], "maxLength": 10},
        "claim": {"type": "string", "maxLength": 500},
        "legal_reasoning": {"type": "string", "maxLength": 1000},
        "norm_text_cited": {
            "type": "array",
            "items": {"type": "string", "maxLength": 20},
            "minItems": 0,
            "maxItems": 10,
        },
        "facts_referenced": {
            "type": "array",
            "items": {"type": "string", "maxLength": 20},
            "minItems": 0,
            "maxItems": 10,
        },
        "evidence_cited": {
            "type": "array",
            "items": {"type": "string", "maxLength": 20},
            "minItems": 0,
            "maxItems": 10,
        },
    },
    "required": [
        "id", "type", "derived_from", "claim", "legal_reasoning",
        "norm_text_cited", "facts_referenced", "evidence_cited",
    ],
    "additionalProperties": False,
}

RESPONDENT_SCHEMA = {
    "type": "object",
    "properties": {
        "filed_brief": {
            "type": "object",
            "properties": {
                "preliminary_objections": {
                    "type": "array",
                    "items": _PRELIMINARY_OBJECTION,
                    "minItems": 0,
                    "maxItems": 5,
                },
                "responses_to_opponent": {
                    "type": "array",
                    "items": _RESPONSE_TO_OPPONENT,
                    "minItems": 1,
                    "maxItems": 10,
                },
                "affirmative_defenses": {
                    "type": "array",
                    "items": _AFFIRMATIVE_DEFENSE,
                    "minItems": 0,
                    "maxItems": 5,
                },
                "requests": {
                    "type": "object",
                    "properties": {
                        "primary": {"type": "string", "maxLength": 500},
                        "fallback": {"type": "string", "maxLength": 500},
                    },
                    "required": ["primary", "fallback"],
                    "additionalProperties": False,
                },
            },
            "required": ["preliminary_objections", "responses_to_opponent", "affirmative_defenses", "requests"],
            "additionalProperties": False,
        },
        "internal_analysis": {
            "type": "object",
            "properties": {
                "strength_self_assessments": {
                    "type": "object",
                    "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "key_vulnerabilities": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 500},
                    "minItems": 1,
                    "maxItems": 5,
                },
                "opponent_strongest_point": {"type": "string", "maxLength": 500},
                "gaps": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 500},
                    "minItems": 0,
                    "maxItems": 5,
                },
            },
            "required": ["strength_self_assessments", "key_vulnerabilities", "opponent_strongest_point", "gaps"],
            "additionalProperties": False,
        },
    },
    "required": ["filed_brief", "internal_analysis"],
    "additionalProperties": False,
}

# --- Judge ---

_PRELIMINARY_OBJECTION_RULING = {
    "type": "object",
    "properties": {
        "objection_id": {"type": "string", "maxLength": 10},
        "sustained": {"type": "boolean"},
        "reasoning": {"type": "string", "maxLength": 1000},
    },
    "required": ["objection_id", "sustained", "reasoning"],
    "additionalProperties": False,
}

_ARGUMENT_EVALUATION = {
    "type": "object",
    "properties": {
        "argument_id": {"type": "string", "maxLength": 10},
        "party": {"type": "string", "enum": ["appellant", "respondent"]},
        "persuasiveness": {"type": "number", "minimum": 0, "maximum": 1},
        "strengths": {"type": "string", "maxLength": 1000},
        "weaknesses": {"type": "string", "maxLength": 500},
        "determinative": {"type": "boolean"},
    },
    "required": ["argument_id", "party", "persuasiveness", "strengths", "weaknesses", "determinative"],
    "additionalProperties": False,
}

_PRECEDENT_ANALYSIS_ITEM = {
    "type": "object",
    "properties": {
        "followed": {"type": "boolean"},
        "distinguished": {"type": "boolean"},
        "reasoning": {"type": "string", "maxLength": 1500},
    },
    "required": ["followed", "distinguished", "reasoning"],
    "additionalProperties": False,
}

_INCORRECT_QUALIFICATION = {
    "type": "object",
    "properties": {
        "consequence": {"type": "string", "enum": ["annulment", "reclassification"]},
        "consequence_reasoning": {"type": "string", "maxLength": 1500},
        "applied_norm": {"type": "string", "maxLength": 500},
        "sanction_determined": {"type": "integer"},
        "points_deducted": {"type": "integer"},
    },
    "required": ["consequence", "consequence_reasoning", "applied_norm", "sanction_determined", "points_deducted"],
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "preliminary_objections_ruling": {
            "type": "array",
            "items": _PRELIMINARY_OBJECTION_RULING,
            "minItems": 0,
            "maxItems": 5,
        },
        "case_reaches_merits": {"type": "boolean"},
        "argument_evaluation": {
            "type": "array",
            "items": _ARGUMENT_EVALUATION,
            "minItems": 1,
            "maxItems": 15,
        },
        "precedent_analysis": {
            "type": "object",
            "additionalProperties": _PRECEDENT_ANALYSIS_ITEM,
        },
        "verdict": {
            "type": "object",
            "properties": {
                "qualification_correct": {"type": "boolean"},
                "qualification_reasoning": {"type": "string", "maxLength": 2000},
                "if_incorrect": {
                    "type": ["object", "null"],
                    "properties": {
                        "consequence": {"type": "string", "enum": ["annulment", "reclassification"]},
                        "consequence_reasoning": {"type": "string", "maxLength": 1500},
                        "applied_norm": {"type": "string", "maxLength": 500},
                        "sanction_determined": {"type": "integer"},
                        "points_deducted": {"type": "integer"},
                    },
                    "required": ["consequence", "consequence_reasoning", "applied_norm", "sanction_determined", "points_deducted"],
                },
                "costs_ruling": {"type": "string", "maxLength": 500},
            },
            "required": ["qualification_correct", "qualification_reasoning", "if_incorrect", "costs_ruling"],
            "additionalProperties": False,
        },
        "reasoning": {"type": "string", "maxLength": 5000},
        "gaps": {
            "type": "array",
            "items": {"type": "string", "maxLength": 500},
            "minItems": 0,
            "maxItems": 5,
        },
    },
    "required": [
        "preliminary_objections_ruling", "case_reaches_merits", "argument_evaluation",
        "precedent_analysis", "verdict", "reasoning", "gaps",
    ],
    "additionalProperties": False,
}

AGENT_SCHEMAS: dict[str, dict] = {
    "advocate_filing": APPELLANT_SCHEMA,
    "advocate_response": RESPONDENT_SCHEMA,
    "adjudicator": JUDGE_SCHEMA,
    # Backward-compatible aliases
    "appellant": APPELLANT_SCHEMA,
    "respondent": RESPONDENT_SCHEMA,
    "judge": JUDGE_SCHEMA,
}
