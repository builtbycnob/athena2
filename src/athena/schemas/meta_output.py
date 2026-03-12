# src/athena/schemas/meta_output.py
"""JSON Schema definitions for meta-agent structured output.

Red Team and Game Theorist schemas with maxLength/enum/minItems/maxItems
constraints for oMLX decode-time enforcement.
"""

_VULNERABILITY_ASSESSMENT = {
    "type": "object",
    "properties": {
        "target_argument_id": {"type": "string", "maxLength": 10},
        "attack_vector": {
            "type": "string",
            "enum": ["logical", "factual", "procedural", "evidentiary"],
        },
        "weakness_description": {"type": "string", "maxLength": 1000},
        "counter_argument": {"type": "string", "maxLength": 1500},
        "severity": {"type": "number", "minimum": 0, "maximum": 1},
        "defensive_recommendation": {"type": "string", "maxLength": 1000},
    },
    "required": [
        "target_argument_id", "attack_vector", "weakness_description",
        "counter_argument", "severity", "defensive_recommendation",
    ],
    "additionalProperties": False,
}

_STRATEGIC_VULNERABILITY = {
    "type": "object",
    "properties": {
        "vulnerability": {"type": "string", "maxLength": 500},
        "impact": {"type": "string", "maxLength": 500},
        "mitigation": {"type": "string", "maxLength": 500},
    },
    "required": ["vulnerability", "impact", "mitigation"],
    "additionalProperties": False,
}

RED_TEAM_SCHEMA = {
    "type": "object",
    "properties": {
        "vulnerability_assessment": {
            "type": "array",
            "items": _VULNERABILITY_ASSESSMENT,
            "minItems": 1,
            "maxItems": 10,
        },
        "strategic_vulnerabilities": {
            "type": "array",
            "items": _STRATEGIC_VULNERABILITY,
            "minItems": 0,
            "maxItems": 5,
        },
        "overall_risk_assessment": {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": ["low", "medium", "high"]},
                "reasoning": {"type": "string", "maxLength": 1500},
            },
            "required": ["level", "reasoning"],
            "additionalProperties": False,
        },
    },
    "required": ["vulnerability_assessment", "strategic_vulnerabilities", "overall_risk_assessment"],
    "additionalProperties": False,
}

# --- Game Theorist ---

_STRATEGY_RANKING_ITEM = {
    "type": "object",
    "properties": {
        "strategy_id": {"type": "string", "maxLength": 30},
        "expected_value_eur": {"type": "number"},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "when_to_use": {"type": "string", "maxLength": 500},
        "caveats": {"type": "string", "maxLength": 500},
    },
    "required": ["strategy_id", "expected_value_eur", "risk_level", "when_to_use", "caveats"],
    "additionalProperties": False,
}

GAME_THEORIST_SCHEMA = {
    "type": "object",
    "properties": {
        "strategic_summary": {"type": "string", "maxLength": 2000},
        "negotiation_position": {
            "type": "object",
            "properties": {
                "batna_interpretation": {"type": "string", "maxLength": 1000},
                "zopa_assessment": {"type": "string", "maxLength": 1000},
                "recommended_opening": {"type": "string", "maxLength": 500},
            },
            "required": ["batna_interpretation", "zopa_assessment", "recommended_opening"],
            "additionalProperties": False,
        },
        "strategy_ranking": {
            "type": "array",
            "items": _STRATEGY_RANKING_ITEM,
            "minItems": 1,
            "maxItems": 10,
        },
        "sensitivity_interpretation": {"type": "string", "maxLength": 1500},
        "settlement_recommendation": {
            "type": "object",
            "properties": {
                "should_settle": {"type": "boolean"},
                "recommended_price_eur": {"type": ["number", "null"]},
                "conditions": {"type": "string", "maxLength": 1000},
                "reasoning": {"type": "string", "maxLength": 1500},
            },
            "required": ["should_settle", "recommended_price_eur", "conditions", "reasoning"],
            "additionalProperties": False,
        },
    },
    "required": [
        "strategic_summary", "negotiation_position", "strategy_ranking",
        "sensitivity_interpretation", "settlement_recommendation",
    ],
    "additionalProperties": False,
}

# --- IRAC Extraction ---

_IRAC_DECOMPOSITION = {
    "type": "object",
    "properties": {
        "seed_arg_id": {"type": "string", "maxLength": 20},
        "claim": {"type": "string", "maxLength": 500},
        "issue": {"type": "string", "maxLength": 1000},
        "rule": {"type": "string", "maxLength": 1500},
        "application": {"type": "string", "maxLength": 2000},
        "conclusion": {"type": "string", "maxLength": 1000},
    },
    "required": [
        "seed_arg_id", "claim", "issue", "rule", "application", "conclusion",
    ],
    "additionalProperties": False,
}

IRAC_SCHEMA = {
    "type": "object",
    "properties": {
        "irac_analyses": {
            "type": "array",
            "items": _IRAC_DECOMPOSITION,
            "minItems": 1,
            "maxItems": 20,
        },
    },
    "required": ["irac_analyses"],
    "additionalProperties": False,
}
