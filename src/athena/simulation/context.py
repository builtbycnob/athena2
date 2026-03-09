# src/athena/simulation/context.py
from typing import Any


def _sanitize_brief_for_opponent(brief: dict) -> dict:
    """Strip internal_analysis, return only filed_brief contents."""
    return brief["filed_brief"]


def _sanitize_brief_for_judge(brief: dict) -> dict:
    """Strip internal_analysis, return only filed_brief contents."""
    return brief["filed_brief"]


def build_context_appellant(case_data: dict, run_params: dict) -> dict:
    return {
        "facts": case_data["facts"],
        "evidence": [
            e for e in case_data["evidence"]
            if e["produced_by"] == "opponente"
            or e["admissibility"] == "uncontested"
        ],
        "legal_texts": case_data["legal_texts"],
        "precedents": case_data["key_precedents"],
        "seed_arguments": case_data["seed_arguments"]["appellant"],
        "own_party": next(
            p for p in case_data["parties"] if p["role"] == "appellant"
        ),
        "stakes": case_data["stakes"],
        "procedural_rules": case_data["jurisdiction"]["procedural_rules"],
        "advocacy_style": run_params["appellant_profile"]["style"],
    }


def build_context_respondent(
    case_data: dict, run_params: dict, appellant_brief: dict
) -> dict:
    return {
        "facts": case_data["facts"],
        "evidence": case_data["evidence"],
        "legal_texts": case_data["legal_texts"],
        "precedents": case_data["key_precedents"],
        "seed_arguments": case_data["seed_arguments"]["respondent"],
        "own_party": next(
            p for p in case_data["parties"] if p["role"] == "respondent"
        ),
        "stakes": case_data["stakes"],
        "procedural_rules": case_data["jurisdiction"]["procedural_rules"],
        "appellant_brief": _sanitize_brief_for_opponent(appellant_brief),
    }


def build_context_judge(
    case_data: dict,
    run_params: dict,
    appellant_brief: dict,
    respondent_brief: dict,
) -> dict:
    return {
        "facts": case_data["facts"],
        "evidence": case_data["evidence"],
        "legal_texts": case_data["legal_texts"],
        "precedents": case_data["key_precedents"],
        "stakes": case_data["stakes"],
        "procedural_rules": case_data["jurisdiction"]["procedural_rules"],
        "appellant_brief": _sanitize_brief_for_judge(appellant_brief),
        "respondent_brief": _sanitize_brief_for_judge(respondent_brief),
        "judge_profile": run_params["judge_profile"],
    }
