# src/athena/jurisdiction/it.py
"""Italian jurisdiction — Giudice di Pace, opposizione sanzione amministrativa.

Wraps existing prompts, schemas, and outcome logic. Zero behavior change.
"""

from athena.jurisdiction.registry import JurisdictionConfig, register_jurisdiction


def _it_outcome_extractor(verdict: dict) -> str:
    """Extract outcome from Italian judge verdict (qualification_correct schema)."""
    if verdict.get("qualification_correct"):
        return "rejection"
    if_incorrect = verdict.get("if_incorrect") or {}
    consequence = if_incorrect.get("consequence", "annulment")
    if consequence == "reclassification":
        return "reclassification"
    return "annulment"


_IT_CONFIG = JurisdictionConfig(
    country="IT",
    prompt_keys={
        "appellant": "appellant_it",
        "respondent": "respondent_it",
        "judge": "judge_it",
    },
    schema_keys={
        "appellant": "appellant",
        "respondent": "respondent",
        "judge": "judge",
    },
    verdict_schema_key="judge",
    outcome_extractor=_it_outcome_extractor,
    outcome_space=["rejection", "annulment", "reclassification"],
    source_hierarchy=(
        "Costituzione > Legge ordinaria > Regolamento > "
        "Giurisprudenza di Cassazione > Prassi"
    ),
    respondent_brief_label="Memoria del Comune (depositata)",
)

register_jurisdiction("IT", _IT_CONFIG)
