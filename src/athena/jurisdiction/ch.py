# src/athena/jurisdiction/ch.py
"""Swiss jurisdiction — Tribunale federale (Bundesgericht), ricorso in materia civile."""

from athena.jurisdiction.registry import JurisdictionConfig, register_jurisdiction


def _ch_outcome_extractor(verdict: dict) -> str:
    """Extract outcome from Swiss judge verdict (appeal_outcome schema)."""
    outcome = verdict.get("appeal_outcome", "dismissed")
    if outcome == "dismissed":
        return "rejection"
    # upheld, partially_upheld, remanded → all favorable to appellant
    return "annulment"


_CH_CONFIG = JurisdictionConfig(
    country="CH",
    prompt_keys={
        "appellant": "appellant_ch",
        "respondent": "respondent_ch",
        "judge": "judge_ch",
    },
    schema_keys={
        "appellant": "appellant",       # same schema, different prompt
        "respondent": "respondent",     # same schema, different prompt
        "judge": "judge_ch",            # different schema
    },
    verdict_schema_key="judge_ch",
    outcome_extractor=_ch_outcome_extractor,
    outcome_space=["rejection", "annulment"],
    source_hierarchy=(
        "Costituzione federale > Leggi federali (CO, CC, LEF, LTF) > "
        "Ordinanze del Consiglio federale > Diritto cantonale > "
        "Giurisprudenza del Tribunale federale (DTF/BGE)"
    ),
    respondent_brief_label="Memoria della controparte (depositata)",
)

register_jurisdiction("CH", _CH_CONFIG)
