# src/athena/validation/enricher.py
"""Jurisdiction-specific templates for procedural rules, legal texts, and parties.

Provides Tier C (template) enrichment for case extraction.
"""

from athena.schemas.case import ProceduralRules

# --- Swiss Federal Tribunal ---

SWISS_PROCEDURAL_RULES = ProceduralRules(
    rite="ricorso_tribunale_federale",
    phases=["ricorso", "risposta", "deliberazione", "sentenza"],
    allowed_moves={
        "appellant": ["memoria_ricorso", "replica", "produzione_documenti"],
        "respondent": ["risposta", "duplica", "produzione_documenti"],
    },
)

SWISS_APPLICABLE_LAW_BY_AREA = {
    "civil_law": ["Codice civile svizzero (CC)", "Codice delle obbligazioni (CO)"],
    "penal_law": ["Codice penale svizzero (CP)", "Codice di procedura penale (CPP)"],
    "public_law": ["Costituzione federale (Cost.)", "Legge sul Tribunale federale (LTF)"],
    "social_law": ["Legge federale sull'assicurazione per l'invalidità (LAI)",
                    "Legge federale sull'assicurazione per la vecchiaia e per i superstiti (LAVS)"],
}

# Default party templates for Swiss bilateral cases
SWISS_PARTY_TEMPLATES = {
    "appellant": {
        "id": "ricorrente",
        "role": "appellant",
        "type": "persona_fisica",
        "objectives": {
            "primary": "accoglimento_ricorso",
            "subordinate": "rinvio_istanza_inferiore",
        },
    },
    "respondent": {
        "id": "controparte",
        "role": "respondent",
        "type": "autorita_cantonale",
        "objectives": {
            "primary": "rigetto_ricorso",
            "subordinate": "conferma_decisione_impugnata",
        },
    },
}

# --- Italian Giudice di Pace (already existing, reference) ---

IT_PROCEDURAL_RULES = ProceduralRules(
    rite="opposizione_sanzione_amministrativa",
    phases=["ricorso", "costituzione_resistente", "udienza", "decisione"],
    allowed_moves={
        "appellant": ["memoria", "produzione_documenti", "discussione_orale"],
        "respondent": ["memoria_costituzione", "produzione_documenti", "discussione_orale"],
    },
)


def get_procedural_rules(country: str, court: str) -> ProceduralRules:
    """Get procedural rules template for a given jurisdiction."""
    if country == "CH" and court == "bundesgericht":
        return SWISS_PROCEDURAL_RULES
    if country == "IT" and court == "giudice_di_pace":
        return IT_PROCEDURAL_RULES
    # Default: Swiss Federal Tribunal
    return SWISS_PROCEDURAL_RULES


def get_applicable_law(country: str, legal_area: str) -> list[str]:
    """Get applicable law references for a given country and legal area."""
    if country == "CH":
        return SWISS_APPLICABLE_LAW_BY_AREA.get(legal_area, ["Legge sul Tribunale federale (LTF)"])
    return []


def get_party_templates(country: str) -> dict[str, dict]:
    """Get default party templates for a jurisdiction."""
    if country == "CH":
        return SWISS_PARTY_TEMPLATES
    return SWISS_PARTY_TEMPLATES  # fallback
