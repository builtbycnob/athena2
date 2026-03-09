import pytest
import yaml
from pathlib import Path


@pytest.fixture
def sample_case_data():
    """Minimal case data for testing."""
    return {
        "case_id": "gdp-milano-17928-2025",
        "jurisdiction": {
            "country": "IT",
            "court": "giudice_di_pace",
            "venue": "Milano",
            "applicable_law": ["D.Lgs. 285/1992", "L. 689/1981"],
            "key_precedents": [
                {
                    "id": "cass_16515_2005",
                    "citation": "Cass. civ. n. 16515/2005",
                    "holding": "Equiparazione contromano/controsenso",
                    "weight": "contested",
                }
            ],
            "procedural_rules": {
                "rite": "opposizione_sanzione_amministrativa",
                "phases": ["ricorso", "costituzione_resistente", "udienza", "decisione"],
                "allowed_moves": {
                    "appellant": ["memoria", "produzione_documenti"],
                    "respondent": ["memoria_costituzione", "produzione_documenti"],
                },
            },
        },
        "parties": [
            {
                "id": "opponente",
                "role": "appellant",
                "type": "persona_fisica",
                "objectives": {
                    "primary": "annullamento_verbale",
                    "subordinate": "riqualificazione_artt_6_7",
                },
            },
            {
                "id": "comune_milano",
                "role": "respondent",
                "type": "pubblica_amministrazione",
                "entity": "Comune di Milano — Polizia Locale",
                "objectives": {
                    "primary": "conferma_verbale",
                    "subordinate": "conferma_anche_con_riduzione",
                },
            },
        ],
        "stakes": {
            "current_sanction": {
                "norm": "art. 143 CdS",
                "fine_range": [170, 680],
                "points_deducted": 4,
            },
            "alternative_sanction": {
                "norm": "artt. 6-7 CdS",
                "fine_range": [42, 173],
                "points_deducted": 0,
            },
            "litigation_cost_estimate": 1500,
        },
        "evidence": [
            {
                "id": "DOC1",
                "type": "atto_pubblico",
                "description": "Verbale Polizia Locale",
                "produced_by": "comune_milano",
                "admissibility": "uncontested",
                "supports_facts": ["F1", "F2", "F3"],
            },
            {
                "id": "DOC2",
                "type": "prova_documentale",
                "description": "Documentazione segnaletica",
                "produced_by": "opponente",
                "admissibility": "uncontested",
                "supports_facts": ["F3"],
            },
        ],
        "facts": {
            "undisputed": [
                {"id": "F1", "description": "Transito in senso vietato", "evidence": ["DOC1"]},
                {"id": "F2", "description": "Verbale ex art. 143 CdS", "evidence": ["DOC1"]},
                {"id": "F3", "description": "Strada a senso unico", "evidence": ["DOC1", "DOC2"]},
            ],
            "disputed": [
                {
                    "id": "D1",
                    "description": "Correttezza qualificazione giuridica",
                    "appellant_position": "Art. 143 inapplicabile",
                    "respondent_position": "Art. 143 applicabile per Cass. 16515/2005",
                    "depends_on_facts": ["F1", "F3"],
                }
            ],
        },
        "legal_texts": [
            {
                "id": "art_143_cds",
                "norm": "Art. 143 D.Lgs. 285/1992",
                "text": "I veicoli devono circolare sulla parte destra della carreggiata e in prossimità del margine destro della medesima, anche quando la strada è libera. [testo di esempio per test]",
            },
            {
                "id": "art_6_cds",
                "norm": "Art. 6 D.Lgs. 285/1992",
                "text": "Il prefetto può, per motivi di sicurezza pubblica o inerenti alla sicurezza della circolazione... [testo di esempio per test]",
            },
            {
                "id": "art_1_l689",
                "norm": "Art. 1 L. 689/1981",
                "text": "Nessuno può essere assoggettato a sanzioni amministrative se non in forza di una legge che sia entrata in vigore prima della commissione della violazione. Le leggi che prevedono sanzioni amministrative si applicano soltanto nei casi e per i tempi in esse considerati.",
            },
        ],
        "seed_arguments": {
            "appellant": [
                {
                    "id": "SEED_ARG1",
                    "claim": "Errata qualificazione giuridica",
                    "direction": "Art. 143 non copre la fattispecie",
                    "references_facts": ["F1", "F3", "D1"],
                },
                {
                    "id": "SEED_ARG2",
                    "claim": "Contraddizione interna del verbale",
                    "direction": "Verbale descrive senso unico, applica norma da doppio senso",
                    "references_facts": ["F3"],
                },
            ],
            "respondent": [
                {
                    "id": "SEED_RARG1",
                    "claim": "Legittimità ex Cass. 16515/2005",
                    "direction": "Cassazione equipara le due condotte",
                    "references_facts": ["F1", "D1"],
                },
            ],
        },
        "key_precedents": [
            {
                "id": "cass_16515_2005",
                "citation": "Cass. civ. n. 16515/2005",
                "holding": "Equiparazione contromano/controsenso",
                "weight": "contested",
            }
        ],
        "timeline": [],
    }


@pytest.fixture
def sample_run_params():
    """Minimal run params for testing."""
    return {
        "run_id": "test__aggressivo__000",
        "judge_profile": {
            "id": "formalista_pro_cass",
            "jurisprudential_orientation": "follows_cassazione",
            "formalism": "high",
        },
        "appellant_profile": {
            "id": "aggressivo",
            "style": "Attacca frontalmente la giurisprudenza sfavorevole.",
        },
        "temperature": {"appellant": 0.5, "respondent": 0.4, "judge": 0.3},
        "language": "it",
    }


@pytest.fixture
def sample_appellant_brief():
    """Valid appellant brief output for testing downstream agents."""
    return {
        "filed_brief": {
            "arguments": [
                {
                    "id": "ARG1",
                    "type": "derived",
                    "derived_from": "SEED_ARG1",
                    "claim": "Errata qualificazione giuridica del fatto",
                    "legal_reasoning": "L'art. 143 disciplina la marcia contromano su strada a doppio senso. Il fatto è avvenuto su senso unico.",
                    "norm_text_cited": ["art_143_cds"],
                    "facts_referenced": ["F1", "F3"],
                    "evidence_cited": ["DOC1"],
                    "precedents_addressed": [
                        {
                            "id": "cass_16515_2005",
                            "strategy": "distinguish",
                            "reasoning": "Il precedente non è in punto.",
                        }
                    ],
                    "supports": None,
                },
            ],
            "requests": {
                "primary": "Annullamento del verbale",
                "subordinate": "Riqualificazione sotto artt. 6-7 CdS",
            },
        },
        "internal_analysis": {
            "strength_self_assessments": {"ARG1": 0.7},
            "key_vulnerabilities": ["Cassazione 16515/2005 contraria"],
            "strongest_point": "Testo letterale art. 143 non copre senso unico",
            "gaps": [],
        },
    }


@pytest.fixture
def sample_respondent_brief():
    """Valid respondent brief output for testing judge."""
    return {
        "filed_brief": {
            "preliminary_objections": [],
            "responses_to_opponent": [
                {
                    "to_argument": "ARG1",
                    "counter_strategy": "rebut",
                    "counter_reasoning": "La Cassazione ha equiparato le due fattispecie.",
                    "norm_text_cited": ["art_143_cds"],
                    "precedents_cited": [
                        {"id": "cass_16515_2005", "relevance": "Direttamente in punto."}
                    ],
                }
            ],
            "affirmative_defenses": [
                {
                    "id": "RARG1",
                    "type": "derived",
                    "derived_from": "SEED_RARG1",
                    "claim": "Legittimità del verbale ex Cass. 16515/2005",
                    "legal_reasoning": "La Cassazione equipara contromano e controsenso.",
                    "norm_text_cited": ["art_143_cds"],
                    "facts_referenced": ["F1"],
                    "evidence_cited": ["DOC1"],
                }
            ],
            "requests": {
                "primary": "Rigetto dell'opposizione",
                "fallback": "Conferma sanzione anche in caso di riqualificazione",
            },
        },
        "internal_analysis": {
            "strength_self_assessments": {"response_to_ARG1": 0.6, "RARG1": 0.6},
            "key_vulnerabilities": ["Testo letterale art. 143 non chiarissimo"],
            "opponent_strongest_point": "Argomento testuale sull'art. 143",
            "gaps": [],
        },
    }
