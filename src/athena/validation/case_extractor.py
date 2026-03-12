# src/athena/validation/case_extractor.py
"""Convert fetched dataset records into ATHENA YAML case files.

Three-tier extraction:
- Tier A (deterministic): case_id, jurisdiction, outcome label
- Tier B (LLM-assisted): facts, parties, seed arguments, precedents from text
- Tier C (template): procedural rules, legal texts, simulation config
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from athena.validation.enricher import (
    get_applicable_law,
    get_party_templates,
    get_procedural_rules,
)
from athena.validation.ground_truth import SWISS_LABEL_MAP, GroundTruth, save_ground_truth


# --- LLM extraction prompt ---

EXTRACTION_PROMPT = """\
Sei un assistente legale. Analizza il seguente testo di una sentenza del Tribunale Federale Svizzero ed estrai le informazioni richieste in formato JSON.

TESTO DELLA SENTENZA:
{text}

Estrai le seguenti informazioni in formato JSON (tutti i campi sono obbligatori):

{{
  "facts_undisputed": [
    {{"id": "F1", "description": "...", "evidence": ["DOC1"]}}
  ],
  "facts_disputed": [
    {{"id": "D1", "description": "...", "appellant_position": "...", "respondent_position": "...", "depends_on_facts": ["F1"]}}
  ],
  "evidence": [
    {{"id": "DOC1", "type": "atto_pubblico", "description": "...", "produced_by": "ricorrente", "admissibility": "uncontested", "supports_facts": ["F1"]}}
  ],
  "seed_arguments_appellant": [
    {{"id": "SEED_ARG1", "claim": "...", "direction": "...", "references_facts": ["F1", "D1"]}}
  ],
  "seed_arguments_respondent": [
    {{"id": "SEED_RARG1", "claim": "...", "direction": "...", "references_facts": ["F1"]}}
  ],
  "key_precedents": [
    {{"id": "prec_1", "citation": "...", "holding": "...", "weight": "binding"}}
  ],
  "legal_texts_cited": ["art. X CC", "art. Y CO"],
  "stakes_description": "Breve descrizione della posta in gioco",
  "timeline": [
    {{"date": "YYYY-MM-DD", "event": "..."}}
  ]
}}

REGOLE:
- Usa IDs sequenziali (F1, F2, D1, D2, DOC1, DOC2, SEED_ARG1, SEED_RARG1, prec_1)
- produced_by deve essere "ricorrente" o "controparte"
- Mantieni il linguaggio giuridico originale
- Se un'informazione non è estraibile, usa un array/oggetto vuoto
- Rispondi SOLO con il JSON, senza testo aggiuntivo
"""


def extract_case_deterministic(
    record: dict[str, Any],
    *,
    country: str = "CH",
    court: str = "bundesgericht",
) -> tuple[dict[str, Any], GroundTruth]:
    """Tier A: deterministic extraction from dataset record.

    Returns (partial_case_dict, ground_truth).
    """
    case_id = f"ch-{record['id']}"
    legal_area = record.get("legal_area", "civil_law")

    proc_rules = get_procedural_rules(country, court)
    applicable_law = get_applicable_law(country, legal_area)

    case_data: dict[str, Any] = {
        "case_id": case_id,
        "jurisdiction": {
            "country": country,
            "court": court,
            "venue": record.get("canton", "unknown"),
            "applicable_law": applicable_law,
            "key_precedents": [],
            "procedural_rules": proc_rules.model_dump(),
        },
        "key_precedents": [],
        # Placeholders for Tier B
        "parties": [],
        "stakes": None,
        "evidence": [],
        "facts": {"undisputed": [], "disputed": []},
        "legal_texts": [],
        "seed_arguments": {"by_party": {}},
        "timeline": [],
    }

    gt = GroundTruth(
        case_id=case_id,
        source="swiss_judgment_prediction",
        outcome=SWISS_LABEL_MAP[record["label"]],
        outcome_raw=record["label"],
        extraction_confidence="high",
        legal_area=legal_area,
        year=record.get("year"),
        canton=record.get("canton"),
        region=record.get("region"),
    )

    return case_data, gt


def extract_case_llm(
    record: dict[str, Any],
    case_data: dict[str, Any],
    *,
    invoke_fn: Any | None = None,
) -> dict[str, Any]:
    """Tier B: LLM-assisted extraction of facts, arguments, etc.

    Args:
        record: Raw dataset record with 'text' field.
        case_data: Partial case dict from Tier A.
        invoke_fn: Optional LLM invocation function. If None, uses athena.agents.llm.invoke_llm.
    """
    if invoke_fn is None:
        from athena.agents.llm import _call_model

        def _default_invoke(**kwargs):
            raw, _finish, _ptok, _otok = _call_model(
                kwargs["system_prompt"],
                kwargs["user_prompt"],
                kwargs.get("temperature", 0.3),
                kwargs.get("max_tokens", 4096),
            )
            return raw  # Return raw text, we parse JSON ourselves

        invoke_fn = _default_invoke

    prompt = EXTRACTION_PROMPT.format(text=record["text"][:8000])  # Truncate very long texts

    extracted = invoke_fn(
        system_prompt="Sei un assistente legale esperto in diritto svizzero.",
        user_prompt=prompt,
        temperature=0.3,
        max_tokens=4096,
    )

    # Parse JSON from response
    if isinstance(extracted, str):
        # Try to extract JSON from response text
        extracted = _parse_json_response(extracted)

    if not isinstance(extracted, dict):
        return case_data  # Return unchanged if extraction failed

    return _merge_extraction(case_data, extracted, record)


def _parse_json_response(text: str) -> dict | str:
    """Try to parse JSON from an LLM response."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fall back to json_repair
    try:
        from json_repair import repair_json
        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass

    return text


def _merge_extraction(
    case_data: dict[str, Any],
    extracted: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    """Merge LLM extraction into case data."""
    country = case_data["jurisdiction"]["country"]
    party_templates = get_party_templates(country)

    # Parties (Tier C template + Tier B enrichment)
    case_data["parties"] = [
        party_templates["appellant"],
        party_templates["respondent"],
    ]

    # Evidence
    if extracted.get("evidence"):
        case_data["evidence"] = extracted["evidence"]

    # Facts
    if extracted.get("facts_undisputed"):
        case_data["facts"]["undisputed"] = extracted["facts_undisputed"]
    if extracted.get("facts_disputed"):
        disputed = []
        for df in extracted["facts_disputed"]:
            # Convert flat format to positions dict
            positions = {}
            if "appellant_position" in df:
                positions["ricorrente"] = df.pop("appellant_position")
            if "respondent_position" in df:
                positions["controparte"] = df.pop("respondent_position")
            if positions:
                df["positions"] = positions
            disputed.append(df)
        case_data["facts"]["disputed"] = disputed

    # Seed arguments
    by_party: dict[str, list] = {}
    if extracted.get("seed_arguments_appellant"):
        by_party["ricorrente"] = extracted["seed_arguments_appellant"]
    if extracted.get("seed_arguments_respondent"):
        by_party["controparte"] = extracted["seed_arguments_respondent"]
    if by_party:
        case_data["seed_arguments"] = {"by_party": by_party}

    # Key precedents
    if extracted.get("key_precedents"):
        case_data["key_precedents"] = extracted["key_precedents"]
        case_data["jurisdiction"]["key_precedents"] = extracted["key_precedents"]

    # Legal texts (Tier B: cited norms → minimal placeholders)
    if extracted.get("legal_texts_cited"):
        legal_texts = []
        for i, norm_ref in enumerate(extracted["legal_texts_cited"][:10]):
            legal_texts.append({
                "id": f"norm_{i+1}",
                "norm": norm_ref,
                "text": f"[Testo da fedlex.admin.ch: {norm_ref}]",
            })
        case_data["legal_texts"] = legal_texts

    # Stakes (minimal placeholder from description)
    stakes_desc = extracted.get("stakes_description", "Ricorso al Tribunale Federale")
    case_data["stakes"] = {
        "current_sanction": {
            "norm": "decisione impugnata",
            "fine_range": [0, 0],
            "points_deducted": 0,
        },
        "alternative_sanction": {
            "norm": "esito favorevole ricorrente",
            "fine_range": [0, 0],
            "points_deducted": 0,
        },
        "litigation_cost_estimate": 2000,
        "non_monetary": stakes_desc,
    }

    # Timeline
    if extracted.get("timeline"):
        case_data["timeline"] = extracted["timeline"]
    else:
        case_data["timeline"] = [
            {"date": f"{record.get('year', 'unknown')}-01-01", "event": "Decisione impugnata"},
            {"date": "pending", "event": "Sentenza Tribunale Federale"},
        ]

    return case_data


def extract_and_save(
    record: dict[str, Any],
    cases_dir: str | Path,
    ground_truth_dir: str | Path,
    *,
    use_llm: bool = True,
    invoke_fn: Any | None = None,
) -> tuple[Path, Path]:
    """Full extraction pipeline: Tier A + B + C → save YAML + ground truth.

    Args:
        record: Dataset record.
        cases_dir: Directory for output YAML files.
        ground_truth_dir: Directory for ground truth JSON files.
        use_llm: If True, run LLM extraction (Tier B). If False, use templates only.
        invoke_fn: Optional custom LLM invocation function.

    Returns:
        (case_yaml_path, ground_truth_path)
    """
    cases_dir = Path(cases_dir)
    cases_dir.mkdir(parents=True, exist_ok=True)

    # Tier A
    case_data, gt = extract_case_deterministic(record)

    # Tier B (optional)
    if use_llm:
        case_data = extract_case_llm(record, case_data, invoke_fn=invoke_fn)
    else:
        # Tier C only: use templates
        country = case_data["jurisdiction"]["country"]
        party_templates = get_party_templates(country)
        case_data["parties"] = [party_templates["appellant"], party_templates["respondent"]]
        case_data["stakes"] = {
            "current_sanction": {"norm": "decisione impugnata", "fine_range": [0, 0], "points_deducted": 0},
            "alternative_sanction": {"norm": "esito favorevole", "fine_range": [0, 0], "points_deducted": 0},
            "litigation_cost_estimate": 2000,
        }
        case_data["timeline"] = [
            {"date": f"{record.get('year', 'unknown')}-01-01", "event": "Decisione"},
        ]
        # Minimal seed arguments from text summary
        case_data["seed_arguments"] = {
            "by_party": {
                "ricorrente": [{"id": "SEED_ARG1", "claim": "Violazione del diritto federale",
                                "direction": "Ricorso fondato", "references_facts": []}],
                "controparte": [{"id": "SEED_RARG1", "claim": "Decisione conforme al diritto",
                                 "direction": "Ricorso infondato", "references_facts": []}],
            }
        }

    # Save YAML
    case_id = case_data["case_id"]
    yaml_path = cases_dir / f"{case_id}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({"case": _prepare_for_yaml(case_data)}, f,
                   default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Save ground truth
    gt_path = save_ground_truth(gt, ground_truth_dir)

    return yaml_path, gt_path


def _prepare_for_yaml(case_data: dict) -> dict:
    """Prepare case data for YAML serialization (case_id → id, etc.)."""
    out = dict(case_data)
    if "case_id" in out:
        out["id"] = out.pop("case_id")
    return out
