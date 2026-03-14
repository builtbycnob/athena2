"""Dynamic JSON schema builder for XGrammar constrained decoding.

Deep-copies static schemas from structured_output.py and injects enum
constraints from case data. With XGrammar, enum constraints are enforced
at the token level: the model CANNOT generate an ID not in the enum.
"""

import copy

from athena.schemas.structured_output import AGENT_SCHEMAS


def build_schema_for_agent(
    schema_key: str,
    case_data: dict,
    prior_briefs: dict[str, dict] | None = None,
    step1_error_count: int | None = None,
) -> dict:
    """Deep-copy static schema, inject enum constraints from case data."""
    schema = copy.deepcopy(AGENT_SCHEMAS[schema_key])

    # Extract categorized IDs from case
    facts = case_data.get("facts", {})
    fact_ids = [f["id"] for f in facts.get("undisputed", [])] + \
               [f["id"] for f in facts.get("disputed", [])]
    evidence_ids = [e["id"] for e in case_data.get("evidence", [])]
    norm_ids = [lt["id"] for lt in case_data.get("legal_texts", [])]
    precedent_ids = [p["id"] for p in case_data.get("key_precedents", [])]

    # Collect argument IDs from prior briefs
    arg_ids = []
    if prior_briefs:
        for brief in prior_briefs.values():
            if brief and "filed_brief" in brief:
                fb = brief["filed_brief"]
                arg_ids.extend(a["id"] for a in fb.get("arguments", []))
                arg_ids.extend(d["id"] for d in fb.get("affirmative_defenses", []))

    # Build field→enum mapping
    field_enums = {
        "facts_referenced": fact_ids,
        "evidence_cited": evidence_ids,
        "norm_text_cited": norm_ids,
        "legal_basis": norm_ids,
        "to_argument": arg_ids,
        "argument_id": arg_ids,
    }

    # Inject error_id enum for Step 2
    if step1_error_count is not None:
        field_enums["error_id"] = list(range(step1_error_count))

    _patch_enum_fields(schema, field_enums, precedent_ids)

    # Dynamic minItems for completeness guarantee
    if arg_ids and "argument_evaluation" in schema.get("properties", {}):
        schema["properties"]["argument_evaluation"]["minItems"] = len(arg_ids)

    return schema


def _patch_enum_fields(schema: dict, field_enums: dict, precedent_ids: list):
    """Recursively walk schema tree, inject enum constraints by field name."""
    if not isinstance(schema, dict):
        return

    props = schema.get("properties", {})
    for field_name, field_schema in props.items():
        # Array of strings → inject enum on items
        if field_name in field_enums and field_schema.get("type") == "array":
            items = field_schema.get("items", {})
            if items.get("type") == "string" and field_enums[field_name]:
                items["enum"] = sorted(set(field_enums[field_name]))

        # Direct string field → inject enum
        elif field_name in field_enums and field_schema.get("type") == "string":
            if field_enums[field_name]:
                field_schema["enum"] = sorted(set(field_enums[field_name]))

        # Direct integer field → inject enum (error_id)
        elif field_name in field_enums and field_schema.get("type") == "integer":
            if field_enums[field_name]:
                field_schema["enum"] = sorted(set(field_enums[field_name]))

        # Nested precedent objects
        elif field_name in ("precedents_addressed", "precedents_cited"):
            items = field_schema.get("items", {})
            if "properties" in items and "id" in items["properties"]:
                if precedent_ids:
                    items["properties"]["id"]["enum"] = sorted(set(precedent_ids))

        # Recurse into nested objects and array items
        _patch_enum_fields(field_schema, field_enums, precedent_ids)

    # Recurse into array items
    if "items" in schema:
        _patch_enum_fields(schema["items"], field_enums, precedent_ids)
