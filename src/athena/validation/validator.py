# src/athena/validation/validator.py
"""Schema validation for converted case YAML files."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from athena.cli import migrate_case_v1
from athena.schemas.case import CaseFile


class ValidationResult:
    """Result of validating a case YAML."""

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.valid = True
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add_error(self, msg: str) -> None:
        self.valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_case_yaml(path: str | Path) -> ValidationResult:
    """Validate a case YAML file against the CaseFile Pydantic model.

    Checks:
    1. YAML parses correctly
    2. All required CaseFile fields present
    3. Referential integrity (evidence↔facts↔arguments IDs)
    4. At least 2 parties
    5. At least 1 seed argument per party
    """
    path = Path(path)
    case_id = path.stem

    result = ValidationResult(case_id)

    # 1. Parse YAML
    try:
        raw = yaml.safe_load(path.read_text())
    except Exception as e:
        result.add_error(f"YAML parse error: {e}")
        return result

    # Unwrap case: key
    case_data = raw.get("case", raw)
    if "id" in case_data and "case_id" not in case_data:
        case_data["case_id"] = case_data.pop("id")
    if "key_precedents" not in case_data:
        jur = case_data.get("jurisdiction")
        if isinstance(jur, dict):
            case_data["key_precedents"] = jur.get("key_precedents", [])
    case_data = migrate_case_v1(case_data)

    # 2. Pydantic validation
    try:
        case_file = CaseFile(**case_data)
    except ValidationError as e:
        for err in e.errors():
            result.add_error(f"Schema: {err['loc']} — {err['msg']}")
        return result

    # 3. Referential integrity
    all_ids = case_file.extract_all_ids()
    evidence_ids = {e.id for e in case_file.evidence}
    fact_ids = {f.id for f in case_file.facts.undisputed} | {f.id for f in case_file.facts.disputed}

    for f in case_file.facts.undisputed:
        for eid in f.evidence:
            if eid not in evidence_ids:
                result.add_error(f"Fact {f.id} references unknown evidence {eid}")

    for party_id, args in case_file.seed_arguments.by_party.items():
        for arg in args:
            for fid in arg.references_facts:
                if fid not in fact_ids:
                    result.add_warning(f"Argument {arg.id} references unknown fact {fid}")

    for e in case_file.evidence:
        for fid in e.supports_facts:
            if fid not in fact_ids:
                result.add_warning(f"Evidence {e.id} supports unknown fact {fid}")

    # 4. Party checks
    party_ids = {p.id for p in case_file.parties}
    if len(case_file.parties) < 2:
        result.add_error("Case must have at least 2 parties")

    # 5. Seed arguments: at least 1 party has arguments
    if not case_file.seed_arguments.by_party:
        result.add_error("No seed arguments defined")
    else:
        for pid, args in case_file.seed_arguments.by_party.items():
            if not args:
                result.add_warning(f"Party {pid} has no seed arguments")

    # 6. Content quality warnings
    if not case_file.facts.undisputed and not case_file.facts.disputed:
        result.add_warning("No facts defined")
    if not case_file.evidence:
        result.add_warning("No evidence defined")
    if not case_file.legal_texts:
        result.add_warning("No legal texts defined")

    return result


def validate_case_dict(case_data: dict[str, Any]) -> ValidationResult:
    """Validate a case dict (already parsed from YAML) against CaseFile."""
    case_id = case_data.get("case_id", "unknown")
    result = ValidationResult(case_id)

    data = dict(case_data)
    if "key_precedents" not in data:
        jur = data.get("jurisdiction")
        if isinstance(jur, dict):
            data["key_precedents"] = jur.get("key_precedents", [])
    data = migrate_case_v1(data)

    try:
        case_file = CaseFile(**data)
    except ValidationError as e:
        for err in e.errors():
            result.add_error(f"Schema: {err['loc']} — {err['msg']}")
        return result

    # Same checks as above but on the dict
    if len(case_file.parties) < 2:
        result.add_error("Case must have at least 2 parties")
    if not case_file.seed_arguments.by_party:
        result.add_error("No seed arguments defined")

    return result
