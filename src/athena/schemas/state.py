from pydantic import BaseModel
from typing import Any


class RunParams(BaseModel):
    run_id: str
    judge_profile: dict[str, Any]
    party_profiles: dict[str, dict[str, Any]]  # party_id → profile
    temperatures: dict[str, float]
    language: str


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str]
    warnings: list[str]
