from pydantic import BaseModel
from typing import Any

from athena.schemas.agents import AppellantBrief, RespondentBrief, JudgeDecision


class RunParams(BaseModel):
    run_id: str
    judge_profile: dict[str, Any]
    appellant_profile: dict[str, Any]
    temperature: dict[str, float]
    language: str


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str]
    warnings: list[str]


class SimulationState(BaseModel):
    case_data: dict[str, Any] | None = None
    run_params: RunParams | None = None
    appellant_output: AppellantBrief | None = None
    respondent_output: RespondentBrief | None = None
    judge_output: JudgeDecision | None = None
    validation_results: list[ValidationResult] | None = None
    current_node: str | None = None
    error: str | None = None
