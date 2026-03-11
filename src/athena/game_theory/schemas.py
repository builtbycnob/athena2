# src/athena/game_theory/schemas.py
"""Pydantic models for game theory analysis I/O."""

from pydantic import BaseModel


class OutcomeValuation(BaseModel):
    outcome: str  # "rejection" | "annulment" | "reclassification"
    description: str
    fine: float  # Expected fine (midpoint of range)
    fine_range: tuple[float, float]
    points: int
    net_value: float  # Value to this party (negative = cost)


class PartyValuations(BaseModel):
    party_id: str
    outcomes: dict[str, OutcomeValuation]
    litigation_cost: float
    status_quo: float  # Value of not litigating


class BATNA(BaseModel):
    party_id: str
    expected_value: float
    expected_value_range: tuple[float, float]  # Using CI bounds
    best_strategy: str | None
    outcome_probabilities: dict[str, float]


class SettlementRange(BaseModel):
    zopa: tuple[float, float] | None  # Zone of Possible Agreement
    nash_solution: float | None
    surplus: float
    settlement_exists: bool


class SensitivityResult(BaseModel):
    parameter: str
    base_value: float
    sweep_values: list[float]
    ev_at_each: list[float]
    threshold: float | None  # Where optimal strategy flips
    impact: float  # Max EV swing (for tornado ranking)


class GameTheoryAnalysis(BaseModel):
    party_valuations: dict[str, PartyValuations]
    batna: dict[str, BATNA]
    settlement: SettlementRange
    sensitivity: list[SensitivityResult]
    expected_value_by_strategy: dict[str, float]
    recommended_strategy: str | None
    analysis_metadata: dict
