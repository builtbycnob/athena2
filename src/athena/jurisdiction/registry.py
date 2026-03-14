# src/athena/jurisdiction/registry.py
"""Jurisdiction registry — maps country code to prompts, schemas, outcome logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class JurisdictionConfig:
    """Configuration bundle for a legal jurisdiction."""

    country: str                          # "IT", "CH"
    prompt_keys: dict[str, str]           # role → prompt registry key
    schema_keys: dict[str, str]           # role → AGENT_SCHEMAS key
    verdict_schema_key: str               # AGENT_SCHEMAS key for judge verdict
    outcome_extractor: Callable[[dict], str]  # verdict dict → outcome string
    outcome_space: list[str]              # ["rejection", "annulment", ...]
    source_hierarchy: str                 # for prompt injection
    respondent_brief_label: str = "Memoria della controparte (depositata)"
    valuation_config: dict = field(default_factory=dict)
    default_temperatures: dict[str, float] = field(default_factory=dict)
    # Per-role model overrides (optional — role_type → model name)
    # e.g. {"judge": "qwen3.5-122b-a10b-4bit"} uses 122B for judge, default for parties
    default_models: dict[str, str] = field(default_factory=dict)
    # Two-step judge (optional — only CH uses this)
    judge_two_step: bool = False
    judge_step1_prompt_key: str | None = None
    judge_step1_schema_key: str | None = None
    judge_step1_temperature: float | None = None
    judge_step2_prompt_key: str | None = None
    judge_step2_schema_key: str | None = None
    judge_step2_temperature: float | None = None


_REGISTRY: dict[str, JurisdictionConfig] = {}


def register_jurisdiction(country: str, config: JurisdictionConfig) -> None:
    """Register a jurisdiction config."""
    _REGISTRY[country.upper()] = config


def get_jurisdiction(country: str) -> JurisdictionConfig:
    """Retrieve config by country code."""
    key = country.upper()
    if key not in _REGISTRY:
        raise KeyError(
            f"Jurisdiction '{key}' not registered. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key]


def get_jurisdiction_for_case(case_data: dict) -> JurisdictionConfig:
    """Auto-detect jurisdiction from case data."""
    country = case_data.get("jurisdiction", {}).get("country", "IT")
    return get_jurisdiction(country)


def list_jurisdictions() -> list[str]:
    """List registered jurisdiction codes."""
    return list(_REGISTRY.keys())
