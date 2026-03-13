# src/athena/api/models.py
"""Pydantic v2 models for the ATHENA API layer."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PipelineOptions(BaseModel):
    """Options that control pipeline behaviour."""

    concurrency: int | None = None
    kg_enabled: bool = False
    skip_meta_agents: bool = False
    skip_game_theory: bool = False
    rag_enabled: bool = False


class ProgressEvent(BaseModel):
    """A single progress event emitted during pipeline execution."""

    stage: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    detail: dict[str, Any] | None = None


class PipelineResult(BaseModel):
    """Complete result of a pipeline run."""

    case_id: str
    results: list[dict[str, Any]]
    aggregated: dict[str, Any]
    game_analysis: dict[str, Any] | None = None
    red_team: dict[str, Any] | None = None
    game_theorist: dict[str, Any] | None = None
    irac: dict[str, Any] | None = None
    memo: str = ""
    table_md: str = ""
    tree_txt: str = ""
    gt_summary_md: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class RunState(BaseModel):
    """Tracks the lifecycle of a single pipeline run."""

    run_id: str
    status: RunStatus = RunStatus.pending
    progress: list[ProgressEvent] = Field(default_factory=list)
    result: PipelineResult | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class RunRequest(BaseModel):
    """Incoming request to start a pipeline run."""

    case_data: dict[str, Any]
    sim_config: dict[str, Any]
    options: PipelineOptions = Field(default_factory=PipelineOptions)
