# src/athena/api/registry.py
"""In-memory run state registry for the API server.

Thread-safe via threading.Lock. Each run gets an asyncio.Queue for SSE streaming.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from datetime import datetime

from athena.api.models import (
    PipelineOptions,
    PipelineResult,
    ProgressEvent,
    RunState,
    RunStatus,
)

_runs: dict[str, RunState] = {}
_queues: dict[str, asyncio.Queue[ProgressEvent | None]] = {}
_lock = threading.Lock()


def create_run() -> str:
    """Create a new run entry and return its ID."""
    run_id = uuid.uuid4().hex[:12]
    with _lock:
        _runs[run_id] = RunState(run_id=run_id)
        _queues[run_id] = asyncio.Queue()
    return run_id


def get_run(run_id: str) -> RunState | None:
    """Get run state by ID. Returns None if not found."""
    with _lock:
        return _runs.get(run_id)


def list_runs(include_results: bool = False) -> list[RunState]:
    """List all runs. Strips result bodies unless include_results=True."""
    with _lock:
        runs = list(_runs.values())
    if not include_results:
        # Return copies without large result bodies
        stripped = []
        for r in runs:
            copy = r.model_copy()
            if copy.result is not None:
                copy.result = None
            stripped.append(copy)
        return stripped
    return runs


def get_queue(run_id: str) -> asyncio.Queue[ProgressEvent | None] | None:
    """Get the SSE event queue for a run."""
    with _lock:
        return _queues.get(run_id)


def mark_running(run_id: str) -> None:
    """Mark run as running."""
    with _lock:
        if run_id in _runs:
            _runs[run_id].status = RunStatus.running
            _runs[run_id].started_at = datetime.utcnow()


def mark_completed(run_id: str, result: PipelineResult) -> None:
    """Mark run as completed with result."""
    with _lock:
        if run_id in _runs:
            _runs[run_id].status = RunStatus.completed
            _runs[run_id].result = result
            _runs[run_id].completed_at = datetime.utcnow()


def mark_failed(run_id: str, error: str) -> None:
    """Mark run as failed with error message."""
    with _lock:
        if run_id in _runs:
            _runs[run_id].status = RunStatus.failed
            _runs[run_id].error = error
            _runs[run_id].completed_at = datetime.utcnow()


def push_event(run_id: str, event: ProgressEvent) -> None:
    """Push a progress event to the run's queue and history."""
    with _lock:
        if run_id in _runs:
            _runs[run_id].progress.append(event)
        q = _queues.get(run_id)
    if q is not None:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


def push_sentinel(run_id: str) -> None:
    """Push None sentinel to signal SSE stream end."""
    with _lock:
        q = _queues.get(run_id)
    if q is not None:
        try:
            q.put_nowait(None)
        except asyncio.QueueFull:
            pass


def reset() -> None:
    """Clear all state (for testing)."""
    with _lock:
        _runs.clear()
        _queues.clear()
