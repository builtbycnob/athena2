# src/athena/api/app.py
"""FastAPI application for ATHENA.

Endpoints:
    GET  /health          — process + oMLX + KG health
    POST /runs            — start a pipeline run (202 Accepted)
    GET  /runs            — list all runs
    GET  /runs/{id}       — get run state
    GET  /runs/{id}/stream — SSE progress stream
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

import httpx

from athena.api.models import (
    PipelineOptions,
    PipelineResult,
    ProgressEvent,
    RunRequest,
    RunState,
    RunStatus,
)
from athena.api import registry


def create_app():
    """Create and configure the FastAPI application."""
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse
    from sse_starlette.sse import EventSourceResponse

    app = FastAPI(
        title="ATHENA API",
        description="Adversarial Tactical Hearing & Equilibrium Navigation Agent",
        version="1.2.0",
    )

    @app.get("/health")
    async def health():
        """Two-tier health check: process alive + oMLX inference backend."""
        result = {"status": "ok", "process": "alive"}

        # Check oMLX
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("http://localhost:8000/v1/models")
                if resp.status_code == 200:
                    result["omlx"] = "ok"
                else:
                    result["omlx"] = "degraded"
        except Exception:
            result["omlx"] = "unavailable"

        # Check KG
        try:
            from athena.knowledge.config import is_kg_enabled
            if is_kg_enabled():
                from athena.knowledge.config import health_check
                kg = health_check()
                result["kg"] = kg.get("status", "unknown")
            else:
                result["kg"] = "disabled"
        except Exception:
            result["kg"] = "unavailable"

        return result

    @app.post("/runs", status_code=202)
    async def create_run(request: RunRequest):
        """Start a new pipeline run. Returns 202 with run_id."""
        run_id = registry.create_run()

        # Launch pipeline in background thread
        asyncio.ensure_future(_run_pipeline_async(run_id, request))

        return {"run_id": run_id, "status": "pending"}

    @app.get("/runs")
    async def list_runs(include_results: bool = Query(False)):
        """List all runs."""
        runs = registry.list_runs(include_results=include_results)
        return [r.model_dump() for r in runs]

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str):
        """Get run state by ID."""
        run = registry.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        return run.model_dump()

    @app.get("/runs/{run_id}/stream")
    async def stream_run(run_id: str):
        """SSE stream of progress events for a run."""
        run = registry.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        queue = registry.get_queue(run_id)

        async def event_generator():
            # First replay any existing progress events
            current = registry.get_run(run_id)
            if current:
                for event in current.progress:
                    yield {
                        "event": "progress",
                        "data": event.model_dump_json(),
                    }

            # If already done, send final event and stop
            if current and current.status in (RunStatus.completed, RunStatus.failed):
                yield {
                    "event": "done",
                    "data": json.dumps({"status": current.status.value}),
                }
                return

            # Stream live events
            if queue is not None:
                while True:
                    event = await queue.get()
                    if event is None:
                        # Sentinel — run finished
                        final = registry.get_run(run_id)
                        yield {
                            "event": "done",
                            "data": json.dumps({
                                "status": final.status.value if final else "unknown",
                            }),
                        }
                        return
                    yield {
                        "event": "progress",
                        "data": event.model_dump_json(),
                    }

        return EventSourceResponse(event_generator())

    return app


async def _run_pipeline_async(run_id: str, request: RunRequest) -> None:
    """Run the pipeline in a background thread, bridging progress to the registry."""
    from athena.api.pipeline import prepare_case_data, prepare_sim_config, run_pipeline

    registry.mark_running(run_id)

    def progress_callback(event: ProgressEvent) -> None:
        registry.push_event(run_id, event)

    try:
        case_data = prepare_case_data(request.case_data)
        sim_config = prepare_sim_config(request.sim_config)

        result = await asyncio.to_thread(
            run_pipeline,
            case_data,
            sim_config,
            request.options,
            progress_callback,
        )

        registry.mark_completed(run_id, result)
    except Exception as e:
        registry.mark_failed(run_id, str(e))
    finally:
        registry.push_sentinel(run_id)
