from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse

from backend.jobs import JobRegistry, get_registry, serialize_sse_progress
from backend.schemas import JobCreateRequest, JobCreateResponse, JobStatusResponse, ScheduleRow

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobCreateResponse)
def create_job(payload: JobCreateRequest, registry: JobRegistry = Depends(get_registry)) -> JobCreateResponse:
    job_id = registry.create_job(payload)
    return JobCreateResponse(job_id=job_id)


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, registry: JobRegistry = Depends(get_registry)) -> JobStatusResponse:
    status = registry.get_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")
    return status


@router.get("/{job_id}/schedule", response_model=list[ScheduleRow])
def get_schedule(
    job_id: str,
    format: str = Query(default="json", pattern="^(json|csv)$"),
    registry: JobRegistry = Depends(get_registry),
):
    job = registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")
    if job.status != "done" or job.schedule_json is None or job.schedule_csv is None:
        raise HTTPException(status_code=409, detail="Schedule not available yet")

    if format == "csv":
        return PlainTextResponse(job.schedule_csv, media_type="text/csv")
    return job.schedule_json


@router.get("/{job_id}/events")
async def stream_events(job_id: str, registry: JobRegistry = Depends(get_registry)) -> StreamingResponse:
    if registry.get(job_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")

    async def generator():
        seq = -1
        while True:
            job = await asyncio.to_thread(registry.wait_for_update, job_id, seq, 1.0)
            if job is None:
                break
            seq = job.event_seq
            yield serialize_sse_progress(job)
            if job.status in {"done", "error"}:
                break

    return StreamingResponse(generator(), media_type="text/event-stream")
