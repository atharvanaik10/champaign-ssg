from __future__ import annotations

import csv
import io
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from alma import GameParams, PatrolParams, generate_patrol_schedule

from backend.schemas import JobCreateRequest, JobStatusResponse

logger = logging.getLogger(__name__)

JobState = Literal["queued", "running", "done", "error"]


@dataclass
class JobRecord:
    job_id: str
    graph_path: str
    game: dict
    patrol: dict
    status: JobState = "queued"
    progress: float = 0.0
    message: str = "Queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    summary: dict[str, float] | None = None
    error: str | None = None
    schedule_json: list[dict] | None = None
    schedule_csv: str | None = None
    event_seq: int = 0

    def duration_seconds(self) -> float | None:
        if not self.started_at:
            return None
        end = self.finished_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()


class JobRegistry:
    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def create_job(self, request: JobCreateRequest) -> str:
        job_id = str(uuid4())
        record = JobRecord(
            job_id=job_id,
            graph_path=request.graph_path,
            game=request.game.model_dump(),
            patrol=request.patrol.model_dump(),
        )
        with self._condition:
            self._jobs[job_id] = record
            self._publish(record, progress=0.01, message="Queued")
        self._executor.submit(self._run_job, job_id)
        return job_id

    def _run_job(self, job_id: str) -> None:
        with self._condition:
            record = self._jobs[job_id]
            record.status = "running"
            record.started_at = datetime.now(timezone.utc)
            self._publish(record, progress=0.05, message="Loading graph...")

        try:
            game = GameParams(**record.game)
            patrol = PatrolParams(**record.patrol)
            graph_path = Path(record.graph_path)

            def callback(progress: float, message: str) -> None:
                with self._condition:
                    current = self._jobs[job_id]
                    self._publish(current, progress=progress, message=message)

            schedule_df, summary = generate_patrol_schedule(
                graph_path=graph_path,
                game_params=game,
                patrol_params=patrol,
                progress=callback,
            )

            schedule_json = schedule_df.to_dict(orient="records")
            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=["time_step", "unit_id", "node_id"])
            writer.writeheader()
            for row in schedule_json:
                writer.writerow(row)

            with self._condition:
                current = self._jobs[job_id]
                current.status = "done"
                current.progress = 1.0
                current.message = "Done"
                current.summary = summary
                current.finished_at = datetime.now(timezone.utc)
                current.schedule_json = schedule_json
                current.schedule_csv = csv_buffer.getvalue()
                current.event_seq += 1
                self._condition.notify_all()
            logger.info("Job %s completed", job_id)
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            with self._condition:
                current = self._jobs[job_id]
                current.status = "error"
                current.error = str(exc)
                current.message = "Failed"
                current.finished_at = datetime.now(timezone.utc)
                current.event_seq += 1
                self._condition.notify_all()

    def _publish(self, record: JobRecord, progress: float, message: str) -> None:
        record.progress = max(0.0, min(1.0, float(progress)))
        record.message = message
        record.event_seq += 1
        self._condition.notify_all()
        logger.info("job=%s progress=%.3f msg=%s", record.job_id, record.progress, record.message)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get_status(self, job_id: str) -> JobStatusResponse | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return JobStatusResponse(
                job_id=job.job_id,
                status=job.status,
                progress=job.progress,
                message=job.message,
                created_at=job.created_at,
                started_at=job.started_at,
                finished_at=job.finished_at,
                duration_seconds=job.duration_seconds(),
                summary=job.summary,
                error=job.error,
            )

    def wait_for_update(self, job_id: str, after_seq: int, timeout: float = 1.0) -> JobRecord | None:
        with self._condition:
            if job_id not in self._jobs:
                return None
            self._condition.wait_for(
                lambda: self._jobs[job_id].event_seq > after_seq or self._jobs[job_id].status in {"done", "error"},
                timeout=timeout,
            )
            return self._jobs.get(job_id)


registry: JobRegistry | None = None


def get_registry(max_workers: int = 4) -> JobRegistry:
    global registry
    if registry is None:
        registry = JobRegistry(max_workers=max_workers)
    return registry


def serialize_sse_progress(job: JobRecord) -> str:
    payload = {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
    }
    return f"event: progress\\ndata: {json.dumps(payload)}\\n\\n"
