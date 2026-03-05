from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from alma import GameParams, PatrolParams
from alma.schedule import generate_patrol_schedule_cached
from alma.data import load_graph_for_animation, load_graph, get_node_list_and_risk
from alma.evaluator import evaluate_schedule, generate_uniform_schedule
import asyncio
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Literal as TypingLiteral
from uuid import uuid4


class GameParamsModel(BaseModel):
    alpha: float = Field(default=1.0, ge=0.0)
    beta: float = Field(default=1.0, ge=0.0)
    gamma: float = Field(default=1.0, ge=0.0)
    delta: float = Field(default=1.0, ge=0.0)
    resource_budget: float = Field(default=10.0, gt=0.0)


class PatrolParamsModel(BaseModel):
    time_steps: int = Field(default=480, ge=1, le=10_000)
    num_units: int = Field(default=5, ge=1, le=1_000)
    start_index: int = Field(default=0, ge=0)
    random_seed: int = Field(default=0, ge=0)


class EvalParamsModel(BaseModel):
    p_event: float = Field(default=0.3, ge=0.0, le=1.0)
    num_runs: int = Field(default=200, ge=1, le=10000)


class PlanRequest(BaseModel):
    graph_path: str = Field(min_length=1)
    game: GameParamsModel = Field(default_factory=GameParamsModel)
    patrol: PatrolParamsModel = Field(default_factory=PatrolParamsModel)
    eval: EvalParamsModel = Field(default_factory=EvalParamsModel)


class ScheduleRow(BaseModel):
    time_step: int
    unit_id: int
    node_id: str


class PlanResponse(BaseModel):
    summary: dict[str, Any]
    schedule: list[ScheduleRow]


app = FastAPI(title="ALMA API (simple)", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- Synchronous plan API (convenience) --------
@app.post("/plan", response_model=PlanResponse)
def create_plan(payload: PlanRequest, format: Literal["json", "csv"] = Query("json")):
    graph_path = Path(payload.graph_path)
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail=f"Graph JSON not found: {graph_path}")

    game = GameParams(**payload.game.model_dump())
    patrol = PatrolParams(**payload.patrol.model_dump())
    df, summary = generate_patrol_schedule_cached(graph_path, game, patrol)
    # Evaluate policies for sync route as well
    graph = load_graph(graph_path)
    node_list, risk = get_node_list_and_risk(graph)
    eval_ssg = evaluate_schedule(
        df,
        node_list,
        risk,
        time_steps=patrol.time_steps,
        p_event=payload.eval.p_event,
        num_runs=payload.eval.num_runs,
        seed=patrol.random_seed,
    )
    df_uniform = generate_uniform_schedule(
        graph,
        node_list,
        time_steps=patrol.time_steps,
        num_units=patrol.num_units,
        seed=patrol.random_seed,
    )
    eval_uni = evaluate_schedule(
        df_uniform,
        node_list,
        risk,
        time_steps=patrol.time_steps,
        p_event=payload.eval.p_event,
        num_runs=payload.eval.num_runs,
        seed=patrol.random_seed,
    )
    # Efficiency vs units sweep (small caps for runtime)
    sweep_max = min(6, max(1, patrol.num_units))
    units_list = list(range(1, sweep_max + 1))
    ssg_means: list[float] = []
    uni_means: list[float] = []
    for u in units_list:
        patrol_u = PatrolParams(
            time_steps=patrol.time_steps,
            num_units=u,
            start_index=patrol.start_index,
            random_seed=patrol.random_seed,
        )
        df_u, _ = generate_patrol_schedule_cached(graph_path, game, patrol_u)
        m_ssg = evaluate_schedule(
            df_u,
            node_list,
            risk,
            time_steps=patrol.time_steps,
            p_event=payload.eval.p_event,
            num_runs=payload.eval.num_runs,
            seed=patrol.random_seed,
        ).get("efficiency_mean", 0.0)
        df_uni_u = generate_uniform_schedule(
            graph,
            node_list,
            time_steps=patrol.time_steps,
            num_units=u,
            seed=patrol.random_seed,
        )
        m_uni = evaluate_schedule(
            df_uni_u,
            node_list,
            risk,
            time_steps=patrol.time_steps,
            p_event=payload.eval.p_event,
            num_runs=payload.eval.num_runs,
            seed=patrol.random_seed,
        ).get("efficiency_mean", 0.0)
        ssg_means.append(float(m_ssg))
        uni_means.append(float(m_uni))
    summary.update({
        "efficiency_ssg_mean": float(eval_ssg.get("efficiency_mean", 0.0)),
        "efficiency_ssg_std": float(eval_ssg.get("efficiency_std", 0.0)),
        "efficiency_uniform_mean": float(eval_uni.get("efficiency_mean", 0.0)),
        "efficiency_uniform_std": float(eval_uni.get("efficiency_std", 0.0)),
        "p_event": float(payload.eval.p_event),
        "num_runs": float(payload.eval.num_runs),
        "eff_by_units_units": units_list,
        "eff_by_units_ssg": ssg_means,
        "eff_by_units_uniform": uni_means,
    })

    if format == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["time_step", "unit_id", "node_id"])
        writer.writeheader()
        for _, row in df.iterrows():
            writer.writerow({
                "time_step": int(row["time_step"]),
                "unit_id": int(row["unit_id"]),
                "node_id": str(row["node_id"]),
            })
        return PlainTextResponse(buf.getvalue(), media_type="text/csv")

    schedule = [
        ScheduleRow(time_step=int(r["time_step"]), unit_id=int(r["unit_id"]), node_id=str(r["node_id"]))
        for r in df.to_dict(orient="records")
    ]
    return PlanResponse(summary=summary, schedule=schedule)


@app.get("/graph")
def get_graph(graph_path: str = Query(..., min_length=1)) -> JSONResponse:
    try:
        nodes, edges = load_graph_for_animation(graph_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    features: list[dict[str, Any]] = []
    for idx, edge in enumerate(edges):
        p1, p2 = edge
        features.append(
            {
                "type": "Feature",
                "id": idx,
                "properties": {},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [list(p1), list(p2)],
                },
            }
        )
    for node_id, coords in nodes.items():
        features.append(
            {
                "type": "Feature",
                "properties": {"node_id": node_id},
                "geometry": {"type": "Point", "coordinates": [coords[0], coords[1]]},
            }
        )
    return JSONResponse({"type": "FeatureCollection", "features": features})


# -------- Minimal in-process job runner with progress (SSE) --------

JobState = TypingLiteral["queued", "running", "done", "error"]


class JobCreateResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobState
    progress: float
    message: str
    error: Optional[str] = None
    summary: Optional[dict[str, Any]] = None


@dataclass
class JobRecord:
    job_id: str
    payload: PlanRequest
    status: JobState = "queued"
    progress: float = 0.0
    message: str = "Queued"
    error: Optional[str] = None
    schedule_json: Optional[list[dict]] = None
    schedule_csv: Optional[str] = None
    summary: Optional[dict[str, float]] = None
    event_seq: int = 0


class Jobs:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def create(self, payload: PlanRequest) -> str:
        job_id = str(uuid4())
        rec = JobRecord(job_id=job_id, payload=payload)
        with self._cond:
            self._jobs[job_id] = rec
            self._publish(rec, 0.01, "Queued")
        self._executor.submit(self._run, job_id)
        return job_id

    def _run(self, job_id: str) -> None:
        with self._cond:
            rec = self._jobs[job_id]
            rec.status = "running"
            self._publish(rec, 0.05, "Loading graph...")
        try:
            p = rec.payload
            game = GameParams(**p.game.model_dump())
            patrol = PatrolParams(**p.patrol.model_dump())

            def cb(frac: float, msg: str) -> None:
                with self._cond:
                    curr = self._jobs[job_id]
                    self._publish(curr, frac, msg)

            df, summary = generate_patrol_schedule_cached(p.graph_path, game, patrol, progress=cb)
            schedule_json = [
                {
                    "time_step": int(r["time_step"]),
                    "unit_id": int(r["unit_id"]),
                    "node_id": str(r["node_id"]),
                }
                for r in df.to_dict(orient="records")
            ]
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=["time_step", "unit_id", "node_id"])
            writer.writeheader()
            for r in schedule_json:
                writer.writerow(r)

            # Policy evaluation (SSG schedule) and baseline (uniform)
            with self._cond:
                curr = self._jobs[job_id]
                self._publish(curr, 0.85, "Evaluating policy...")
            graph = load_graph(p.graph_path)
            node_list, risk = get_node_list_and_risk(graph)
            eval_ssg = evaluate_schedule(
                df,
                node_list,
                risk,
                time_steps=patrol.time_steps,
                p_event=p.eval.p_event,
                num_runs=p.eval.num_runs,
                seed=patrol.random_seed,
            )
            with self._cond:
                curr = self._jobs[job_id]
                self._publish(curr, 0.92, "Evaluating baseline...")
            df_uniform = generate_uniform_schedule(
                graph,
                node_list,
                time_steps=patrol.time_steps,
                num_units=patrol.num_units,
                seed=patrol.random_seed,
            )
            eval_uni = evaluate_schedule(
                df_uniform,
                node_list,
                risk,
                time_steps=patrol.time_steps,
                p_event=p.eval.p_event,
                num_runs=p.eval.num_runs,
                seed=patrol.random_seed,
            )
            # Sweep
            sweep_max = min(6, max(1, patrol.num_units))
            units_list = list(range(1, sweep_max + 1))
            ssg_means: list[float] = []
            uni_means: list[float] = []
            for u in units_list:
                patrol_u = PatrolParams(
                    time_steps=patrol.time_steps,
                    num_units=u,
                    start_index=patrol.start_index,
                    random_seed=patrol.random_seed,
                )
                df_u, _ = generate_patrol_schedule_cached(p.graph_path, game, patrol_u)
                m_ssg = evaluate_schedule(
                    df_u,
                    node_list,
                    risk,
                    time_steps=patrol.time_steps,
                    p_event=p.eval.p_event,
                    num_runs=p.eval.num_runs,
                    seed=patrol.random_seed,
                ).get("efficiency_mean", 0.0)
                df_uni_u = generate_uniform_schedule(
                    graph,
                    node_list,
                    time_steps=patrol.time_steps,
                    num_units=u,
                    seed=patrol.random_seed,
                )
                m_uni = evaluate_schedule(
                    df_uni_u,
                    node_list,
                    risk,
                    time_steps=patrol.time_steps,
                    p_event=p.eval.p_event,
                    num_runs=p.eval.num_runs,
                    seed=patrol.random_seed,
                ).get("efficiency_mean", 0.0)
                ssg_means.append(float(m_ssg))
                uni_means.append(float(m_uni))

            with self._cond:
                curr = self._jobs[job_id]
                curr.status = "done"
                curr.progress = 1.0
                curr.message = "Done"
                summary.update({
                    "efficiency_ssg_mean": float(eval_ssg.get("efficiency_mean", 0.0)),
                    "efficiency_ssg_std": float(eval_ssg.get("efficiency_std", 0.0)),
                    "efficiency_uniform_mean": float(eval_uni.get("efficiency_mean", 0.0)),
                    "efficiency_uniform_std": float(eval_uni.get("efficiency_std", 0.0)),
                    "p_event": float(p.eval.p_event),
                    "num_runs": float(p.eval.num_runs),
                    "eff_by_units_units": units_list,
                    "eff_by_units_ssg": ssg_means,
                    "eff_by_units_uniform": uni_means,
                })
                curr.summary = summary
                curr.schedule_json = schedule_json
                curr.schedule_csv = buf.getvalue()
                curr.event_seq += 1
                self._cond.notify_all()
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).exception("Job failed")
            with self._cond:
                curr = self._jobs[job_id]
                curr.status = "error"
                curr.error = str(exc)
                curr.message = "Failed"
                curr.event_seq += 1
                self._cond.notify_all()

    def _publish(self, rec: JobRecord, progress: float, message: str) -> None:
        rec.progress = max(0.0, min(1.0, float(progress)))
        rec.message = message
        rec.event_seq += 1
        self._cond.notify_all()

    def status(self, job_id: str) -> JobStatusResponse | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return None
            return JobStatusResponse(
                job_id=rec.job_id,
                status=rec.status,
                progress=rec.progress,
                message=rec.message,
                error=rec.error,
                summary=rec.summary,
            )

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def wait_update(self, job_id: str, after_seq: int, timeout: float = 1.0) -> JobRecord | None:
        with self._cond:
            if job_id not in self._jobs:
                return None
            self._cond.wait_for(lambda: self._jobs[job_id].event_seq > after_seq, timeout=timeout)
            return self._jobs.get(job_id)


JOBS = Jobs(max_workers=2)


def _sse_progress_payload(job: JobRecord) -> str:
    payload = {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
    }
    return f"event: progress\ndata: {json.dumps(payload)}\n\n"


@app.post("/jobs", response_model=JobCreateResponse)
def jobs_create(payload: PlanRequest) -> JobCreateResponse:
    job_id = JOBS.create(payload)
    return JobCreateResponse(job_id=job_id)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def jobs_status(job_id: str) -> JobStatusResponse:
    status = JOBS.status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")
    return status


@app.get("/jobs/{job_id}/schedule")
def jobs_schedule(job_id: str, format: Literal["json", "csv"] = Query("json")):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")
    if job.status != "done" or job.schedule_json is None or job.schedule_csv is None:
        raise HTTPException(status_code=409, detail="Schedule not available yet")
    if format == "csv":
        return PlainTextResponse(job.schedule_csv, media_type="text/csv")
    return JSONResponse(job.schedule_json)


@app.get("/jobs/{job_id}/events")
async def jobs_events(job_id: str) -> StreamingResponse:
    if JOBS.get(job_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")

    async def gen():
        seq = -1
        while True:
            job = await asyncio.to_thread(JOBS.wait_update, job_id, seq, 1.0)
            if job is None:
                break
            seq = job.event_seq
            yield _sse_progress_payload(job)
            if job.status in {"done", "error"}:
                break

    return StreamingResponse(gen(), media_type="text/event-stream")


# Serve the built web app if present (web/dist)
dist_path = Path(__file__).parent / "web" / "dist"
if dist_path.exists():
    app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="static")
