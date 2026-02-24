from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


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


class JobCreateRequest(BaseModel):
    graph_path: str = Field(min_length=1)
    game: GameParamsModel = Field(default_factory=GameParamsModel)
    patrol: PatrolParamsModel = Field(default_factory=PatrolParamsModel)

    @field_validator("graph_path")
    @classmethod
    def validate_graph_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("graph_path cannot be empty")
        return normalized


class JobCreateResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    progress: float = Field(ge=0.0, le=1.0)
    message: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    summary: dict[str, float] | None = None
    error: str | None = None


class ScheduleRow(BaseModel):
    time_step: int
    unit_id: int
    node_id: str


class GraphResponse(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[dict[str, Any]]


class APIError(BaseModel):
    detail: str
