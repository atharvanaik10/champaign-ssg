from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

import backend.jobs as jobs_module
from backend.main import app


client = TestClient(app)


def _reset_registry() -> None:
    jobs_module.registry = jobs_module.JobRegistry(max_workers=2)


def _stub_solver(monkeypatch):
    def fake_schedule(**_kwargs):
        data = pd.DataFrame(
            [
                {"time_step": 0, "unit_id": 0, "node_id": "A"},
                {"time_step": 1, "unit_id": 0, "node_id": "B"},
            ]
        )
        return data, {"best_defender_utility": 1.23, "nodes": 2.0, "edges": 1.0}

    monkeypatch.setattr("backend.jobs.generate_patrol_schedule", fake_schedule)


def test_job_lifecycle_happy_path(monkeypatch):
    _stub_solver(monkeypatch)
    _reset_registry()
    payload = {
        "graph_path": "data/uiuc_graph.json",
        "game": {"resource_budget": 10},
        "patrol": {"time_steps": 20, "num_units": 2, "random_seed": 1},
    }
    create = client.post("/jobs", json=payload)
    assert create.status_code == 200
    job_id = create.json()["job_id"]

    status = client.get(f"/jobs/{job_id}").json()
    assert status["status"] in {"queued", "running", "done"}

    for _ in range(20):
        data = client.get(f"/jobs/{job_id}").json()
        if data["status"] == "done":
            break

    done = client.get(f"/jobs/{job_id}").json()
    assert done["status"] == "done"
    assert done["summary"]["best_defender_utility"] is not None

    schedule_json = client.get(f"/jobs/{job_id}/schedule", params={"format": "json"})
    assert schedule_json.status_code == 200
    assert len(schedule_json.json()) > 0

    schedule_csv = client.get(f"/jobs/{job_id}/schedule", params={"format": "csv"})
    assert schedule_csv.status_code == 200
    assert "time_step,unit_id,node_id" in schedule_csv.text


def test_invalid_input_returns_400():
    _reset_registry()
    payload = {
        "graph_path": "data/uiuc_graph.json",
        "patrol": {"time_steps": 0, "num_units": 0},
    }
    response = client.post("/jobs", json=payload)
    assert response.status_code == 400


def test_graph_endpoint():
    response = client.get("/graph", params={"graph_path": "data/uiuc_graph.json"})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert isinstance(data["features"], list)
