from __future__ import annotations

import pandas as pd

from backend.jobs import JobRegistry
from backend.schemas import JobCreateRequest


def test_registry_job_completion_with_stubbed_solver(monkeypatch):
    def fake_schedule(**_kwargs):
        data = pd.DataFrame(
            [
                {"time_step": 0, "unit_id": 0, "node_id": "A"},
                {"time_step": 1, "unit_id": 0, "node_id": "B"},
            ]
        )
        return data, {"best_defender_utility": 1.23, "nodes": 2.0, "edges": 1.0}

    monkeypatch.setattr("backend.jobs.generate_patrol_schedule", fake_schedule)

    registry = JobRegistry(max_workers=1)
    request = JobCreateRequest(
        graph_path="data/uiuc_graph.json",
        patrol={"time_steps": 10, "num_units": 1, "random_seed": 1},
    )
    job_id = registry.create_job(request)

    seq = -1
    for _ in range(20):
        status = registry.wait_for_update(job_id, after_seq=seq, timeout=1.0)
        assert status is not None
        seq = status.event_seq
        if status.status in {"done", "error"}:
            break

    final = registry.get_status(job_id)
    assert final is not None
    assert final.status == "done"
    assert final.progress == 1.0
