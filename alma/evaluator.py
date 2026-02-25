from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import pandas as pd

from alma.patrol import build_uniform_transition_matrix, simulate_patrol, pick_diverse_start_nodes


def evaluate_schedule(
    schedule_df: pd.DataFrame,
    node_list: list[str],
    risk: np.ndarray,
    time_steps: int,
    p_event: float = 0.3,
    num_runs: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Evaluate a concrete schedule by Monte Carlo.

    Builds a mapping of timestep -> set(node_id) from the schedule and simulates
    crime events occurring with probability p_event at each timestep, located
    with probability proportional to node risk. Returns mean and std of
    efficiency = caught_risk / total_risk over runs with >=1 crime.
    """
    risk = np.array(risk, float)
    if risk.sum() <= 0:
        return {"efficiency_mean": 0.0, "efficiency_std": 0.0, "runs": 0.0}
    crime_probs = risk / risk.sum()

    # Construct patrol occupancy per timestep
    sched = schedule_df[["time_step", "unit_id", "node_id"]]
    sched = sched.sort_values(["time_step", "unit_id"]).reset_index(drop=True)
    patrol_by_time: dict[int, set[str]] = {int(t): set() for t in range(time_steps + 1)}
    for t, sub in sched.groupby("time_step"):
        patrol_by_time[int(t)] = set(str(n) for n in sub["node_id"].tolist())

    efficiencies: list[float] = []
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    n = len(node_list)

    for _ in range(num_runs):
        caught, total = 0.0, 0.0
        for t in range(time_steps + 1):
            if py_rng.random() < p_event:
                j = int(rng.choice(n, p=crime_probs))
                nid = node_list[j]
                rj = risk[j]
                total += rj
                if nid in patrol_by_time.get(t, set()):
                    caught += rj
        if total > 0:
            efficiencies.append(caught / total)

    if not efficiencies:
        return {"efficiency_mean": 0.0, "efficiency_std": 0.0, "runs": 0.0}
    arr = np.array(efficiencies, float)
    return {
        "efficiency_mean": float(arr.mean()),
        "efficiency_std": float(arr.std()),
        "runs": float(len(arr)),
    }


def generate_uniform_schedule(
    graph,
    node_list: list[str],
    time_steps: int,
    num_units: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a baseline uniform random walk schedule with diverse starts."""
    matrix = build_uniform_transition_matrix(graph, node_list)
    starts = pick_diverse_start_nodes(graph, node_list, num_units, seed=seed)
    idx = {nid: i for i, nid in enumerate(node_list)}
    start_indices = [idx[n] for n in starts]
    records = simulate_patrol(matrix, node_list, start_idx=start_indices, time_steps=time_steps, num_units=num_units)
    return pd.DataFrame(records, columns=["time_step", "unit_id", "node_id"])

