from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import pandas as pd

from alma.patrol import build_uniform_transition_matrix, simulate_patrol, pick_diverse_start_nodes


def compute_schedule_metrics(
    schedule_df: pd.DataFrame,
    node_list: list[str],
) -> dict[str, object]:
    """Compute deterministic movement and coverage metrics for a schedule.

    Movement is counted in hops (number of transitions where the node changes)
    per unit; coverage counts unique nodes visited.

    Args:
        schedule_df: DataFrame with columns [time_step, unit_id, node_id].
        node_list: All node IDs present in the graph (defines total nodes).

    Returns:
        Dict containing:
          - movement_total_hops: int
          - movement_by_unit_hops: list[int]
          - coverage_total_ratio: float (unique nodes / total nodes)
          - coverage_total_count: int (unique nodes)
          - coverage_total_nodes: int (len(node_list))
          - coverage_by_unit_ratio: list[float]
          - coverage_by_unit_count: list[int]
    """
    if schedule_df.empty:
        total_nodes = len(node_list)
        return {
            "movement_total_hops": 0,
            "movement_by_unit_hops": [],
            "coverage_total_ratio": 0.0 if total_nodes > 0 else 0.0,
            "coverage_total_count": 0,
            "coverage_total_nodes": total_nodes,
            "coverage_by_unit_ratio": [],
            "coverage_by_unit_count": [],
        }

    df = schedule_df[["time_step", "unit_id", "node_id"]].copy()
    df = df.sort_values(["unit_id", "time_step"]).reset_index(drop=True)

    # Per-unit hops: count transitions where node changes between consecutive timesteps
    movement_by_unit: list[int] = []
    coverage_by_unit_counts: list[int] = []
    total_nodes = len(node_list)

    for unit_id, sub in df.groupby("unit_id"):
        nodes = sub["node_id"].astype(str).tolist()
        hops = 0
        if len(nodes) >= 2:
            prev = nodes[0]
            for curr in nodes[1:]:
                if curr != prev:
                    hops += 1
                prev = curr
        movement_by_unit.append(int(hops))
        coverage_by_unit_counts.append(int(len(set(nodes))))

    movement_total = int(sum(movement_by_unit))
    # Total coverage across all units
    coverage_total_nodes_set = set(df["node_id"].astype(str).tolist())
    coverage_total_count = int(len(coverage_total_nodes_set))
    coverage_total_ratio = float(coverage_total_count / total_nodes) if total_nodes > 0 else 0.0

    coverage_by_unit_ratio = [
        (c / total_nodes) if total_nodes > 0 else 0.0 for c in coverage_by_unit_counts
    ]

    return {
        "movement_total_hops": movement_total,
        "movement_by_unit_hops": [int(x) for x in movement_by_unit],
        "coverage_total_ratio": float(coverage_total_ratio),
        "coverage_total_count": int(coverage_total_count),
        "coverage_total_nodes": int(total_nodes),
        "coverage_by_unit_ratio": [float(x) for x in coverage_by_unit_ratio],
        "coverage_by_unit_count": [int(x) for x in coverage_by_unit_counts],
    }


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

    The crime model generates an event at each timestep with probability
    `p_event`. Event location is drawn with probability proportional to risk.
    An event is considered prevented iff a patrol unit occupies the node at the
    same timestep. Efficiency is defined as `caught_risk / total_risk`.

    Args:
        schedule_df: DataFrame with columns [time_step, unit_id, node_id].
        node_list: All node IDs (risk/indices are aligned to this list).
        risk: Risk array aligned to `node_list`.
        time_steps: Maximum timestep T in the schedule.
        p_event: Probability a crime occurs at each timestep.
        num_runs: Number of Monte Carlo runs to average over.
        seed: Seed for reproducibility.

    Returns:
        Dict with keys: `efficiency_mean`, `efficiency_std`, `runs`.
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
    """Generate a baseline uniform random walk schedule with diverse starts.

    Args:
        graph: NetworkX graph.
        node_list: Node IDs aligned to the transition matrix.
        time_steps: Simulate from 0..T (inclusive).
        num_units: Number of patrol units.
        seed: Seed for reproducible start selection.

    Returns:
        DataFrame with columns [time_step, unit_id, node_id].
    """
    matrix = build_uniform_transition_matrix(graph, node_list)
    starts = pick_diverse_start_nodes(graph, node_list, num_units, seed=seed)
    idx = {nid: i for i, nid in enumerate(node_list)}
    start_indices = [idx[n] for n in starts]
    records = simulate_patrol(matrix, node_list, start_idx=start_indices, time_steps=time_steps, num_units=num_units)
    return pd.DataFrame(records, columns=["time_step", "unit_id", "node_id"])
