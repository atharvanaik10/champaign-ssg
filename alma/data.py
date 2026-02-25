from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_graph(path: str | Path) -> nx.Graph:
    """Load a graph from a JSON adjacency list.

    The expected format is a mapping `node_id -> {lat, lon, risk_factor, neighbors}`.

    Args:
        path: Path to the JSON file.

    Returns:
        A simple, undirected NetworkX graph with node attributes: `lat`, `lon`,
        `risk_factor`, and optional `crimes`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Graph JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    graph = nx.Graph()

    for node_id, info in data.items():
        graph.add_node(
            node_id,
            lat=info.get("lat"),
            lon=info.get("lon"),
            risk_factor=info.get("risk_factor", 0.0),
            crimes=info.get("crimes", []),
        )

    for node_id, info in data.items():
        for neighbor in info.get("neighbors", []):
            if neighbor in data:
                graph.add_edge(node_id, neighbor)

    logger.info("Loaded graph: %s nodes, %s edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def get_node_list_and_risk(graph: nx.Graph) -> tuple[list[str], np.ndarray]:
    """Extract node IDs and per-node risk as a NumPy array.

    Args:
        graph: NetworkX graph with `risk_factor` stored on nodes.

    Returns:
        A tuple `(node_list, risk)` where `node_list` is a list of node IDs and
        `risk` is a float array aligned to `node_list`.
    """
    node_list = list(graph.nodes())
    risk = np.array(
        [graph.nodes[node_id].get("risk_factor", 0.0) for node_id in node_list],
        dtype=float,
    )
    logger.info("Total risk over nodes: %.2f", risk.sum())
    return node_list, risk


def load_graph_for_animation(path: str | Path) -> tuple[Dict[str, Tuple[float, float]], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """Load node coordinates and edge segments for lightweight map rendering.

    This avoids bringing NetworkX into callers that only need geometry.

    Args:
        path: Path to the JSON adjacency graph.

    Returns:
        nodes: Mapping of node_id -> (lon, lat).
        edges: List of ((lon1, lat1), (lon2, lat2)) tuples for each edge.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Graph JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    nodes: Dict[str, Tuple[float, float]] = {}
    for node_id, attrs in data.items():
        nodes[node_id] = (float(attrs["lon"]), float(attrs["lat"]))

    edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for node_id, attrs in data.items():
        x1, y1 = float(attrs["lon"]), float(attrs["lat"])
        for neighbor in attrs.get("neighbors", []):
            if node_id < neighbor and neighbor in data:
                x2, y2 = float(data[neighbor]["lon"]), float(data[neighbor]["lat"])
                edges.append(((x1, y1), (x2, y2)))

    return nodes, edges


def load_patrol_schedule(csv_path: str | Path) -> tuple[list[int], list[int], dict[int, dict[int, str]]]:
    """Load a patrol schedule CSV and index it for playback.

    The CSV is expected to have columns: `unit_id,time_step,node_id`.

    Args:
        csv_path: Path to the schedule CSV.

    Returns:
        units: List of unit IDs present in the schedule.
        timesteps: Sorted list of timesteps present.
        by_time: Mapping `t -> {unit_id -> node_id}` for quick lookup.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Patrol schedule not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[["unit_id", "time_step", "node_id"]]
    df = df.sort_values(["time_step", "unit_id"]).reset_index(drop=True)

    units = list(df["unit_id"].unique())
    timesteps = list(df["time_step"].unique())

    by_time: dict[int, dict[int, str]] = {}
    for t, sub in df.groupby("time_step"):
        by_time[int(t)] = dict(zip(sub["unit_id"].astype(int), sub["node_id"]))

    return units, timesteps, by_time
