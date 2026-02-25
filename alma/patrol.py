from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


def build_transition_matrix(graph: nx.Graph, node_list: list[str], coverage: np.ndarray) -> np.ndarray:
    idx = {node_id: i for i, node_id in enumerate(node_list)}
    n = len(node_list)
    matrix = np.zeros((n, n))

    for u_i, node_id in enumerate(node_list):
        neighbors = list(graph.neighbors(node_id))
        if not neighbors:
            matrix[u_i, u_i] = 1.0
            continue

        weights = np.array([coverage[idx[v]] for v in neighbors], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights /= weights.sum()

        for w_i, neighbor in enumerate(neighbors):
            matrix[u_i, idx[neighbor]] = weights[w_i]

    if not np.allclose(matrix.sum(axis=1), 1.0):
        logger.warning("Transition matrix rows do not sum to 1.")
    return matrix


def build_uniform_transition_matrix(graph: nx.Graph, node_list: list[str]) -> np.ndarray:
    idx = {node_id: i for i, node_id in enumerate(node_list)}
    n = len(node_list)
    matrix = np.zeros((n, n))

    for u_i, node_id in enumerate(node_list):
        neighbors = list(graph.neighbors(node_id))
        if not neighbors:
            matrix[u_i, u_i] = 1.0
            continue
        probs = np.ones(len(neighbors)) / len(neighbors)
        for w_i, neighbor in enumerate(neighbors):
            matrix[u_i, idx[neighbor]] = probs[w_i]

    if not np.allclose(matrix.sum(axis=1), 1.0):
        logger.warning("Uniform transition matrix rows do not sum to 1.")
    return matrix


def simulate_patrol(
    matrix: np.ndarray,
    node_list: list[str],
    start_idx: int | Sequence[int],
    time_steps: int,
    num_units: int = 1,
    progress: Optional[Callable[[float, str], None]] = None,
) -> list[tuple[int, int, str]]:
    n = len(node_list)
    if isinstance(start_idx, (list, tuple, np.ndarray)):
        if len(start_idx) != num_units:
            raise ValueError("start_idx list length must equal num_units")
        current = [int(s) % n for s in start_idx]
    else:
        current = [int(start_idx) % n] * num_units
    records: list[tuple[int, int, str]] = []

    update_every = max(1, (time_steps + 1) // 50)
    for t in range(time_steps + 1):
        for unit in range(num_units):
            records.append((t, unit, node_list[current[unit]]))
        if t < time_steps:
            for unit in range(num_units):
                current[unit] = int(np.random.choice(n, p=matrix[current[unit]]))
        if progress is not None and (t % update_every == 0 or t == time_steps):
            progress(t / float(time_steps if time_steps > 0 else 1), "Simulating patrol...")
    return records


def pick_diverse_start_nodes(graph: nx.Graph, node_list: list[str], k: int, seed: int = 0) -> list[str]:
    import random

    rng = random.Random(seed)
    first = rng.choice(node_list)
    chosen: list[str] = [first]

    for _ in range(1, max(1, k)):
        nearest: dict[str, float] = {nid: float("inf") for nid in node_list}
        for c in chosen:
            dist = nx.single_source_shortest_path_length(graph, c)
            for nid, d in dist.items():
                if d < nearest[nid]:
                    nearest[nid] = d
        best_node: Optional[str] = None
        best_dist = -1.0
        for nid in node_list:
            d = nearest.get(nid, float("inf"))
            if d != float("inf") and d > best_dist:
                best_dist = d
                best_node = nid
        if best_node is None:
            best_node = rng.choice(node_list)
        chosen.append(best_node)
    return chosen[:k]
