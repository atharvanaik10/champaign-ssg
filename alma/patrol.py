from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional

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
    start_idx: int,
    time_steps: int,
    num_units: int = 1,
    progress: Optional[Callable[[float, str], None]] = None,
) -> list[tuple[int, int, str]]:
    n = len(node_list)
    current = [start_idx] * num_units
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
