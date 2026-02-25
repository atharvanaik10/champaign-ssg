from __future__ import annotations

import logging
from dataclasses import asdict
import hashlib
import json
import os
from tempfile import NamedTemporaryFile
from pathlib import Path

import numpy as np
import pandas as pd

from alma.config import GameParams, PatrolParams
from alma.data import get_node_list_and_risk, load_graph
from typing import Callable, Optional
from alma.patrol import build_transition_matrix, simulate_patrol, pick_diverse_start_nodes
from alma.ssg import build_payoffs_from_risk, solve_ssg

logger = logging.getLogger(__name__)


def generate_patrol_schedule(
    graph_path: str | Path,
    game_params: GameParams,
    patrol_params: PatrolParams,
    progress: Optional[Callable[[float, str], None]] = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Generate a patrol schedule and summary statistics.

    This function loads the graph, solves the SSG to produce a per-node coverage
    vector, builds a coverage-biased transition matrix, and simulates a
    multi‑unit patrol for `time_steps` timesteps. When multiple units are used,
    diverse starting nodes are chosen to spread coverage.

    Args:
        graph_path: Path to the JSON adjacency graph.
        game_params: SSG parameters (alpha/beta/gamma/delta, budget).
        patrol_params: Patrol simulation parameters (T, units, seed, start).
        progress: Optional callback receiving `(fraction, message)` updates.

    Returns:
        (schedule_df, summary) where:
          - schedule_df: DataFrame with columns [time_step, unit_id, node_id].
          - summary: Dict of scalar metrics (utility, num nodes/edges, params).
    """
    np.random.seed(patrol_params.random_seed)

    def report(frac: float, msg: str):
        if progress is not None:
            try:
                progress(max(0.0, min(1.0, float(frac))), msg)
            except Exception:
                # Keep core logic robust if the callback errors
                pass

    report(0.05, "Loading graph...")
    graph = load_graph(graph_path)
    node_list, risk = get_node_list_and_risk(graph)

    report(0.20, "Building payoffs from risk...")
    R_d, P_d, R_a, P_a = build_payoffs_from_risk(
        risk,
        game_params.alpha,
        game_params.beta,
        game_params.gamma,
        game_params.delta,
    )
    report(0.40, "Solving SSG (optimizer)...")
    coverage, best_utility = solve_ssg(
        R_d,
        P_d,
        R_a,
        P_a,
        np.ones_like(risk),
        game_params.resource_budget,
    )

    report(0.60, "Building transition matrix...")
    matrix = build_transition_matrix(graph, node_list, coverage)
    report(0.70, "Simulating patrol...")
    # Choose diverse starting nodes for each unit
    if patrol_params.num_units > 1:
        start_nodes = pick_diverse_start_nodes(graph, node_list, patrol_params.num_units, seed=patrol_params.random_seed)
        idx_map = {nid: i for i, nid in enumerate(node_list)}
        start_indices = [idx_map[nid] for nid in start_nodes]
    else:
        start_indices = patrol_params.start_index
    schedule = simulate_patrol(
        matrix,
        node_list,
        start_idx=start_indices,
        time_steps=patrol_params.time_steps,
        num_units=patrol_params.num_units,
        progress=lambda f, _m: report(0.70 + 0.25 * f, "Simulating patrol..."),
    )
    df = pd.DataFrame(schedule, columns=["time_step", "unit_id", "node_id"])

    summary = {
        "best_defender_utility": float(best_utility),
        "nodes": float(len(node_list)),
        "edges": float(graph.number_of_edges()),
    }
    summary.update({f"game_{k}": float(v) for k, v in asdict(game_params).items()})
    summary.update({f"patrol_{k}": float(v) for k, v in asdict(patrol_params).items()})

    logger.info("Generated patrol schedule: %s rows", len(df))
    report(1.0, "Done")
    return df, summary


def _cache_key_for_inputs(graph_path: str | Path, game_params: GameParams, patrol_params: PatrolParams) -> str:
    """Compute a stable content-based cache key for inputs.

    Hashes the graph file bytes, the serialized parameter dicts, and a version
    token to produce a short hex key suitable for filenames.

    Args:
        graph_path: Path to graph JSON.
        game_params: Game parameter object.
        patrol_params: Patrol parameter object.

    Returns:
        A short hex string suitable for cache filenames.
    """
    """Compute a stable cache key from graph content and parameters."""
    path = Path(graph_path)
    h = hashlib.sha256()
    # Hash file contents for stability across paths
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    h.update(json.dumps(asdict(game_params), sort_keys=True).encode("utf-8"))
    h.update(json.dumps(asdict(patrol_params), sort_keys=True).encode("utf-8"))
    # Include a version token to invalidate if algorithms change
    h.update(b"alma-schedule-v1")
    return h.hexdigest()[:24]


def generate_patrol_schedule_cached(
    graph_path: str | Path,
    game_params: GameParams,
    patrol_params: PatrolParams,
    *,
    cache_dir: str | Path = "cache",
    use_cache: bool = True,
    progress: Optional[Callable[[float, str], None]] = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Cached wrapper around `generate_patrol_schedule`.

    Caches by hashing the graph file contents and the parameter values. If a
    matching result is found, loads the cached CSV/JSON and returns immediately.
    Otherwise, computes and stores the result for future runs.

    Args:
        graph_path: Path to the JSON adjacency graph.
        game_params: SSG parameters.
        patrol_params: Patrol simulation parameters.
        cache_dir: Directory to store CSV/JSON cache files.
        use_cache: Whether to read/write the cache.
        progress: Optional callback receiving `(fraction, message)` updates.

    Returns:
        (schedule_df, summary) as described in `generate_patrol_schedule`.
    """
    def report(frac: float, msg: str):
        if progress is not None:
            try:
                progress(max(0.0, min(1.0, float(frac))), msg)
            except Exception:
                pass

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    key = _cache_key_for_inputs(graph_path, game_params, patrol_params)
    csv_path = cache_path / f"schedule_{key}.csv"
    meta_path = cache_path / f"summary_{key}.json"

    if use_cache and csv_path.exists() and meta_path.exists():
        report(0.05, "Checking cache...")
        try:
            df = pd.read_csv(csv_path)
            with meta_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            report(1.0, "Loaded from cache")
            return df, {k: float(v) for k, v in summary.items()}
        except Exception:
            # Fall back to computing if cache is corrupted
            pass

    # Miss: compute and store
    def _forward(frac: float, msg: str):
        # Reserve a small headroom for write time
        report(0.02 + 0.95 * frac, msg)

    df, summary = generate_patrol_schedule(
        graph_path=graph_path,
        game_params=game_params,
        patrol_params=patrol_params,
        progress=_forward,
    )

    # Atomic writes
    try:
        with NamedTemporaryFile("w", delete=False, dir=str(cache_path), suffix=".csv") as tf:
            df.to_csv(tf.name, index=False)
            tmp_csv = tf.name
        os.replace(tmp_csv, csv_path)

        with NamedTemporaryFile("w", delete=False, dir=str(cache_path), suffix=".json", encoding="utf-8") as tf:
            json.dump(summary, tf, ensure_ascii=False)
            tmp_json = tf.name
        os.replace(tmp_json, meta_path)
    except Exception:
        # Ignore cache write failures to avoid breaking primary path
        pass

    report(1.0, "Done")
    return df, summary
