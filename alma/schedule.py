from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from alma.config import GameParams, PatrolParams
from alma.data import get_node_list_and_risk, load_graph
from alma.patrol import build_transition_matrix, simulate_patrol
from alma.ssg import build_payoffs_from_risk, solve_ssg

logger = logging.getLogger(__name__)


def generate_patrol_schedule(
    graph_path: str | Path,
    game_params: GameParams,
    patrol_params: PatrolParams,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Generate a patrol schedule and return the schedule plus summary stats."""
    np.random.seed(patrol_params.random_seed)

    graph = load_graph(graph_path)
    node_list, risk = get_node_list_and_risk(graph)

    R_d, P_d, R_a, P_a = build_payoffs_from_risk(
        risk,
        game_params.alpha,
        game_params.beta,
        game_params.gamma,
        game_params.delta,
    )
    coverage, best_utility = solve_ssg(
        R_d,
        P_d,
        R_a,
        P_a,
        np.ones_like(risk),
        game_params.resource_budget,
    )

    matrix = build_transition_matrix(graph, node_list, coverage)
    schedule = simulate_patrol(
        matrix,
        node_list,
        start_idx=patrol_params.start_index,
        time_steps=patrol_params.time_steps,
        num_units=patrol_params.num_units,
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
    return df, summary
