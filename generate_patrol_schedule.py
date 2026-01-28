#!/usr/bin/env python3
"""Generate a patrol schedule CSV from a graph and game parameters."""

from __future__ import annotations

import argparse
from pathlib import Path

from alma.config import GameParams, PatrolParams
from alma.logging_utils import configure_logging
from alma.schedule import generate_patrol_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a patrol schedule CSV.")
    parser.add_argument("--graph", default="data/uiuc_graph.json", help="Path to graph JSON adjacency list.")
    parser.add_argument("--output", default="patrol_schedule.csv", help="Output CSV path.")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--budget", type=float, default=10.0, help="Resource budget (K).")
    parser.add_argument("--time-steps", type=int, default=480)
    parser.add_argument("--num-units", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    game = GameParams(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        resource_budget=args.budget,
    )
    patrol = PatrolParams(
        time_steps=args.time_steps,
        num_units=args.num_units,
        start_index=args.start_index,
        random_seed=args.seed,
    )

    df, _summary = generate_patrol_schedule(args.graph, game, patrol)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved patrol schedule to {output_path}")


if __name__ == "__main__":
    main()
