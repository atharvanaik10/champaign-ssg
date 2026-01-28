#!/usr/bin/env python3
"""Animate patrol routes over the road graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from alma.animation import animate_patrols
from alma.config import AnimationParams
from alma.data import load_graph_for_animation, load_patrol_schedule
from alma.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a patrol schedule animation.")
    parser.add_argument("--graph", default="data/uiuc_graph.json", help="Path to graph JSON adjacency list.")
    parser.add_argument("--schedule", default="patrol_schedule.csv", help="Patrol schedule CSV.")
    parser.add_argument("--output", default="assets/patrol_animation.gif", help="Output GIF path.")
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    nodes, edges = load_graph_for_animation(args.graph)
    units, timesteps, schedule_map = load_patrol_schedule(args.schedule)

    animation = AnimationParams(fps=args.fps, output_path=args.output)
    Path(animation.output_path).parent.mkdir(parents=True, exist_ok=True)
    animate_patrols(
        nodes,
        edges,
        units,
        timesteps,
        schedule_map,
        out_path=animation.output_path,
        fps=animation.fps,
    )
    print(f"Saved animation to: {animation.output_path}")


if __name__ == "__main__":
    main()
