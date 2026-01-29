from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameParams:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 1.0
    resource_budget: float = 10.0


@dataclass(frozen=True)
class PatrolParams:
    time_steps: int = 480
    num_units: int = 5
    start_index: int = 0
    random_seed: int = 0


@dataclass(frozen=True)
class AnimationParams:
    fps: int = 8
    output_path: str = "assets/patrol_animation.gif"
