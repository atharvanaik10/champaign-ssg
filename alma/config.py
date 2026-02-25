from __future__ import annotations

from dataclasses import dataclass

"""Typed configuration objects for ALMA.

These dataclasses group parameters for the Stackelberg game and the patrol
simulation. They are immutable and safe to pass across layers (UI, API,
orchestration) without additional validation logic.
"""


@dataclass(frozen=True)
class GameParams:
    """Parameters for the Stackelberg Security Game (SSG).

    Attributes:
        alpha: Defender reward when an attacked node is covered.
        beta: Defender loss when an attacked node is uncovered.
        gamma: Attacker reward when an attacked node is uncovered.
        delta: Attacker loss when an attacked node is covered.
        resource_budget: Upper bound on the weighted sum of coverage (K).
    """
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 1.0
    resource_budget: float = 10.0


@dataclass(frozen=True)
class PatrolParams:
    """Parameters for the patrol simulation.

    Attributes:
        time_steps: Number of timesteps (T) to simulate, inclusive (0..T).
        num_units: Number of patrol units moving simultaneously.
        start_index: If an int, the shared start node index. When multiple
            units are used, ALMA will typically replace this with diverse
            starting positions; see schedule.generate_patrol_schedule.
        random_seed: Seed for reproducible random choices.
    """
    time_steps: int = 480
    num_units: int = 5
    start_index: int = 0
    random_seed: int = 0


@dataclass(frozen=True)
class AnimationParams:
    """(Legacy) Parameters for GIF rendering utilities.

    Attributes:
        fps: Frames per second.
        output_path: Path to the animated GIF output.
    """
    fps: int = 8
    output_path: str = "assets/patrol_animation.gif"
