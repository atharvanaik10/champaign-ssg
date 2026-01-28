from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

logger = logging.getLogger(__name__)


def animate_patrols(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    units: List[int],
    timesteps: List[int],
    schedule_map: dict[int, dict[int, str]],
    out_path: str | Path,
    fps: int = 2,
) -> str:
    out_path = str(out_path)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    unit_colors = {unit: colors[i % len(colors)] for i, unit in enumerate(units)}

    fig, ax = plt.subplots(figsize=(8, 8))

    for (x1, y1), (x2, y2) in edges:
        ax.plot([x1, x2], [y1, y2], color="#cccccc", linewidth=0.3, zorder=1)

    xs = [xy[0] for xy in nodes.values()]
    ys = [xy[1] for xy in nodes.values()]
    ax.scatter(xs, ys, s=3, color="#666666", alpha=0.7, zorder=2)

    scatters = {}
    for unit in units:
        scat = ax.scatter([], [], s=50, color=unit_colors[unit], label=str(unit), zorder=3)
        scatters[unit] = scat

    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    def node_xy(node_id: str):
        return nodes[str(node_id)]

    frame_positions = []
    for timestep in timesteps:
        unit_pos = {}
        mapping = schedule_map.get(timestep, {})
        for unit in units:
            node_id = mapping[unit]
            unit_pos[unit] = node_xy(node_id)
        frame_positions.append((timestep, unit_pos))

    def init():
        t0, pos0 = frame_positions[0]
        for unit in units:
            x, y = pos0[unit]
            scatters[unit].set_offsets([[x, y]])
        ax.set_title(f"Time {t0}")
        return list(scatters.values())

    def update(frame_idx: int):
        t, pos = frame_positions[frame_idx]
        for unit in units:
            x, y = pos[unit]
            scatters[unit].set_offsets([[x, y]])
        ax.set_title(f"Time {t}")
        return list(scatters.values())

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_positions),
        init_func=init,
        blit=True,
        interval=1000 // max(1, fps),
    )
    anim.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    logger.info("Saved animation to: %s", out_path)
    return out_path
