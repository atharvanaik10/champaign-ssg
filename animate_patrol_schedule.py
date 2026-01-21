#!/usr/bin/env python3
"""
Animate patrol routes over the UIUC road graph.

Inputs
- A simple adjacency JSON produced by build_uiuc_graph (node_id -> {lat, lon, neighbors, risk_factor, crimes?}).
- A patrol schedule CSV with columns: unit_id, time_step, node_id.

Output
- An animated GIF (or other Matplotlib-supported animation format) showing unit positions over time.

Assumptions
- Files exist and columns/keys are as described.
- No fancy styling; keep it simple and readable.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def load_graph_from_adjacency(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Nodes: id -> (lon, lat)
    nodes = {}
    for nid, attrs in data.items():
        nodes[nid] = (float(attrs["lon"]), float(attrs["lat"]))

    # Edges: build undirected edges from neighbors; avoid duplicates
    edges = []
    for nid, attrs in data.items():
        x1, y1 = float(attrs["lon"]), float(attrs["lat"]) 
        for nbr in attrs.get("neighbors", []):
            # add edge once
            if nid < nbr and nbr in data:
                x2, y2 = float(data[nbr]["lon"]), float(data[nbr]["lat"]) 
                edges.append(((x1, y1), (x2, y2)))

    return nodes, edges


def load_patrol_schedule(csv_path):
    df = pd.read_csv(csv_path)
    # normalize/ensure expected columns
    df = df[["unit_id", "time_step", "node_id"]]
    # Sort by time for clean animation
    df = df.sort_values(["time_step", "unit_id"]).reset_index(drop=True)
    units = list(df["unit_id"].unique())
    timesteps = list(df["time_step"].unique())

    # Build mapping: time_step -> {unit_id: node_id}
    by_time = {}
    for t, sub in df.groupby("time_step"):
        by_time[t] = dict(zip(sub["unit_id"], sub["node_id"]))

    return units, timesteps, by_time


def animate_patrols(nodes, edges, units, timesteps, schedule_map, out_path="assets/patrol_animation.gif", fps=2):
    # Precompute unit colors and initial positions
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    unit_colors = {u: colors[i % len(colors)] for i, u in enumerate(units)}

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw edges
    for (x1, y1), (x2, y2) in edges:
        ax.plot([x1, x2], [y1, y2], color="#cccccc", linewidth=0.3, zorder=1)

    # Draw nodes
    xs = [xy[0] for xy in nodes.values()]
    ys = [xy[1] for xy in nodes.values()]
    ax.scatter(xs, ys, s=3, color="#666666", alpha=0.7, zorder=2)

    # Initialize a scatter for each unit
    scatters = {}
    for u in units:
        # set dummy coordinate initially; will update in init
        scat = ax.scatter([], [], s=50, color=unit_colors[u], label=str(u), zorder=3)
        scatters[u] = scat

    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    # Helper to get lon/lat for a node id
    def node_xy(node_id):
        return nodes[str(node_id)]
    
    # Build frame data: for each timestep, positions per unit
    frame_positions = []
    for t in timesteps:
        unit_pos = {}
        mapping = schedule_map.get(t, {})
        for u in units:
            nid = mapping[u]
            unit_pos[u] = node_xy(nid)
        frame_positions.append((t, unit_pos))

    def init():
        # place units for the first frame
        t0, pos0 = frame_positions[0]
        for u in units:
            x, y = pos0[u]
            scatters[u].set_offsets([[x, y]])
        ax.set_title(f"Time {t0}")
        return list(scatters.values())

    def update(frame_idx):
        t, pos = frame_positions[frame_idx]
        for u in units:
            x, y = pos[u]
            scatters[u].set_offsets([[x, y]])
        ax.set_title(f"Time {t}")
        return list(scatters.values())

    anim = FuncAnimation(fig, update, frames=len(frame_positions), init_func=init, blit=True, interval=1000 // max(1, fps))
    anim.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    return out_path


def main():
    # Set inputs/outputs here
    graph_json = "data/uiuc_graph.json"         # from build_uiuc_graph
    schedule_csv = "patrol_schedule.csv"   # unit_id,time_step,node_id
    output_gif = "assets/patrol_animation.gif"
    fps = 10

    nodes, edges = load_graph_from_adjacency(graph_json)
    units, timesteps, schedule_map = load_patrol_schedule(schedule_csv)
    out = animate_patrols(nodes, edges, units, timesteps, schedule_map, out_path=output_gif, fps=fps)
    print(f"Saved animation to: {out}")


if __name__ == "__main__":
    main()

