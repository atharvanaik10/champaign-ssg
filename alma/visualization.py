from __future__ import annotations

"""
Visualization utilities for ALMA.

This module provides small, composable helpers for building Plotly
figures and precomputing route geometry. Keeping this logic here makes
it easier to reuse in notebooks or downstream apps and simplifies the
Streamlit app code.
"""

from typing import Dict, Iterable, List, Mapping, Tuple

import pandas as pd
import plotly.graph_objects as go


LonLat = Tuple[float, float]
NodeCoords = Mapping[str, LonLat]
Edges = Iterable[Tuple[LonLat, LonLat]]


def make_unit_paths(schedule_df: pd.DataFrame, nodes: NodeCoords) -> Dict[int, pd.DataFrame]:
    """Precompute per-unit lon/lat paths over time.

    Parameters
    - schedule_df: DataFrame with columns [unit_id, time_step, node_id]
    - nodes: Mapping of node_id -> (lon, lat)

    Returns
    - Dict mapping unit_id -> DataFrame[time_step, lon, lat]
    """
    required_cols = {"unit_id", "time_step", "node_id"}
    missing = required_cols - set(schedule_df.columns)
    if missing:
        raise ValueError(f"schedule_df missing columns: {sorted(missing)}")

    unit_paths: Dict[int, pd.DataFrame] = {}
    for unit, sub in (
        schedule_df.sort_values(["unit_id", "time_step"]).groupby("unit_id")
    ):
        df = sub.assign(
            lon=lambda d: d["node_id"].map(lambda nid: nodes[str(nid)][0]),
            lat=lambda d: d["node_id"].map(lambda nid: nodes[str(nid)][1]),
        )[["time_step", "lon", "lat"]].reset_index(drop=True)
        unit_paths[int(unit)] = df
    return unit_paths


def build_route_figure(
    nodes: NodeCoords,
    edges: Edges,
    unit_paths: Mapping[int, pd.DataFrame],
    schedule_map: Mapping[int, Mapping[int, str]],
    units: Iterable[int],
    current_t: int,
    show_trails: bool = True,
) -> go.Figure:
    """Construct an interactive route map for the current timestep.

    Parameters
    - nodes: node_id -> (lon, lat)
    - edges: iterable of ((lon1, lat1), (lon2, lat2)) tuples
    - unit_paths: unit -> DataFrame[time_step, lon, lat]
    - schedule_map: time_step -> {unit_id -> node_id}
    - units: iterable of unit ids to include
    - current_t: timestep to render
    - show_trails: whether to draw the path up to `current_t`

    Returns
    - Plotly Figure with roads, trails (optional), and unit markers
    """
    # Prepare a light network backdrop
    edge_x: List[float | None] = []
    edge_y: List[float | None] = []
    for (x1, y1), (x2, y2) in edges:
        edge_x += [x1, x2, None]
        edge_y += [y1, y2, None]

    # Colors per unit (10-color palette cycling)
    palette_hex = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    unit_color = {u: palette_hex[i % len(palette_hex)] for i, u in enumerate(units)}

    fig = go.Figure()

    # Roads backdrop
    fig.add_trace(
        go.Scattergl(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="rgba(160,160,160,0.5)", width=1),
            hoverinfo="skip",
            name="roads",
        )
    )

    # Trails up to current_t
    if show_trails:
        for u in units:
            p = unit_paths.get(u)
            if p is None:
                continue
            p = p[p["time_step"] <= int(current_t)]
            if not p.empty:
                fig.add_trace(
                    go.Scattergl(
                        x=p["lon"],
                        y=p["lat"],
                        mode="lines",
                        line=dict(color=unit_color[u], width=2),
                        name=f"Unit {u} trail",
                        hoverinfo="skip",
                    )
                )

    # Current unit positions
    mapping = schedule_map.get(int(current_t), {})
    pos_x: List[float] = []
    pos_y: List[float] = []
    labels: List[str] = []
    colors: List[str] = []
    for u in units:
        nid = mapping.get(u)
        if nid is None:
            continue
        lon, lat = nodes[str(nid)]
        pos_x.append(lon)
        pos_y.append(lat)
        labels.append(f"Unit {u}<br>{lon:.5f}, {lat:.5f}")
        colors.append(unit_color[u])
    if pos_x:
        fig.add_trace(
            go.Scattergl(
                x=pos_x,
                y=pos_y,
                mode="markers",
                marker=dict(size=8, color=colors),
                text=labels,
                hoverinfo="text",
                name="units",
            )
        )

    # Consistent layout and aspect ratio
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"Time {int(current_t)}",
        xaxis=dict(title="Longitude", showgrid=False, zeroline=False),
        yaxis=dict(
            title="Latitude",
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig

