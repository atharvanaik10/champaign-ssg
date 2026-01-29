from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st

from alma.animation import animate_patrols
from alma.config import GameParams, PatrolParams
from alma.data import load_graph_for_animation
from alma.logging_utils import configure_logging
from alma.schedule import generate_patrol_schedule

configure_logging()
logger = logging.getLogger(__name__)


st.set_page_config(page_title="ALMA Patrol Planner", layout="wide")

st.title("ALMA Patrol Planner")
st.markdown(
    "Optimize patrol allocations using a Stackelberg Security Game and visualize the resulting route plan."
)


@st.cache_data(show_spinner=False)
def _load_graph_nodes_edges(graph_path: str):
    return load_graph_for_animation(graph_path)


def _build_schedule_map(df: pd.DataFrame) -> tuple[list[int], list[int], dict[int, dict[int, str]]]:
    units = sorted(df["unit_id"].unique().tolist())
    timesteps = sorted(df["time_step"].unique().tolist())
    by_time: dict[int, dict[int, str]] = {}
    for t, sub in df.groupby("time_step"):
        by_time[int(t)] = dict(zip(sub["unit_id"].astype(int), sub["node_id"]))
    return units, timesteps, by_time


@st.cache_data(show_spinner=False)
def _reverse_geocode(lat: float, lon: float, api_key: str) -> str | None:
    if not api_key:
        return None
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"latlng": f"{lat},{lon}", "key": api_key}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    results = data.get("results", [])
    if not results:
        return None
    return results[0].get("formatted_address")


def _build_route_url(route_points: list[tuple[float, float]]) -> str | None:
    if len(route_points) < 2:
        return None
    origin = f"{route_points[0][0]},{route_points[0][1]}"
    destination = f"{route_points[-1][0]},{route_points[-1][1]}"
    waypoints = "|".join(f"{lat},{lon}" for lat, lon in route_points[1:-1])
    params = [
        f"origin={quote_plus(origin)}",
        f"destination={quote_plus(destination)}",
        "travelmode=driving",
    ]
    if waypoints:
        params.append(f"waypoints={quote_plus(waypoints)}")
    return f"https://www.google.com/maps/dir/?api=1&{'&'.join(params)}"


with st.sidebar:
    st.header("Inputs")
    graph_path = st.text_input("Graph JSON", value="data/uiuc_graph.json")

    st.subheader("Game parameters")
    alpha = st.number_input("Defender reward (alpha)", value=1.0, step=0.1)
    beta = st.number_input("Defender loss (beta)", value=1.0, step=0.1)
    gamma = st.number_input("Attacker reward (gamma)", value=1.0, step=0.1)
    delta = st.number_input("Attacker loss (delta)", value=1.0, step=0.1)
    budget = st.number_input("Resource budget (K)", value=10.0, step=1.0, min_value=0.0)

    st.subheader("Patrol parameters")
    time_steps = st.slider("Time steps", min_value=60, max_value=1000, value=480, step=10)
    num_units = st.slider("Patrol units", min_value=1, max_value=10, value=5, step=1)
    start_index = st.number_input("Start node index", value=0, step=1, min_value=0)
    seed = st.number_input("Random seed", value=0, step=1, min_value=0)

    st.subheader("Outputs")
    generate_gif = st.checkbox("Render animated GIF", value=True)
    gif_fps = st.slider("GIF FPS", min_value=1, max_value=15, value=8)

    st.subheader("Google Maps")
    maps_api_key = st.text_input("Google Maps API key", type="password")

    run_button = st.button("Generate patrol plan")


if run_button:
    graph_file = Path(graph_path)
    if not graph_file.exists():
        st.error(f"Graph JSON not found at {graph_file}.")
    else:
        with st.spinner("Solving Stackelberg game and simulating patrols..."):
            game = GameParams(
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
                resource_budget=budget,
            )
            patrol = PatrolParams(
                time_steps=time_steps,
                num_units=num_units,
                start_index=start_index,
                random_seed=seed,
            )
            schedule_df, summary = generate_patrol_schedule(graph_path, game, patrol)

        st.success("Patrol plan ready.")

        metric_cols = st.columns(3)
        metric_cols[0].metric("Best defender utility", f"{summary['best_defender_utility']:.3f}")
        metric_cols[1].metric("Nodes", f"{int(summary['nodes'])}")
        metric_cols[2].metric("Edges", f"{int(summary['edges'])}")

        st.subheader("Patrol schedule")
        nodes, edges = _load_graph_nodes_edges(graph_path)
        units, timesteps, schedule_map = _build_schedule_map(schedule_df)
        selected_unit = st.selectbox("Select unit", options=units, index=0)

        unit_schedule = schedule_df[schedule_df["unit_id"] == selected_unit].copy()
        unit_schedule["node_id"] = unit_schedule["node_id"].astype(str)
        unit_schedule["latitude"] = unit_schedule["node_id"].map(lambda nid: nodes[nid][1])
        unit_schedule["longitude"] = unit_schedule["node_id"].map(lambda nid: nodes[nid][0])

        maps_api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
        if maps_api_key:
            addresses = {}
            for node_id in unit_schedule["node_id"].unique():
                lat = nodes[node_id][1]
                lon = nodes[node_id][0]
                try:
                    address = _reverse_geocode(lat, lon, maps_api_key)
                except requests.RequestException:
                    address = None
                addresses[node_id] = address or f"{lat:.6f}, {lon:.6f}"
            unit_schedule["address"] = unit_schedule["node_id"].map(addresses)
        else:
            unit_schedule["address"] = unit_schedule.apply(
                lambda row: f"{row['latitude']:.6f}, {row['longitude']:.6f}", axis=1
            )

        st.dataframe(
            unit_schedule[["time_step", "unit_id", "node_id", "address"]],
            use_container_width=True,
            height=380,
        )

        csv_bytes = unit_schedule.to_csv(index=False).encode("utf-8")
        route_points = [
            (row["latitude"], row["longitude"])
            for _, row in unit_schedule.sort_values("time_step").iterrows()
        ]
        route_url = _build_route_url(route_points)
        button_cols = st.columns([1, 1])
        with button_cols[0]:
            st.download_button(
                "Download schedule CSV",
                data=csv_bytes,
                file_name=f"patrol_schedule_unit_{selected_unit}.csv",
            )
        with button_cols[1]:
            if route_url:
                st.link_button("Open route in Google Maps", route_url)
            else:
                st.button("Open route in Google Maps", disabled=True)

        if generate_gif:
            st.subheader("Animated route")
            output_dir = Path("cache")
            output_dir.mkdir(parents=True, exist_ok=True)
            gif_path = output_dir / f"patrol_animation_unit_{selected_unit}.gif"
            with st.spinner("Rendering GIF..."):
                animate_patrols(
                    nodes,
                    edges,
                    [selected_unit],
                    timesteps,
                    schedule_map,
                    out_path=gif_path,
                    fps=gif_fps,
                )
            st.image(str(gif_path), caption="Optimized patrol route animation")
