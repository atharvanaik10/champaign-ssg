from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
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
        st.dataframe(schedule_df, use_container_width=True, height=380)

        csv_bytes = schedule_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download schedule CSV", data=csv_bytes, file_name="patrol_schedule.csv")

        if generate_gif:
            st.subheader("Animated route")
            nodes, edges = _load_graph_nodes_edges(graph_path)
            units, timesteps, schedule_map = _build_schedule_map(schedule_df)
            output_dir = Path("cache")
            output_dir.mkdir(parents=True, exist_ok=True)
            gif_path = output_dir / "patrol_animation.gif"
            with st.spinner("Rendering GIF..."):
                animate_patrols(
                    nodes,
                    edges,
                    units,
                    timesteps,
                    schedule_map,
                    out_path=gif_path,
                    fps=gif_fps,
                )
            st.image(str(gif_path), caption="Optimized patrol route animation")
