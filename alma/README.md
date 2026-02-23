ALMA Package
============

Overview
--------
- Purpose: Compute and visualize patrol schedules for a Stackelberg Security Game (SSG) on a road network.
- Modules:
  - `config.py`: Typed parameter objects for game and patrol settings.
  - `data.py`: Graph I/O, schedule loading, utilities for animation prep.
  - `patrol.py`: Patrol simulation helpers.
  - `ssg.py`: Stackelberg Security Game LP and solver glue.
  - `schedule.py`: High-level orchestration to generate a patrol schedule and summary.
  - `animation.py`: Matplotlib-based GIF export helpers (legacy/offline).
  - `visualization.py`: Plotly figure builders for interactive maps.
  - `logging_utils.py`: Minimal logging setup used across modules.

Key Concepts
------------
- Graph: The campus/street network with `lat/lon`, neighbors, and risk metadata loaded from JSON.
- SSG: An LP that allocates coverage per node under a resource budget K.
- Patrol: Simulated movement of `num_units` over `time_steps`, starting from a given node index.

Usage
-----
The repository includes a Streamlit app (`app.py`) that wraps the package.

Basic programmatic usage:

```python
from alma.config import GameParams, PatrolParams
from alma.schedule import generate_patrol_schedule
from alma.data import load_graph_for_animation
from alma.visualization import make_unit_paths, build_route_figure

game = GameParams(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, resource_budget=10.0)
patrol = PatrolParams(time_steps=480, num_units=5, start_index=0, random_seed=0)

schedule_df, summary = generate_patrol_schedule("data/uiuc_graph.json", game, patrol)
nodes, edges = load_graph_for_animation("data/uiuc_graph.json")

# Build interactive Plotly figure for a single timestep
units = sorted(schedule_df["unit_id"].unique().tolist())
timesteps = sorted(schedule_df["time_step"].unique().tolist())
schedule_map = {
    int(t): dict(zip(sub["unit_id"].astype(int), sub["node_id"]))
    for t, sub in schedule_df.groupby("time_step")
}
unit_paths = make_unit_paths(schedule_df, nodes)

fig = build_route_figure(
    nodes=nodes,
    edges=edges,
    unit_paths=unit_paths,
    schedule_map=schedule_map,
    units=units,
    current_t=int(timesteps[0]),
    show_trails=True,
)
fig.show()
```

Development Notes
-----------------
- The Plotly map logic is intentionally factored out into `visualization.py` to keep the app minimal and improve reuse.
- All public helpers include docstrings and type hints to ease integration and static analysis.
- The app uses `st.session_state` to persist results across reruns; this avoids blank UI when interacting with widgets like sliders.

