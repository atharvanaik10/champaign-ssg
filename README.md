# ALMA: Active Law‑enforcement Mixed‑strategy Allocator

Risk‑aware patrol planning for the UIUC/Champaign area using Stackelberg Security Games (SSG).

This repository is intentionally simple: a small, well‑documented Python library (`alma/`), a tiny FastAPI server (`server.py`), and a minimal Svelte+Tailwind UI in `web/`.

## Project Layout

- `alma/`: Core compute library (graph I/O, SSG solve, patrol simulation).
- `server.py`: Minimal FastAPI with two endpoints: `/plan` and `/graph`. Also serves static files from `web/dist` after a build.
- Caching: Repeated runs with the same inputs are loaded from `cache/` automatically (see below).
- `web/`: Minimal Svelte + Tailwind app (single page) that calls the API and renders a MapLibre map and schedule.
- `data/`: Sample graph JSON (`uiuc_graph.json`).

Removed legacy scaffolding (complex backend, SvelteKit app, Streamlit scripts) to keep the POC easy to read and extend.

## Quick Start

### Python setup

```bash
python -m venv .venv && source .venv/bin/activate  # optional
pip install -r requirements.txt
```

### Start the API

```bash
uvicorn server:app --reload
```

API will run on `http://localhost:8000`.

### Web UI (optional)

```bash
cd web
npm install
npm run build           # outputs to web/dist
cd ..
uvicorn server:app --reload   # serves web/dist at /
```

Open `http://localhost:8000` and click Start to generate a schedule.

## API

- `POST /plan?format=json|csv` — Run a plan (synchronous) and return the schedule.
  - Body: `{ graph_path, game: {alpha,beta,gamma,delta,resource_budget}, patrol: {time_steps, num_units, start_index, random_seed} }`
  - Returns JSON `{ summary, schedule }` or CSV when `format=csv`.

- `GET /graph?graph_path=...` — Graph as GeoJSON FeatureCollection for the map.

## Library (`alma/`)

Key modules:
- `config.py`: Typed parameter objects (`GameParams`, `PatrolParams`).
- `data.py`: Graph I/O and helpers for animation/visualization.
- `ssg.py`: Stackelberg Security Game solver (CVXPY/ECOS/OSQP).
- `patrol.py`: Transition matrix and patrol simulation.
- `schedule.py`: High‑level orchestration that wires everything.
- `cli.py`: Simple CLI to export schedules as CSV.

Example usage:

```python
from alma.config import GameParams, PatrolParams
from alma.schedule import generate_patrol_schedule

game = GameParams(alpha=1, beta=1, gamma=1, delta=1, resource_budget=10)
patrol = PatrolParams(time_steps=120, num_units=5, start_index=0, random_seed=0)
df, summary = generate_patrol_schedule('data/uiuc_graph.json', game, patrol)
print(df.head(), summary)
```

CLI:

```bash
python -m alma.cli --graph data/uiuc_graph.json --output patrol_schedule.csv --time-steps 120 --num-units 5
```

## Notes

- The UI is intentionally lean: one page, simple form, MapLibre for context, and a compact table.
- If you’re iterating on research (utility functions, constraints, budgets), concentrate changes inside `alma/`.
- The API remains synchronous for simplicity; swap in a background task if you need long runs.

### Caching behavior

- The solver/simulation is cached on disk under `cache/` keyed by the graph file content and the parameter values.
- Cache is automatic via `generate_patrol_schedule_cached(...)`. To clear cache, delete files in `cache/`.
