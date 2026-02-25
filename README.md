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

## Setup (Data Prep)

The demo expects a road graph with risk attached from a processed crime log. Run the one‑time setup CLI to build these artifacts.

### Requirements

- Python packages (install on your env):
  - `pip install osmnx openai python-dotenv`
- API keys (available as environment variables or in a `.env` file):
  - `GOOGLE_MAPS_API_KEY`: for geocoding the crime log locations
  - `OPENAI_API_KEY`: for classifying crime severities (1–5)

### Commands

- Process the raw crime log (writes `data/crime_log_processed_location.csv` and `data/crime_log_processed.csv`):

  ```bash
  python -m alma.setup process-crime \
    --input-xlsx "data/Clery Crime Log - Police Contacts Only - 2021-October 31 2025.xlsx" \
    --out-base data/crime_log_processed
  ```

- Build the road graph and risk (writes `data/uiuc_graph.json` and preview images in `assets/`):

  ```bash
  python -m alma.setup build-graph \
    --west -88.24442 --south 40.09396 --east -88.21858 --north 40.11668 \
    --crimes-csv data/crime_log_processed.csv \
    --out-adjacency data/uiuc_graph.json \
    --out-image-osmnx assets/uiuc_graph.png \
    --out-image-simple assets/uiuc_graph_simple.png \
    --out-image-risk assets/uiuc_graph_risk.png \
    --tolerance-m 15
  ```

- Run both steps in sequence:

  ```bash
  python -m alma.setup all \
    --input-xlsx "data/Clery Crime Log - Police Contacts Only - 2021-October 31 2025.xlsx" \
    --out-base data/crime_log_processed \
    --west -88.24442 --south 40.09396 --east -88.21858 --north 40.11668 \
    --out-adjacency data/uiuc_graph.json \
    --out-image-osmnx assets/uiuc_graph.png \
    --out-image-simple assets/uiuc_graph_simple.png \
    --out-image-risk assets/uiuc_graph_risk.png \
    --tolerance-m 15
  ```

The CLI is explicit and does not run automatically; use it whenever you need to regenerate inputs. It fails fast if dependencies or API keys are missing so you can fix configuration early.

### Caching behavior

- The solver/simulation is cached on disk under `cache/` keyed by the graph file content and the parameter values.
- Cache is automatic via `generate_patrol_schedule_cached(...)`. To clear cache, delete files in `cache/`.
