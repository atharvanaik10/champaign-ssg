# ALMA: Active Law-enforcement Mixed-strategy Allocator

Risk-aware patrol planning for the UIUC/Champaign area using Stackelberg Security Games (SSG).

## Architecture

ALMA now uses a decoupled architecture:

- `alma/`: core compute library (graph loading, SSG solve, patrol simulation).
- `backend/`: FastAPI service for job orchestration, status/progress APIs, graph/schedule endpoints.
- `frontend/`: SvelteKit + Tailwind app for interactive controls, progress, map playback, and schedule export.

## Backend (FastAPI)

### Install

```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### Run

```bash
uvicorn backend.main:app --reload
# or
scripts/dev.sh
```

Backend runs on `http://localhost:8000`.

### API endpoints

- `POST /jobs`: Start optimization job.
- `GET /jobs/{job_id}`: Job status, progress, timings, summary.
- `GET /jobs/{job_id}/schedule?format=json|csv`: Schedule output.
- `GET /jobs/{job_id}/events`: SSE progress stream.
- `GET /graph?graph_path=...`: Graph as GeoJSON FeatureCollection.

Progress phases are surfaced from `alma.schedule.generate_patrol_schedule(..., progress=callback)` and stored in the in-process thread-safe job registry.

### Tests

```bash
pytest backend/tests
```

## Frontend (SvelteKit + Tailwind)

### Install

```bash
cd frontend
npm install
```

### Run

```bash
npm run dev
```

Frontend runs on `http://localhost:5173` and calls backend on `http://localhost:8000` (CORS enabled for local dev).

## Frontend features

- Input form for graph and optimization parameters.
- Job creation and progress via SSE (polling fallback).
- MapLibre map rendering roads (LineString) + patrol units (Point layer).
- Playback controls: play/pause, seek slider, FPS control, keyboard shortcuts (`space`, `←`, `→`).
- Schedule table + CSV download.

## Legacy Streamlit UI

Existing Streamlit scripts remain in the repository for backwards compatibility. The new production path is backend + frontend.
