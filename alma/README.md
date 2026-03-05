ALMA Package
============

Overview
--------
- Purpose: Compute patrol schedules for a Stackelberg Security Game (SSG) on a road network and expose compact, testable building blocks for research.
- Modules:
  - `config.py`: Typed parameter objects for game and patrol settings.
  - `data.py`: Graph I/O (adjacency JSON), helpers for map rendering.
  - `patrol.py`: Transition matrix + patrol simulation.
  - `ssg.py`: Stackelberg Security Game LP and solver glue.
  - `schedule.py`: High-level orchestration to generate a patrol schedule and summary.
  - `logging_utils.py`: Minimal logging setup used across modules.
  - `cli.py`: Simple CLI to export schedules.

Graph Format
------------
Adjacency-style JSON mapping `node_id -> { lat, lon, risk_factor, neighbors: [id,...] }`. Example snippet:

```json
{
  "A": { "lat": 40.11, "lon": -88.23, "risk_factor": 1.0, "neighbors": ["B"] },
  "B": { "lat": 40.12, "lon": -88.22, "risk_factor": 2.1, "neighbors": ["A", "C"] }
}
```

Programmatic Usage
------------------
```python
from alma.config import GameParams, PatrolParams
from alma.schedule import generate_patrol_schedule

game = GameParams(alpha=1, beta=1, gamma=1, delta=1, resource_budget=10)
patrol = PatrolParams(time_steps=120, num_units=5, start_index=0, random_seed=0)
df, summary = generate_patrol_schedule('data/uiuc_graph.json', game, patrol)
```

CLI
---
```bash
python -m alma.cli --graph data/uiuc_graph.json --output patrol.csv --time-steps 120 --num-units 5
```

Research Notes
--------------
- Utility design lives in `ssg.py` (`build_payoffs_from_risk`) — the natural hook for experimenting with different objectives, costs, or constraints.
- Movement policy is separated (`patrol.py`) so you can test alternate transition rules without touching the solver.
- `schedule.generate_patrol_schedule` accepts a `progress` callback but remains UI-agnostic to keep algorithms testable in isolation.
