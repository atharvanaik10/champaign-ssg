# ALMA: Active Law-enforcement Mixed-strategy Allocator

Risk-aware patrol planning for the UIUC/Champaign area using Stackelberg Security Games.

## Overview

This project builds a campus-scale road graph from OpenStreetMap, maps historical crime reports onto that graph to estimate per-node risk, solves a Stackelberg Security Game (SSG) to allocate patrol attention, and then generates and animates concrete patrol routes derived from the optimal coverage distribution.

## What the application does

ALMA helps public safety teams experiment with patrol strategies by:

- Modeling the road network as a graph with per-node risk.
- Optimizing patrol coverage with a Stackelberg Security Game.
- Converting coverage probabilities into concrete patrol routes.
- Visualizing the result as both a schedule table and an animated map.

## Features

- Data ingestion: Parses the UIUC Clery Crime Log (Excel), geocodes locations (Google Maps API), and assigns a severity score (1–5) to each incident using a lightweight classifier (gpt-5-nano) with basic caching.
- Graph building: Downloads and simplifies an OSM road network for the campus area (OSMnx), consolidates intersections, and exports a JSON adjacency list with per-node lat/lon and risk.
- Risk modeling: Attaches incidents to the nearest road node and computes a `risk_factor` per node that scales with incident severity and frequency.
- Game-theoretic allocation: Formulates and solves a single-defender SSG with a resource budget (CVXPy) to produce optimal coverage probabilities over nodes.
- Patrol synthesis: Converts coverage into a biased Markov policy on the graph and simulates multi-unit random-walk patrols; exports a patrol schedule CSV.
- Visualization: Animated GIF showing patrol units moving over time on the network.
- Lightweight graph library: A minimal undirected, weighted graph class for prototyping risk-aware structures and serialization.

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the UI

The Streamlit UI allows you to tune patrol parameters and immediately visualize the resulting schedule and animation.

```bash
streamlit run app.py
```

### Generate a schedule (CLI)

```bash
python generate_patrol_schedule.py \
  --graph data/uiuc_graph.json \
  --output patrol_schedule.csv \
  --budget 10 \
  --num-units 5 \
  --time-steps 480
```

### Render the animation (CLI)

```bash
python animate_patrol_schedule.py \
  --graph data/uiuc_graph.json \
  --schedule patrol_schedule.csv \
  --output assets/patrol_animation.gif \
  --fps 10
```

## Notes

- External services: Uses Google Maps Geocoding API and the OpenAI API for severity classification (both optional with caching/heuristics).
- Scope: Bounding box and parameters are tuned for UIUC/Champaign but are configurable for other regions or data sources.
