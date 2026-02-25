from __future__ import annotations

"""ALMA setup CLI: build the road graph and process the crime log.

This module merges the legacy one-off scripts into a single command-line tool
that you run explicitly (it never runs on import). It prepares the required
data artifacts for the UIUC/Champaign demo: a cleaned/collapsed road graph with
per-node risk derived from the processed crime log.

Dependencies (required)
-----------------------
- OSMnx: downloads and simplifies the road network.
- Google Maps Geocoding API: converts crime addresses to lat/lon
  (requires `GOOGLE_MAPS_API_KEY`).
- OpenAI API: classifies crime descriptions into severities 1–5
  (requires `OPENAI_API_KEY`).

Subcommands
-----------
- build-graph: Download/construct the road graph via OSMnx, attach crime risk
  from a processed CSV, and export adjacency JSON + preview images.
- process-crime: Process the raw crime log into geocoded/severity-scored CSVs.
- all: Run both steps (process-crime then build-graph).

This tool fails fast if required dependencies or API keys are missing.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable

import networkx as nx
import osmnx as ox
import numpy as np
import pandas as pd
from openai import OpenAI


# ---------------------------- Graph construction ----------------------------

DAMPENING_FACTOR = 0.5


def to_simple_graph(G_multi) -> nx.Graph:
    """Convert an OSMnx Multi(Di)Graph to a simple undirected Graph.

    - Collapses parallel edges by keeping the shortest 'length'.
    - Adds node attributes: lat from `y`, lon from `x`, default risk_factor=1.0.
    """
    G_undirected = ox.convert.to_undirected(G_multi)

    G = nx.Graph()
    try:
        G.graph.update(G_undirected.graph)
    except Exception:
        pass
    if "crs" not in G.graph:
        G.graph["crs"] = "epsg:4326"

    for n, data in G_undirected.nodes(data=True):
        lat = data.get("y")
        lon = data.get("x")
        G.add_node(n, lat=lat, lon=lon, x=lon, y=lat, risk_factor=1.0)

    for u, v, edata in G_undirected.edges(data=True):
        w = float(edata.get("length", 1.0))
        geom = edata.get("geometry")
        if G.has_edge(u, v):
            if w < G[u][v].get("length", w):
                G[u][v]["length"] = w
                if geom is not None:
                    G[u][v]["geometry"] = geom
        else:
            attrs = {"length": w}
            if geom is not None:
                attrs["geometry"] = geom
            G.add_edge(u, v, **attrs)

    return G


def plot_graph_osmnx(G, out_path: str, dpi: int = 200) -> str:
    fig, ax = ox.plot_graph(
        G,
        node_size=10,
        node_color="#444444",
        edge_color="#222222",
        edge_linewidth=0.4,
        bgcolor="white",
        show=False,
        close=False,
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt  # local import

    plt.close(fig)
    return out_path


def plot_simple_graph(G: nx.Graph, out_path: str, dpi: int = 200) -> str:
    import matplotlib.pyplot as plt  # local import

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    for u, v in G.edges():
        x = [G.nodes[u]["lon"], G.nodes[v]["lon"]]
        y = [G.nodes[u]["lat"], G.nodes[v]["lat"]]
        ax.plot(x, y, color="#222222", linewidth=0.3, alpha=0.9)
    xs = [data["lon"] for _, data in G.nodes(data=True)]
    ys = [data["lat"] for _, data in G.nodes(data=True)]
    ax.scatter(xs, ys, s=5, color="#444444", zorder=3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_risk_graph(G: nx.Graph, out_path: str) -> str:
    import matplotlib.pyplot as plt  # local import

    fig, ax = plt.subplots(figsize=(8, 8))
    for u, v in G.edges():
        x = [G.nodes[u]["lon"], G.nodes[v]["lon"]]
        y = [G.nodes[u]["lat"], G.nodes[v]["lat"]]
        ax.plot(x, y, color="#000000", linewidth=1.1, zorder=1)
    xs, ys, sizes = [], [], []
    for _, data in G.nodes(data=True):
        xs.append(data["lon"])
        ys.append(data["lat"])
        rf = float(data.get("risk_factor", 1.0))
        sizes.append(20 * rf)
    ax.scatter(xs, ys, s=sizes, c="#d62728", alpha=0.6, edgecolors="none", zorder=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out_path


def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def attach_crimes_to_graph(
    G: nx.Graph,
    csv_path: str | Path,
    id_col: str = "Number",
    lat_col: str = "lat",
    lon_col: str = "lon",
    sev_col: str = "severity",
) -> None:
    """Attach crime IDs and risk contribution to the nearest graph node.

    Updates per-node `crimes` and `risk_factor` attributes in place.
    """
    df = pd.read_csv(csv_path)

    nodes = list(G.nodes())
    node_lats = np.array([G.nodes[n]["lat"] for n in nodes], dtype=float)
    node_lons = np.array([G.nodes[n]["lon"] for n in nodes], dtype=float)

    for n in nodes:
        G.nodes[n]["crimes"] = []
        G.nodes[n]["_sev_sum"] = 0.0
        G.nodes[n]["_sev_count"] = 0

    for _, row in df.iterrows():
        clat = float(row[lat_col])
        clon = float(row[lon_col])
        cnum = str(row[id_col])
        csev = float(row[sev_col])
        dists = haversine_dist(clat, clon, node_lats, node_lons)
        idx = int(np.argmin(dists))
        nid = nodes[idx]
        G.nodes[nid]["crimes"].append(cnum)
        G.nodes[nid]["_sev_sum"] += csev
        G.nodes[nid]["_sev_count"] += 1

    for n in nodes:
        cnt = G.nodes[n]["_sev_count"]
        if cnt > 0:
            avg = G.nodes[n]["_sev_sum"] / cnt
            G.nodes[n]["risk_factor"] = float(avg * ((1.0 + cnt) ** DAMPENING_FACTOR))
        G.nodes[n].pop("_sev_sum", None)
        G.nodes[n].pop("_sev_count", None)


def save_adjacency_json(G: nx.Graph, out_path: str) -> str:
    """Save a simple JSON adjacency list with node attributes."""
    adj: dict[str, dict] = {}
    for n, data in G.nodes(data=True):
        entry = {
            "lat": data.get("lat"),
            "lon": data.get("lon"),
            "risk_factor": data.get("risk_factor", 1.0),
            "neighbors": [str(v) for v in G.neighbors(n)],
        }
        if "crimes" in data:
            entry["crimes"] = [str(x) for x in data["crimes"]]
        adj[str(n)] = entry
    Path(out_path).write_text(json.dumps(adj, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def build_graph(
    west: float,
    south: float,
    east: float,
    north: float,
    crimes_csv: str | Path,
    out_adjacency: str | Path = "data/uiuc_graph.json",
    out_image_osmnx: str | Path = "assets/uiuc_graph.png",
    out_image_simple: str | Path = "assets/uiuc_graph_simple.png",
    out_image_risk: str | Path = "assets/uiuc_graph_risk.png",
    consolidate_tolerance_m: float = 15.0,
) -> None:
    """Build UIUC road graph with risk from crimes and export artifacts.

    Requires OSMnx. Downloads, simplifies, attaches crime risk, and exports
    adjacency JSON plus preview images.
    """
    ox = _require_osmnx()
    # 1) Download raw graph for the bounding box [W,S,E,N]
    G_raw = ox.graph_from_bbox([west, south, east, north], network_type="drive", simplify=True)
    # Consolidate close intersections in a projected CRS
    G_temp = ox.project_graph(G_raw)
    G_consolidated = ox.consolidate_intersections(G_temp, tolerance=consolidate_tolerance_m, rebuild_graph=True)
    G_consolidated = ox.project_graph(G_consolidated, to_crs="EPSG:4326")

    # 2) Convert to a simple undirected graph with lat/lon/risk_factor
    G = to_simple_graph(G_consolidated)

    # 3) Export images
    Path(out_image_osmnx).parent.mkdir(parents=True, exist_ok=True)
    Path(out_image_simple).parent.mkdir(parents=True, exist_ok=True)
    Path(out_image_risk).parent.mkdir(parents=True, exist_ok=True)
    plot_graph_osmnx(G_consolidated, str(out_image_osmnx))
    plot_simple_graph(G, str(out_image_simple))

    # 4) Attach crimes, export adjacency
    attach_crimes_to_graph(G, str(crimes_csv))
    Path(out_adjacency).parent.mkdir(parents=True, exist_ok=True)
    save_adjacency_json(G, str(out_adjacency))

    # 5) Plot risk-scaled nodes
    plot_risk_graph(G, str(out_image_risk))


# ---------------------------- Crime log processing --------------------------

OPENAI_RPM_LIMIT = 500
_rate_window_start = time.time()
_rate_window_count = 0


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass


def _enforce_openai_rate_limit() -> None:
    global _rate_window_start, _rate_window_count
    now = time.time()
    elapsed = now - _rate_window_start
    if elapsed >= 60:
        _rate_window_start = now
        _rate_window_count = 0
        return
    if _rate_window_count >= OPENAI_RPM_LIMIT:
        time.sleep(max(0.0, 60 - elapsed))
        _rate_window_start = time.time()
        _rate_window_count = 0


def process_crime_log(
    input_xlsx: str | Path,
    out_csv_base: str | Path = "data/crime_log_processed",
    address_column: str = "Location",
    openai_model: str = "gpt-5-nano",
) -> None:
    """Process the raw crime log: geocode locations and assign severities.

    Requires GOOGLE_MAPS_API_KEY and (optionally) OPENAI_API_KEY in the
    environment. Writes two CSVs: `<base>_location.csv` (geocoded) and
    `<base>.csv` (with severity column).
    """
    import re
    import time
    import urllib.parse
    import urllib.request

    # Load .env if present and Excel
    _maybe_load_dotenv()
    # Load Excel
    df = pd.read_excel(input_xlsx)

    # Geocode via Google Maps
    key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY not set in environment")
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    lats, lons = [], []
    for i, address in enumerate(df[address_column].astype(str).tolist(), 1):
        cleaned = re.sub(r"(?i)university of illinois", "", address)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
        params = {"address": f"{cleaned} Champaign", "key": key}
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("status") != "OK":
            raise RuntimeError(f"Geocoding failed: {data.get('status')} - {data.get('error_message','')}")
        loc = data["results"][0]["geometry"]["location"]
        lats.append(loc["lat"])
        lons.append(loc["lng"])
    df_geo = df.copy()
    df_geo["lat"], df_geo["lon"] = lats, lons

    # Save geocoded CSV
    Path(out_csv_base).parent.mkdir(parents=True, exist_ok=True)
    geo_path = f"{out_csv_base}_location.csv"
    df_geo.to_csv(geo_path, index=False)

    # Severity classification
    KNOWN = {
        "underage drinking": 1,
        "liquor law violation": 1,
        "open container": 1,
        "noise complaint": 1,
        "vandalism": 2,
        "criminal damage": 2,
        "trespass": 2,
        "theft": 3,
        "retail theft": 3,
        "vehicle theft": 3,
        "burglary": 4,
        "robbery": 4,
        "aggravated battery": 5,
        "weapons offense": 5,
        "sexual assault": 5,
    }

    def normalize_description(text: str) -> str:
        s = (text or "").lower().strip()
        s = re.sub(r"\s+", " ", s)
        return s

    sev_cache: dict[str, int] = {}
    cache_path = Path("severity_cache.json")
    if cache_path.exists():
        try:
            sev_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            sev_cache = {}

    # Optional OpenAI fallback
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    client = OpenAI(api_key=openai_key)

    def classify_severity(desc: str) -> int:
        n = normalize_description(desc)
        if n in sev_cache:
            return int(sev_cache[n])
        for k, v in KNOWN.items():
            if k in n:
                sev_cache[n] = int(v)
                return int(v)
        system = (
            "You are a strict crime severity classifier. Output only a single digit 1-5.\n"
            "Scale: 1=minor (underage drinking, noise), 2=low harm (trespass, vandalism), "
            "3=property crimes (theft), 4=threat/force (robbery, burglary), 5=violent/sexual/weapons."
        )
        prompt = f"{system}\n\nCrime description: {desc}\nSeverity (1-5) only:"
        # Rate control
        _enforce_openai_rate_limit()
        resp = client.responses.create(model=openai_model, input=prompt)
        # count call for throttle bookkeeping
        global _rate_window_count  # type: ignore
        _rate_window_count += 1
        sev = int(str(resp.output_text).strip()[0])
        sev_cache[n] = sev
        return sev

    df_out = df_geo.copy()
    df_out["severity"] = df_out["Description"].apply(classify_severity)
    out_path = f"{out_csv_base}.csv"
    df_out.to_csv(out_path, index=False)
    cache_path.write_text(json.dumps(sev_cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------------- CLI ------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the one-time setup tool.

    Returns:
        argparse.Namespace with `cmd` indicating the subcommand and its flags.
    """
    p = argparse.ArgumentParser(description="ALMA one-time setup tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("build-graph", help="Build UIUC road graph and exports")
    g.add_argument("--west", type=float, default=-88.24442)
    g.add_argument("--south", type=float, default=40.09396)
    g.add_argument("--east", type=float, default=-88.21858)
    g.add_argument("--north", type=float, default=40.11668)
    g.add_argument("--crimes-csv", default="data/crime_log_processed.csv")
    g.add_argument("--out-adjacency", default="data/uiuc_graph.json")
    g.add_argument("--out-image-osmnx", default="assets/uiuc_graph.png")
    g.add_argument("--out-image-simple", default="assets/uiuc_graph_simple.png")
    g.add_argument("--out-image-risk", default="assets/uiuc_graph_risk.png")
    g.add_argument("--tolerance-m", type=float, default=15.0)

    c = sub.add_parser("process-crime", help="Process crime log: geocode + severity")
    c.add_argument("--input-xlsx", default="data/Clery Crime Log - Police Contacts Only - 2021-October 31 2025.xlsx")
    c.add_argument("--out-base", default="data/crime_log_processed")
    c.add_argument("--address-column", default="Location")
    c.add_argument("--openai-model", default="gpt-5-nano")

    a = sub.add_parser("all", help="Run both steps: process-crime then build-graph")
    a.add_argument("--input-xlsx", default="data/Clery Crime Log - Police Contacts Only - 2021-October 31 2025.xlsx")
    a.add_argument("--out-base", default="data/crime_log_processed")
    a.add_argument("--address-column", default="Location")
    a.add_argument("--openai-model", default="gpt-5-nano")
    a.add_argument("--west", type=float, default=-88.24442)
    a.add_argument("--south", type=float, default=40.09396)
    a.add_argument("--east", type=float, default=-88.21858)
    a.add_argument("--north", type=float, default=40.11668)
    a.add_argument("--out-adjacency", default="data/uiuc_graph.json")
    a.add_argument("--out-image-osmnx", default="assets/uiuc_graph.png")
    a.add_argument("--out-image-simple", default="assets/uiuc_graph_simple.png")
    a.add_argument("--out-image-risk", default="assets/uiuc_graph_risk.png")
    a.add_argument("--tolerance-m", type=float, default=15.0)

    return p.parse_args()


def main() -> None:
    """Entry point for `python -m alma.setup`.

    Dispatches to the selected subcommand. Exits with exceptions on missing
    dependencies or API keys to make failures visible during setup.
    """
    args = parse_args()
    if args.cmd == "build-graph":
        build_graph(
            args.west,
            args.south,
            args.east,
            args.north,
            args.crimes_csv,
            out_adjacency=args.out_adjacency,
            out_image_osmnx=args.out_image_osmnx,
            out_image_simple=args.out_image_simple,
            out_image_risk=args.out_image_risk,
            consolidate_tolerance_m=args.tolerance_m,
        )
    elif args.cmd == "process-crime":
        process_crime_log(
            input_xlsx=args.input_xlsx,
            out_csv_base=args.out_base,
            address_column=args.address_column,
            openai_model=args.openai_model,
        )
    elif args.cmd == "all":
        process_crime_log(
            input_xlsx=args.input_xlsx,
            out_csv_base=args.out_base,
            address_column=args.address_column,
            openai_model=args.openai_model,
        )
        build_graph(
            args.west,
            args.south,
            args.east,
            args.north,
            crimes_csv=f"{args.out_base}.csv",
            out_adjacency=args.out_adjacency,
            out_image_osmnx=args.out_image_osmnx,
            out_image_simple=args.out_image_simple,
            out_image_risk=args.out_image_risk,
            consolidate_tolerance_m=args.tolerance_m,
        )


if __name__ == "__main__":
    main()
