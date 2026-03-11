from __future__ import annotations

"""ALMA setup tool: build the road graph and process the crime log.

This module prepares the required data artifacts for the UIUC/Champaign demo:
1) a cleaned/collapsed road graph with per-node risk derived from the processed
   crime log, and 2) geocoded crime CSVs with severity scores.

Important: configuration is controlled via module-level constants below. We no
longer accept CLI arguments for these settings. Adjust the values at the top of
this file and run the module, e.g. `python -m alma.setup`.

Dependencies (required)
-----------------------
- OSMnx: downloads and simplifies the road network.
- Google Maps Geocoding API: converts crime addresses to lat/lon and performs
  reverse geocoding for node addresses (requires `GOOGLE_MAPS_API_KEY`).
- OpenAI API: classifies crime descriptions into severities 1–5
  (requires `OPENAI_API_KEY`).

This tool fails fast if required dependencies or API keys are missing.
"""

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

# ---------------------------------------------------------------------------
# Global configuration (edit these values to control behavior)
# ---------------------------------------------------------------------------
# Build-graph settings: bounding box for UIUC/Champaign area and I/O paths.
WEST = -88.24442
SOUTH = 40.08267
EAST = -88.21858
NORTH = 40.11668
CRIMES_CSV = "data/crime_log_processed.csv"
OUT_ADJACENCY = "data/uiuc_graph.json"
OUT_IMAGE_OSMNX = "assets/uiuc_graph.png"
OUT_IMAGE_SIMPLE = "assets/uiuc_graph_simple.png"
OUT_IMAGE_RISK = "assets/uiuc_graph_risk.png"
CONSOLIDATE_TOLERANCE_M = 15.0

# Reverse geocoding behavior for graph nodes.
# - Addresses are attached to each node via Google Maps Reverse Geocoding.
# - Results are cached to avoid redundant API calls across runs.
REVERSE_GEOCODE_CACHE = Path("cache/reverse_geocode_cache.json")

# Crime processing configuration (forward geocode + severity classification).
CRIME_INPUT_XLSX = "data/Clery Crime Log - Police Contacts Only - 2021-October 31 2025.xlsx"
CRIME_OUT_BASE = "data/crime_log_processed"
CRIME_ADDRESS_COLUMN = "Location"
OPENAI_MODEL = "gpt-5-nano"

# Which action to run when executing this module directly.
# Choose one of: "build-graph", "process-crime", or "all".
RUN_MODE = "build-graph"


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
            "address": data.get("address"),
            "risk_factor": data.get("risk_factor", 1.0),
            "neighbors": [str(v) for v in G.neighbors(n)],
        }
        if "crimes" in data:
            entry["crimes"] = [str(x) for x in data["crimes"]]
        adj[str(n)] = entry
    Path(out_path).write_text(json.dumps(adj, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def attach_addresses_to_graph(G: nx.Graph) -> None:
    """Reverse geocode each node's lat/lon to a human-readable address.

    Uses Google Maps Geocoding API with the `latlng` parameter. Mimics the
    forward geocoding approach used in `process_crime_log` but for reverse
    geocoding. Results are cached between runs to limit API usage.
    """
    import urllib.parse
    import urllib.request

    _maybe_load_dotenv()
    key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY not set in environment")

    # Load and prepare cache
    cache: dict[str, str] = {}
    try:
        if REVERSE_GEOCODE_CACHE.exists():
            cache = json.loads(REVERSE_GEOCODE_CACHE.read_text(encoding="utf-8"))
    except Exception:
        cache = {}

    base_url = "https://maps.googleapis.com/maps/api/geocode/json"

    def key_for(lat: float, lon: float) -> str:
        return f"{lat:.6f},{lon:.6f}"

    updated = False
    for n, data in G.nodes(data=True):
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))
        cache_key = key_for(lat, lon)
        if cache_key in cache:
            G.nodes[n]["address"] = cache[cache_key]
            continue
        params = {"latlng": f"{lat},{lon}", "key": key}
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        status = payload.get("status")
        if status != "OK":
            raise RuntimeError(
                f"Reverse geocoding failed ({status}): {payload.get('error_message','')}"
            )
        formatted = payload["results"][0].get("formatted_address")
        G.nodes[n]["address"] = formatted
        cache[cache_key] = formatted
        updated = True

    if updated:
        REVERSE_GEOCODE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        REVERSE_GEOCODE_CACHE.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )


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

    # 4) Attach crimes and reverse-geocoded addresses, then export adjacency
    attach_crimes_to_graph(G, str(crimes_csv))
    attach_addresses_to_graph(G)
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


def main() -> None:
    """Entry point for `python -m alma.setup` using global config.

    Set `RUN_MODE` at the top of this file to control which action runs.
    """
    if RUN_MODE == "build-graph":
        build_graph(
            WEST,
            SOUTH,
            EAST,
            NORTH,
            CRIMES_CSV,
            out_adjacency=OUT_ADJACENCY,
            out_image_osmnx=OUT_IMAGE_OSMNX,
            out_image_simple=OUT_IMAGE_SIMPLE,
            out_image_risk=OUT_IMAGE_RISK,
            consolidate_tolerance_m=CONSOLIDATE_TOLERANCE_M,
        )
    elif RUN_MODE == "process-crime":
        process_crime_log(
            input_xlsx=CRIME_INPUT_XLSX,
            out_csv_base=CRIME_OUT_BASE,
            address_column=CRIME_ADDRESS_COLUMN,
            openai_model=OPENAI_MODEL,
        )
    elif RUN_MODE == "all":
        process_crime_log(
            input_xlsx=CRIME_INPUT_XLSX,
            out_csv_base=CRIME_OUT_BASE,
            address_column=CRIME_ADDRESS_COLUMN,
            openai_model=OPENAI_MODEL,
        )
        build_graph(
            WEST,
            SOUTH,
            EAST,
            NORTH,
            crimes_csv=f"{CRIME_OUT_BASE}.csv",
            out_adjacency=OUT_ADJACENCY,
            out_image_osmnx=OUT_IMAGE_OSMNX,
            out_image_simple=OUT_IMAGE_SIMPLE,
            out_image_risk=OUT_IMAGE_RISK,
            consolidate_tolerance_m=CONSOLIDATE_TOLERANCE_M,
        )
    else:
        raise SystemExit(
            "Invalid RUN_MODE. Use one of: 'build-graph', 'process-crime', 'all'."
        )


if __name__ == "__main__":
    main()
