#!/usr/bin/env python3
"""
Build a simple NetworkX road graph for UIUC from a bounding box using OSMnx.

What this script does
- Downloads via `osmnx.graph_from_bbox(..., network_type="drive", simplify=True)`.
- Converts the OSMnx MultiDiGraph into a simple undirected NetworkX Graph.
- Each node has attributes: `lat`, `lon`, and `risk_factor` (default 1.0). Also keeps `x`,`y` for plotting.
- Plots the original OSMnx graph to an image file for best fidelity.

Notes
- Using a bounding box can truncate roads at the boundary — this is expected and OK.

Requirements
- Python packages: osmnx, networkx, matplotlib
  Install with: `pip install osmnx networkx matplotlib`
"""

import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def to_simple_graph(G_multi):
    """Convert an OSMnx Multi(Di)Graph to a simple undirected Graph with lat/lon/risk_factor.

    - Collapses parallel edges between the same nodes by keeping the shortest 'length'.
    - Adds node attributes: 'lat' from y, 'lon' from x, and 'risk_factor'=1.0.
    """
    G_undirected = ox.convert.to_undirected(G_multi)

    G = nx.Graph()
    # Preserve CRS/metadata if present
    try:
        G.graph.update(G_undirected.graph)
    except Exception:
        pass
    if "crs" not in G.graph:
        G.graph["crs"] = "epsg:4326"

    # Nodes: copy positions into lat/lon (and keep x/y), add risk_factor default
    for n, data in G_undirected.nodes(data=True):
        lat = data.get("y")
        lon = data.get("x")
        G.add_node(n, lat=lat, lon=lon, x=lon, y=lat, risk_factor=1.0)

    # Edges: keep the shortest length among parallel edges; carry geometry if present
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


def plot_graph(G, out_path="uiuc_osm_graph.png", dpi=200):
    """Plot the graph with OSMnx/matplotlib and export to an image file."""
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
    plt.close(fig)
    return out_path


def plot_simple_graph(G: nx.Graph, out_path="uiuc_osm_graph_simple.png", dpi=200):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)

    # Draw edges as straight lines between node positions
    for u, v in G.edges():
        x = [G.nodes[u]["lon"], G.nodes[v]["lon"]]
        y = [G.nodes[u]["lat"], G.nodes[v]["lat"]]
        ax.plot(x, y, color="#222222", linewidth=0.3, alpha=0.9)

    # Draw nodes
    xs = [data["lon"] for _, data in G.nodes(data=True)]
    ys = [data["lat"] for _, data in G.nodes(data=True)]
    ax.scatter(xs, ys, s=5, color="#444444", zorder=3)

    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_risk_graph(G: nx.Graph, out_path="assets/uiuc_graph_risk.png"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw edges
    for u, v in G.edges():
        x = [G.nodes[u]["lon"], G.nodes[v]["lon"]]
        y = [G.nodes[u]["lat"], G.nodes[v]["lat"]]
        ax.plot(x, y, color="#000000", linewidth=1.1, zorder=1)

    # Node sizes based on risk_factor
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


def attach_crimes_to_graph(G: nx.Graph, csv_path, id_col="Number", lat_col="lat", lon_col="lon", sev_col="severity"):
    df = pd.read_csv(csv_path)

    nodes = list(G.nodes())
    node_lats = np.array([G.nodes[n]["lat"] for n in nodes], dtype=float)
    node_lons = np.array([G.nodes[n]["lon"] for n in nodes], dtype=float)

    # Initialize per-node accumulators
    for n in nodes:
        G.nodes[n]["crimes"] = []
        G.nodes[n]["_sev_sum"] = 0.0
        G.nodes[n]["_sev_count"] = 0

    for _, row in df.iterrows():
        clat = float(row[lat_col])
        clon = float(row[lon_col])
        cnum = str(row[id_col])
        csev = float(row[sev_col])

        # Compute distances to all nodes, pick nearest
        dists = haversine_dist(clat, clon, node_lats, node_lons)
        idx = int(np.argmin(dists))
        nid = nodes[idx]

        G.nodes[nid]["crimes"].append(cnum)
        G.nodes[nid]["_sev_sum"] += csev
        G.nodes[nid]["_sev_count"] += 1

    # Finalize risk_factor as severity scaled by crime count (log-damped)
    for n in nodes:
        cnt = G.nodes[n]["_sev_count"]
        if cnt > 0:
            avg = G.nodes[n]["_sev_sum"] / cnt
            # Scale by log1p(count) to reward busier nodes without exploding
            G.nodes[n]["risk_factor"] = float(avg * (1.0 + np.log1p(cnt)))
        # Clean up temp fields
        G.nodes[n].pop("_sev_sum", None)
        G.nodes[n].pop("_sev_count", None)

def save_adjacency_json(G: nx.Graph, out_path: str) -> str:
    """Save a simple JSON adjacency list with node attributes.

    Format:
    {
      "node_id": {"lat": ..., "lon": ..., "risk_factor": 1.0, "neighbors": ["nbr1", ...]},
      ...
    }
    """
    import json

    adj = {}
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

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(adj, f, ensure_ascii=False, indent=2)
    return out_path

def main():
    # Fill these in as needed (WGS84 degrees)
    # Example bbox roughly covering the UIUC campus area
    north = 40.11668
    south = 40.09396
    east = -88.21858
    west = -88.24442
    output_image = "assets/uiuc_graph.png"            # original OSMnx graph image
    output_simple_image = "assets/uiuc_graph_simple.png"  # simple graph image
    output_adjacency = "data/uiuc_graph.json"         # JSON adjacency list output
    crimes_csv = "data/crime_log_processed.csv"         # processed crimes CSV (Number, lat, lon, severity)
    output_risk_image = "assets/uiuc_graph_risk.png"   # risk-scaled nodes image

    # 1) Load OSMnx graph from the bounding box
    G_raw = ox.graph_from_bbox([west, south, east, north], network_type="drive", simplify=True)

    # 1b) Consolidate intersections within a small tolerance (meters)
    G_temp = ox.project_graph(G_raw)
    G_consolidated = ox.consolidate_intersections(G_temp, tolerance=15, rebuild_graph=True)
    G_consolidated = ox.project_graph(G_consolidated, to_crs="EPSG:4326")

    # 2) Convert to simple undirected graph with lat/lon/risk_factor
    G = to_simple_graph(G_consolidated)

    # 3) Plot/export image (plot the original OSMnx graph for best fidelity)
    out_img = plot_graph(G_consolidated, out_path=output_image)
    print(f"Saved plot to: {out_img}")
    # Also plot the simplified graph G using a lightweight matplotlib plotter
    out_img_simple = plot_simple_graph(G, out_path=output_simple_image)
    print(f"Saved simple plot to: {out_img_simple}")

    # 4) Attach crimes to the simple graph and export adjacency
    attach_crimes_to_graph(G, crimes_csv)
    out_adj = save_adjacency_json(G, output_adjacency)
    print(f"Saved adjacency JSON to: {out_adj}")

    # 5) Plot risk-scaled node sizes
    out_risk = plot_risk_graph(G, out_path=output_risk_image)
    print(f"Saved risk plot to: {out_risk}")


if __name__ == "__main__":
    main()
