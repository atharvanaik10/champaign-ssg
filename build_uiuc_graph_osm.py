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
        node_size=3,
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
        adj[str(n)] = {
            "lat": data.get("lat"),
            "lon": data.get("lon"),
            "risk_factor": data.get("risk_factor", 1.0),
            "neighbors": [str(v) for v in G.neighbors(n)],
        }

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
    output_image = "assets/uiuc_osm_graph.png"          # output image path (png/jpg/pdf/svg)
    output_adjacency = "data/uiuc_osm_adj.json"       # JSON adjacency list output

    # 1) Load OSMnx graph from the bounding box
    G_raw = ox.graph_from_bbox([west, south, east, north], network_type="drive", simplify=True)

    # 2) Convert to simple undirected graph with lat/lon/risk_factor
    G = to_simple_graph(G_raw)

    # 3) Plot/export image (plot the original OSMnx graph for best fidelity)
    out_img = plot_graph(G_raw, out_path=output_image)
    print(f"Saved plot to: {out_img}")

    # 4) Export simple adjacency list JSON
    out_adj = save_adjacency_json(G, output_adjacency)
    print(f"Saved adjacency JSON to: {out_adj}")


if __name__ == "__main__":
    main()
