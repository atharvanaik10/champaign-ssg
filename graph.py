"""
Lightweight undirected weighted graph for security games.

Each node is represented as a Node object with:
- id: node identifier (hashable)
- risk_factor: float
- lat: Optional[float]
- lon: Optional[float]

Edges are undirected and carry a numeric weight (default 1.0).

Design goals:
- Minimal dependencies, predictable performance.
- Clear, typed API with helpful errors.
- Undirected consistency (u,v) == (v,u).

Example
-------
>>> g = Graph()
>>> g.add_node("A", risk_factor=0.8, lat=40.11, lon=-88.23)
>>> g.add_node("B", risk_factor=0.5)
>>> g.add_edge("A", "B", weight=2.0)
>>> g.get_edge_weight("A", "B")
2.0
>>> g.neighbors("A")
{'B': 2.0}
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, Iterator, List, Optional, Tuple


NodeId = Hashable


class Node:
    """A node in the graph with id, risk, and optional coordinates."""

    def __init__(
        self,
        node_id: Hashable,
        *,
        risk_factor: float = 0.0,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> None:
        self.id: Hashable = node_id
        self.risk_factor: float = float(risk_factor)
        self.lat: Optional[float] = None if lat is None else float(lat)
        self.lon: Optional[float] = None if lon is None else float(lon)

    def as_dict(self) -> Dict[str, object]:
        return {"id": self.id, "risk_factor": self.risk_factor, "lat": self.lat, "lon": self.lon}


class Graph:
    """Undirected weighted graph with per-node risk and coverage.

    Implementation details:
    - Nodes are stored in `_nodes` mapping NodeId -> NodeAttrs.
    - Adjacency is stored as `_adj[u][v] = weight` for undirected edges.
      Both directions are present and kept consistent.
    """

    def __init__(self) -> None:
        self._nodes: Dict[NodeId, Node] = {}
        self._adj: Dict[NodeId, Dict[NodeId, float]] = {}

    # --------- Node API ---------
    def add_node(
        self,
        node: NodeId,
        *,
        risk_factor: float = 0.0,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        overwrite: bool = False,
    ) -> None:
        """Add a node with attributes.

        - If the node exists and `overwrite=False`, attributes are left unchanged.
        - If the node exists and `overwrite=True`, attributes are replaced.
        """

        if node in self._nodes and not overwrite:
            return

        self._nodes[node] = Node(node, risk_factor=float(risk_factor), lat=lat, lon=lon)
        self._adj.setdefault(node, {})

    def add_nodes_from(
        self,
        nodes: Iterable[Tuple[NodeId, float, Optional[float], Optional[float]]] | Iterable[NodeId] | Iterable[Tuple[NodeId, float]] | Iterable[Tuple[NodeId, float, Optional[float]]],
        *,
        default_risk: float = 0.0,
        default_lat: Optional[float] = None,
        default_lon: Optional[float] = None,
        overwrite: bool = False,
    ) -> None:
        """Add many nodes.

        Accepts either an iterable of node IDs, pairs (node, risk), triples (node, risk, lat),
        or quadruples (node, risk, lat, lon).
        """

        for item in nodes:
            if isinstance(item, tuple):
                if len(item) == 2:
                    n, r = item  # type: ignore[misc]
                    self.add_node(n, risk_factor=float(r), lat=default_lat, lon=default_lon, overwrite=overwrite)
                elif len(item) == 3:
                    n, r, lat = item  # type: ignore[misc]
                    self.add_node(n, risk_factor=float(r), lat=lat, lon=default_lon, overwrite=overwrite)
                elif len(item) == 4:
                    n, r, lat, lon = item  # type: ignore[misc]
                    self.add_node(n, risk_factor=float(r), lat=lat, lon=lon, overwrite=overwrite)
                else:
                    raise ValueError("Node tuple must be (id), (id,risk), (id,risk,lat), or (id,risk,lat,lon)")
            else:
                self.add_node(item, risk_factor=default_risk, lat=default_lat, lon=default_lon, overwrite=overwrite)  # type: ignore[arg-type]

    def has_node(self, node: NodeId) -> bool:
        return node in self._nodes

    def remove_node(self, node: NodeId) -> None:
        """Remove a node and all incident edges.

        Raises KeyError if the node does not exist.
        """

        if node not in self._nodes:
            raise KeyError(f"Node not found: {node!r}")

        # Remove incident edges
        for nbr in list(self._adj.get(node, {}).keys()):
            self._adj[nbr].pop(node, None)
        self._adj.pop(node, None)
        self._nodes.pop(node, None)

    def set_risk_factor(self, node: NodeId, risk_factor: float) -> None:
        if node not in self._nodes:
            raise KeyError(f"Node not found: {node!r}")
        self._nodes[node].risk_factor = float(risk_factor)

    def get_risk_factor(self, node: NodeId) -> float:
        if node not in self._nodes:
            raise KeyError(f"Node not found: {node!r}")
        return self._nodes[node].risk_factor

    def get_node(self, node: NodeId) -> Node:
        if node not in self._nodes:
            raise KeyError(f"Node not found: {node!r}")
        return self._nodes[node]

    # --------- Edge API ---------
    def add_edge(self, u: NodeId, v: NodeId, *, weight: float = 1.0) -> None:
        """Add or update an undirected edge between `u` and `v`.

        Nodes are created on-the-fly if they do not yet exist.
        """

        if u == v:
            raise ValueError("Self-loops are not supported in this graph")

        # Ensure nodes exist
        if u not in self._nodes:
            self.add_node(u)
        if v not in self._nodes:
            self.add_node(v)

        w = float(weight)
        self._adj[u][v] = w
        self._adj[v][u] = w

    def has_edge(self, u: NodeId, v: NodeId) -> bool:
        return u in self._adj and v in self._adj[u]

    def get_edge_weight(self, u: NodeId, v: NodeId) -> float:
        if not self.has_edge(u, v):
            raise KeyError(f"Edge not found: ({u!r}, {v!r})")
        return self._adj[u][v]

    def set_edge_weight(self, u: NodeId, v: NodeId, weight: float) -> None:
        if not self.has_edge(u, v):
            raise KeyError(f"Edge not found: ({u!r}, {v!r})")
        w = float(weight)
        self._adj[u][v] = w
        self._adj[v][u] = w

    def remove_edge(self, u: NodeId, v: NodeId) -> None:
        if not self.has_edge(u, v):
            raise KeyError(f"Edge not found: ({u!r}, {v!r})")
        self._adj[u].pop(v, None)
        self._adj[v].pop(u, None)

    # --------- Queries ---------
    def nodes(self) -> Iterator[NodeId]:
        return iter(self._nodes.keys())

    def node_items(self) -> Iterator[Tuple[NodeId, Node]]:
        return iter(self._nodes.items())

    def edges(self) -> Iterator[Tuple[NodeId, NodeId, float]]:
        """Iterate over edges (u, v, weight) with u <= v by string repr.

        Ensures each undirected edge is yielded exactly once.
        """

        seen: set[Tuple[NodeId, NodeId]] = set()
        for u, nbrs in self._adj.items():
            for v, w in nbrs.items():
                key = (u, v) if repr(u) <= repr(v) else (v, u)
                if key in seen:
                    continue
                seen.add(key)
                yield key[0], key[1], w

    def neighbors(self, node: NodeId) -> Dict[NodeId, float]:
        if node not in self._nodes:
            raise KeyError(f"Node not found: {node!r}")
        return dict(self._adj.get(node, {}))

    def degree(self, node: NodeId) -> int:
        return len(self.neighbors(node))

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    def number_of_edges(self) -> int:
        # Count each undirected edge once
        return sum(1 for _ in self.edges())

    def total_risk(self) -> float:
        return sum(attrs.risk_factor for attrs in self._nodes.values())

    # coverage attribute removed; no total_coverage

    # --------- Serialization ---------
    def to_dict(self) -> Dict[str, object]:
        """Serialize graph to a plain dict (JSON-friendly)."""

        nodes: List[Tuple[str, float, Optional[float], Optional[float]]] = []
        for n, node in self._nodes.items():
            nodes.append((repr(n), node.risk_factor, node.lat, node.lon))

        edges: List[Tuple[str, str, float]] = []
        for u, v, w in self.edges():
            edges.append((repr(u), repr(v), w))

        return {"nodes": nodes, "edges": edges}

    # --------- Pickle I/O ---------
    def save_pickle(self, path: str) -> None:
        """Serialize the graph to a pickle file at `path`.

        Uses Python's pickle; objects must be pickleable (Node IDs included).
        """
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_pickle(cls, path: str) -> "Graph":
        """Load a graph instance from a pickle file at `path`."""
        import pickle

        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Pickle at {path!r} does not contain a {cls.__name__}")
        return obj

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[Tuple[NodeId, NodeId, float | int]] | Iterable[Tuple[NodeId, NodeId]],
        *,
        default_weight: float = 1.0,
    ) -> "Graph":
        """Build a graph from an iterable of edges.

        Edges can be (u, v) or (u, v, weight).
        """

        g = cls()
        for e in edges:
            if len(e) == 2:  # type: ignore[arg-type]
                u, v = e  # type: ignore[misc]
                g.add_edge(u, v, weight=default_weight)
            elif len(e) == 3:  # type: ignore[arg-type]
                u, v, w = e  # type: ignore[misc]
                g.add_edge(u, v, weight=float(w))
            else:
                raise ValueError("Edge tuple must be (u,v) or (u,v,w)")
        return g

    # --------- Utilities ---------
    def verify_undirected_invariants(self) -> None:
        """Sanity checks: symmetric edges and no dangling adjacency entries.

        Raises AssertionError if invariants do not hold.
        """

        for u, nbrs in self._adj.items():
            assert u in self._nodes, f"Adjacency contains unknown node {u!r}"
            for v, w in nbrs.items():
                assert v in self._nodes, f"Adjacency references unknown neighbor {v!r}"
                assert self._adj.get(v, {}).get(u, None) == w, (
                    f"Asymmetry detected between {u!r} and {v!r}"
                )

    # --------- Python dunder helpers ---------
    def __contains__(self, node: NodeId) -> bool:  # `node in g`
        return self.has_node(node)

    def __len__(self) -> int:  # `len(g)`
        return self.number_of_nodes()

    def __repr__(self) -> str:
        return f"Graph(nodes={len(self._nodes)}, edges={self.number_of_edges()})"
