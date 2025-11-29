# Champaign Security Games

## Graph Class Documentation

The graph of Champaign county is implemented in a custom graph class.

### Quick Start

```
from graph import Graph

g = Graph()
g.add_node("A", risk_factor=0.8, lat=40.11, lon=-88.23)
g.add_node("B", risk_factor=0.5)
g.add_edge("A", "B", weight=2.0)

print(g.get_edge_weight("A", "B"))  # 2.0
print(g.neighbors("A"))              # {'B': 2.0}
print(g.total_risk())                 # 1.3
```

### Core API

- `add_node(node, risk_factor=0.0, lat=None, lon=None, overwrite=False)`
- `add_nodes_from(iterable, default_risk=0.0, default_lat=None, default_lon=None)`
- `remove_node(node)`
- `set_risk_factor(node, value)` / `get_risk_factor(node)` / `get_node(node)`
- `add_edge(u, v, weight=1.0)` / `remove_edge(u, v)`
- `get_edge_weight(u, v)` / `set_edge_weight(u, v, weight)`
- `neighbors(node)` / `degree(node)`
- `nodes()` / `edges()` / `number_of_nodes()` / `number_of_edges()`
- `total_risk()`

Nodes are represented by a `Node` class with id, `risk_factor`, and optional `lat`/`lon`.
Edges are undirected; the class keeps both directions consistent.
