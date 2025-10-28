# Champaign Security Games

## Graph Class Documentation

The graph of Champaign county is implemented in a custom graph class.

### Quick Start

```
from graph import Graph

g = Graph()
g.add_node("A", risk_factor=0.8, coverage=0.2)
g.add_node("B", risk_factor=0.5, coverage=0.6)
g.add_edge("A", "B", weight=2.0)

print(g.get_edge_weight("A", "B"))  # 2.0
print(g.neighbors("A"))              # {'B': 2.0}
print(g.total_risk())                 # 1.3
```

### Core API

- `add_node(node, risk_factor=0.0, coverage=0.0, overwrite=False)`
- `add_nodes_from(iterable, default_risk=0.0, default_coverage=0.0)`
- `remove_node(node)`
- `set_risk_factor(node, value)` / `set_coverage(node, value)`
- `get_risk_factor(node)` / `get_coverage(node)` / `get_node_attrs(node)`
- `add_edge(u, v, weight=1.0)` / `remove_edge(u, v)`
- `get_edge_weight(u, v)` / `set_edge_weight(u, v, weight)`
- `neighbors(node)` / `degree(node)`
- `nodes()` / `edges()` / `number_of_nodes()` / `number_of_edges()`
- `total_risk()` / `total_coverage()`

Edges are undirected; the class keeps both directions consistent.
