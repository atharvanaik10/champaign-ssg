"""
Hardcoded UIUC campus street-intersection graph with minimal attributes.

This version uses a fixed set of real intersections around the UIUC campus core
(bounded approximately by Springfield Ave (N), Gregory Dr (S), Wright St (W),
and Lincoln Ave (E)). All node attributes are set to 1, and all edge weights
are set to 1. No Gaussian risk models, hotspots, or coverage — just a simple
unweighted grid-like graph of major campus streets.

Usage
-----
python build_uiuc_graph.py
"""

from graph import Graph


def build_grid_part() -> Graph:
    """Create a simple hardcoded UIUC campus intersection graph.

    Intersections are labeled as "<Street_EW> & <Street_NS>". Streets chosen
    are common throughfares in the campus core:
    - East/West: Springfield Ave, Green St, Armory Ave, Gregory Dr
    - North/South: Wright St, Goodwin Ave, Mathews Ave, Lincoln Ave

    Edges connect adjacent intersections along each street, forming a grid.
    All edge weights are 1. All nodes have risk_factor=1.0 and no coverage.
    """

    g = Graph()

    streets_ew = ["University Ave", "Clark St", "Main St", "White St", 
                  "Stoughton St", "Springfield Ave", "Healey St", "Green St", 
                  "John St", "Daniel St", "Chalmers St", "Armory Ave"]
    streets_ns = ["1st St", "2nd St", "3rd St", "4th St", "5th St", "6th St", "Wright St"]

    # Create nodes for each intersection with risk_factor=1.0 (no coverage used)
    nodes = [f"{ew} & {ns}" for ew in streets_ew for ns in streets_ns]
    for n in nodes:
        g.add_node(n, risk_factor=1.0)

    # Helper to format node IDs
    def nid(ew: str, ns: str) -> str:
        return f"{ew} & {ns}"

    # Connect intersections horizontally (along each E/W street)
    for ew in streets_ew:
        for i in range(len(streets_ns) - 1):
            u = nid(ew, streets_ns[i])
            v = nid(ew, streets_ns[i + 1])
            g.add_edge(u, v, weight=1.0)

    # Connect intersections vertically (along each N/S street)
    for ns in streets_ns:
        for j in range(len(streets_ew) - 1):
            u = nid(streets_ew[j], ns)
            v = nid(streets_ew[j + 1], ns)
            g.add_edge(u, v, weight=1.0)

    return g


def main() -> None:
    outfile = "uiuc_campus_graph.pkl"
    g = build_grid_part()
    g.verify_undirected_invariants()
    g.save_pickle(outfile)
    print(f"Saved graph with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges to {outfile}")


if __name__ == "__main__":
    main()
