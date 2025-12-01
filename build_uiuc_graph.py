"""Minimal UIUC campus graph + optional Google geocoding."""

from graph import Graph
import os
import json
from urllib.parse import urlencode, quote_plus
from urllib.request import urlopen
from dotenv import load_dotenv

load_dotenv() 


def build_grid_part():
    g = Graph()

   
    streets_ns = ["University Ave", "Clark St", "Main St", "White St", 
                  "Stoughton St", "Springfield Ave", "Healey St", "Green St", 
                  "John St", "Daniel St", "Chalmers St", "Armory Ave"]
    streets_ew = ["1st St", "2nd St", "3rd St", "4th St", "5th St", "6th St", 
                  "Wright St"]
    
    for ew in streets_ew:
        for ns in streets_ns:
            g.add_node(f"{ew} & {ns}", risk_factor=1.0)

    def nid(ew, ns):
        return f"{ew} & {ns}"

    for ew in streets_ew:
        for i in range(len(streets_ns) - 1):
            g.add_edge(nid(ew, streets_ns[i]), nid(ew, streets_ns[i + 1]), weight=1.0)

    for ns in streets_ns:
        for j in range(len(streets_ew) - 1):
            g.add_edge(nid(streets_ew[j], ns), nid(streets_ew[j + 1], ns), weight=1.0)

    return g


def build_other_part(g):
    # Neil & [University, Springfield, Green, Stadium, Kirby, St. Marys]
    # Oak & [John, Daniel, Chalmers, Armory, Gregory, Stadium, Kirby, St. Marys]
    # Locust & [Green, John, Daniel, Chalmers, Armory]
    # North Quad
    # Bardeen Quad
    # Main Quad


    return g


def geocode_address(address, api_key):
    base = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    url = f"{base}?{urlencode(params, quote_via=quote_plus)}"
    try:
        with urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        loc = payload["results"][0]["geometry"]["location"]
        return float(loc["lat"]), float(loc["lng"])
    except Exception:
        raise RuntimeError(f"Failed to geocode: {address}")


def populate_lat_lon(g, api_key=None, city_suffix="Champaign"):
    key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
    for nid in g.nodes():
        lat, lon = geocode_address(f"{nid} {city_suffix}", key)
        node = g.get_node(nid)
        node.lat = lat
        node.lon = lon


def main():
    g = build_grid_part()
    # TODO build other part
    populate_lat_lon(g)
    g.save_pickle("uiuc_campus_graph.pkl")
    print(f"Saved graph: nodes={g.number_of_nodes()} edges={g.number_of_edges()}")

if __name__ == "__main__":
    main()
