from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from alma.data import load_graph_for_animation

from backend.schemas import GraphResponse

router = APIRouter(tags=["graph"])


@router.get("/graph", response_model=GraphResponse)
def get_graph(graph_path: str = Query(..., min_length=1)) -> GraphResponse:
    try:
        nodes, edges = load_graph_for_animation(graph_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    features: list[dict] = []
    for idx, edge in enumerate(edges):
        p1, p2 = edge
        features.append(
            {
                "type": "Feature",
                "id": idx,
                "properties": {},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [list(p1), list(p2)],
                },
            }
        )
    for node_id, coords in nodes.items():
        features.append(
            {
                "type": "Feature",
                "properties": {"node_id": node_id},
                "geometry": {
                    "type": "Point",
                    "coordinates": [coords[0], coords[1]],
                },
            }
        )
    return GraphResponse(features=features)
