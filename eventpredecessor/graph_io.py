
from __future__ import annotations

import json
from typing import Any, Dict, List

import networkx as nx

from .logging_utils import get_logger

logger = get_logger(__name__)


def graph_to_full_json(G: nx.DiGraph) -> Dict[str, Any]:
    """Export the full graph structure: events (nodes) + edges.

    - Nodes: 'events' (list of dicts with 5W1H + category + event_id).
    - Edges: 'edges' (list of dicts with source, target, similarity, etc.).

    Requirements:
    - Each node must have an 'event' attribute that is a dict (5W1H + category).
    - Edges may have:
        * 'similarity' (float)
        * 'component_link' (bool) to mark edges added in the component-link phase.
    """
    events: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # Nodes
    for node_id, data in G.nodes(data=True):
        ev = dict(data.get("event", {}))  # shallow copy
        ev["event_id"] = node_id  # ensure event_id matches the node id
        events.append(ev)

    # Edges
    for u, v, edata in G.edges(data=True):
        edges.append(
            {
                "source": u,
                "target": v,
                "similarity": float(edata.get("similarity", 0.0)),
                "component_link": bool(edata.get("component_link", False)),
            }
        )

    return {
        "events": events,
        "edges": edges,
    }


def save_full_graph_json(G: nx.DiGraph, path: str) -> None:
    """Save the full graph (events + edges) to a JSON file."""
    graph_obj = graph_to_full_json(G)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph_obj, f, ensure_ascii=False, indent=2)
    logger.info(
        "Saved full graph JSON.",
        extra={
            "extra_data": {
                "events": len(graph_obj["events"]),
                "edges": len(graph_obj["edges"]),
                "path": path,
            }
        },
    )


def load_full_graph_from_json(path: str) -> nx.DiGraph:
    """Rebuild a DiGraph from a JSON saved with graph_to_full_json / save_full_graph_json."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    events = obj.get("events", [])
    edges = obj.get("edges", [])

    G = nx.DiGraph()

    # Nodes
    for ev in events:
        if not isinstance(ev, dict):
            continue
        ev_id = ev.get("event_id")
        if not ev_id:
            continue
        G.add_node(ev_id, event=ev)

    # Edges
    for e in edges:
        src = e.get("source")
        tgt = e.get("target")
        if src in G and tgt in G:
            G.add_edge(
                src,
                tgt,
                similarity=float(e.get("similarity", 0.0)),
                component_link=bool(e.get("component_link", False)),
            )

    logger.info(
        "Loaded graph JSON.",
        extra={
            "extra_data": {
                "path": path,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
            }
        },
    )
    return G
