
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
from yfiles_jupyter_graphs import GraphWidget


class EventGraphPlotter:
    """
    Plotter for 5W1H event graphs using yFiles, based on the full JSON format:

    {
      "events": [ {event_id, what, when, where, who, why, how, category, ...}, ... ],
      "edges":  [ { "source": "...", "target": "...", "similarity": ..., "component_link": ... }, ... ]
    }

    - You can pass either:
        * a path to a JSON file with this structure, OR
        * a Python dict with keys "events" and "edges".

    - Rebuilds a DiGraph using:
        * event_id as node id
        * edges list as directed edges

    - Provides:
        * plot_with_sidebar()  -> standard graph view with data sidebar
        * plot_on_map()        -> map-based visualization using lat/long
    """

    def __init__(
        self,
        graph_or_path: Union[str, Dict[str, Any]],
        ignore_events_without_coordinates: bool = True,
    ) -> None:
        """
        :param graph_or_path:
            - Path to a JSON file with keys "events" and "edges", OR
            - A Python dict with keys "events" and "edges".
        :param ignore_events_without_coordinates:
            - If True, events without lat/long are ignored only in the map view.
              They still appear in the sidebar graph.
        """
        if isinstance(graph_or_path, str):
            import json
            with open(graph_or_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        else:
            obj = graph_or_path

        self.events: List[Dict[str, Any]] = list(obj.get("events", []))
        self.edges_data: List[Dict[str, Any]] = list(obj.get("edges", []))

        self.ignore_events_without_coordinates = ignore_events_without_coordinates

        # Map event_id -> event dict
        self.events_by_id: Dict[str, Dict[str, Any]] = {
            e["event_id"]: e
            for e in self.events
            if isinstance(e, dict) and "event_id" in e
        }

        # Build the NetworkX graph (for analysis or debugging)
        self.graph: nx.DiGraph = self._build_graph()

        # Precompute node/edge lists for map visualization
        self.map_nodes, self.map_edges = self._build_map_graph_data()

    # ------------------------------------------------------------------
    # Internal graph construction (NetworkX)
    # ------------------------------------------------------------------

    def _build_graph(self) -> nx.DiGraph:
        """Build a DiGraph where:
        - Each node is an event (id = event_id).
        - Each edge comes from the 'edges' list in the JSON.
        """
        G = nx.DiGraph()

        # Add nodes
        for ev in self.events:
            if not isinstance(ev, dict):
                continue
            ev_id = ev.get("event_id")
            if not ev_id:
                continue
            G.add_node(ev_id, data=ev)

        # Add edges
        for e in self.edges_data:
            if not isinstance(e, dict):
                continue
            src = e.get("source")
            tgt = e.get("target")
            if not src or not tgt:
                continue
            if src not in G or tgt not in G:
                continue

            G.add_edge(
                src,
                tgt,
                similarity=float(e.get("similarity", 0.0)),
                component_link=bool(e.get("component_link", False)),
            )

        return G

    # ------------------------------------------------------------------
    # Helpers for map visualization
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_main_coordinate(event: Dict[str, Any]) -> Optional[List[float]]:
        """Extract [lat, long] from the first location with valid coordinates.

        Expects:
        event['where']['locations'] = list of dicts with 'lat' and 'long'.
        """
        where = event.get("where", {})
        locations = where.get("locations", [])

        # Defensive: if a single dict was stored instead of a list
        if isinstance(locations, dict):
            locations = [locations]

        for loc in locations:
            if not isinstance(loc, dict):
                continue
            lat = loc.get("lat")
            lng = loc.get("long")  # schema uses "long"
            if lat is not None and lng is not None:
                try:
                    return [float(lat), float(lng)]
                except Exception:
                    continue

        return None

    def _build_common_node_properties(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        """Common node properties used in both sidebar and map views."""
        where = ev.get("where", {})
        return {
            "label": ev.get("what", ""),
            "when": ev.get("when", ""),
            "who": ev.get("who", ""),
            "where_text": where.get("text", ""),
            "why": ev.get("why", ""),
            "how": ev.get("how", ""),
            "category": ev.get("category", ""),
            "locations": where.get("locations", []),
        }

    def _build_map_graph_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build node and edge lists for GraphWidget.map_layout()."""
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        # Nodes with coordinates
        for ev in self.events:
            if not isinstance(ev, dict):
                continue
            ev_id = ev.get("event_id")
            if not ev_id:
                continue

            coord = self._extract_main_coordinate(ev)

            if coord is None and self.ignore_events_without_coordinates:
                # Node will not appear in the map view, but still appears in sidebar view
                continue

            node_props = self._build_common_node_properties(ev)

            node: Dict[str, Any] = {
                "id": ev_id,
                "properties": node_props,
            }
            if coord is not None:
                node["coordinates"] = coord

            nodes.append(node)

        node_ids_with_coords = {n["id"] for n in nodes}

        # Edges only between nodes present in the map view
        for e in self.edges_data:
            if not isinstance(e, dict):
                continue
            src = e.get("source")
            tgt = e.get("target")
            if not src or not tgt:
                continue

            if src in node_ids_with_coords and tgt in node_ids_with_coords:
                edges.append(
                    {
                        "start": src,
                        "end": tgt,
                        "label": "",
                        "properties": {
                            "similarity": float(e.get("similarity", 0.0)),
                            "component_link": bool(e.get("component_link", False)),
                        },
                        "directed": True,
                    }
                )

        return nodes, edges

    # ------------------------------------------------------------------
    # Sidebar graph view
    # ------------------------------------------------------------------

    def plot_with_sidebar(
        self,
        start_panel: str = "Data",
        sidebar_enabled: bool = True,
    ) -> GraphWidget:
        """Plot the graph with a data sidebar showing 5W1H per node."""
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        # Nodes
        for ev in self.events:
            if not isinstance(ev, dict):
                continue
            ev_id = ev.get("event_id")
            if not ev_id:
                continue

            node_props = self._build_common_node_properties(ev)

            nodes.append(
                {
                    "id": ev_id,
                    "properties": node_props,
                }
            )

        # Edges (use the 'edges' list directly)
        for e in self.edges_data:
            if not isinstance(e, dict):
                continue
            src = e.get("source")
            tgt = e.get("target")
            if not src or not tgt:
                continue
            if src not in self.events_by_id or tgt not in self.events_by_id:
                continue

            edges.append(
                {
                    "start": src,
                    "end": tgt,
                    "label": "",
                    "properties": {
                        "similarity": float(e.get("similarity", 0.0)),
                        "component_link": bool(e.get("component_link", False)),
                    },
                    "directed": True,
                }
            )

        w = GraphWidget()
        w.nodes = nodes
        w.edges = edges

        w.set_sidebar(enabled=sidebar_enabled, start_with=start_panel)
        return w

    # ------------------------------------------------------------------
    # Map-based view
    # ------------------------------------------------------------------

    def plot_on_map(
        self,
        use_heat_mapping: bool = False,
    ) -> GraphWidget:
        """Plot the graph on a map using lat/long from event locations."""
        w = GraphWidget()
        w.nodes = list(self.map_nodes)  # shallow copy
        w.edges = list(self.map_edges)

        # Field that contains node coordinates
        w.node_coordinate_mapping = "coordinates"

        if use_heat_mapping:
            # Simple example: hotter nodes if they have more locations
            def heat(element: Dict[str, Any]) -> float:
                if "start" in element:
                    return 0.0  # edges are neutral
                props = element.get("properties", {})
                locs = props.get("locations", [])
                return min(1.0, len(locs) / 5.0)

            w.set_heat_mapping(heat)

        w.map_layout()
        return w
