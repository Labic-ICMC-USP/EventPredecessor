
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from dateutil import parser as dateparser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .logging_utils import get_logger

logger = get_logger(__name__)


class PrecursorEventGraphBuilder:
    """Build a directed graph of precursor events based on semantic similarity.

    - Nodes are events (one node per EventSchema).
    - Each node stores:
        * event: the event dict (5W1H + category)
        * article: the article dict that originated this event
        * time: a datetime used for temporal ordering
        * iteration: an integer iteration index (t, t-1, t-2, ...)
        * category: event['category']

    Step 1 (threshold edges):
    - For every "younger" event B, choose at most ONE best "older" parent A,
      such that:
        * time(A) < time(B) OR
          time(A) == time(B) and iteration(A) < iteration(B),
        * sim(A, B) >= mean + std over all candidate pairs,
        * (optionally) category(A) == category(B) if same_category_only=True.
      -> Guarantees at most ONE parent per event in this phase.

    Step 2 (component linking - restricted strategy):
    - Consider the undirected view of the graph.
    - Compute connected components and sort them by size (descending).
    - Take the largest component as the "MAIN" component.
    - For each other component C (from largest to smallest):
        * Phase A (strict, if same_category_only=True):
            - Find the best pair (u in MAIN, v in C) with:
                + valid temporal order (older -> newer),
                + same category,
                + dst (newer node) has in-degree == 0 in the current DiGraph,
                + maximum similarity.
        * Phase B (relaxed):
            - If Phase A failed (or same_category_only=False), repeat ignoring
              category, but still requiring:
                + valid temporal order,
                + dst in-degree == 0.
        * If a valid pair is found, add ONE edge older(u,v)->newer(u,v) and
          conceptually merge C into MAIN.
    - Never assign more than ONE parent to any node.
    """

    def __init__(
        self,
        paired_results: List[Dict[str, Any]],
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        same_category_only: bool = True,
        connect_components: bool = True,
    ) -> None:
        """
        :param paired_results: list of dicts produced earlier, each containing:
            {
                "iteration": int,
                "article": {...},
                "event": {...}
            }
        :param embedding_model_name: sentence-transformers model to use.
        :param same_category_only:
            If True, threshold phase only considers pairs with same category.
            In the component-link phase, we first try to respect categories and
            only relax this rule if no valid edge is found.
        :param connect_components:
            If True, run the second step to connect disconnected components.
        """
        self.paired_results = paired_results
        self.embedding_model_name = embedding_model_name
        self.same_category_only = same_category_only
        self.connect_components = connect_components

        self.model: Optional[SentenceTransformer] = None
        self.graph: nx.DiGraph = nx.DiGraph()
        self.stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformer model."""
        if self.model is None:
            logger.info(
                "Loading embedding model.",
                extra={"extra_data": {"model_name": self.embedding_model_name}},
            )
            self.model = SentenceTransformer(self.embedding_model_name)
        return self.model

    @staticmethod
    def _parse_time(event_dict: Dict[str, Any], article_dict: Dict[str, Any]) -> datetime:
        """Compute a datetime for ordering events.

        Priority:
        1) event['when'] if it is a parseable date string
        2) article['published'] if it is a parseable date string
        3) fallback: current UTC time
        """
        when = event_dict.get("when")
        if isinstance(when, str) and when.strip():
            try:
                dt = dateparser.parse(when)
                if dt is not None:
                    return dt.replace(tzinfo=None)
            except Exception:
                pass

        published = article_dict.get("published")
        if isinstance(published, str) and published.strip():
            try:
                dt = dateparser.parse(published)
                if dt is not None:
                    return dt.replace(tzinfo=None)
            except Exception:
                pass

        return datetime.utcnow().replace(tzinfo=None)

    @staticmethod
    def _is_older(
        t_i: float,
        t_j: float,
        it_i: float,
        it_j: float,
    ) -> Optional[Tuple[int, int]]:
        """Decide temporal direction between two nodes.

        Returns:
            (0, 1) if i -> j (i older, j newer),
            (1, 0) if j -> i (j older, i newer),
            None if there is no clear order (same time and same iteration).
        """
        if t_i < t_j or (t_i == t_j and it_i < it_j):
            return (0, 1)  # i older -> j
        if t_j < t_i or (t_i == t_j and it_j < it_i):
            return (1, 0)  # j older -> i
        return None

    def _connect_until_single_component(
        self,
        node_ids: List[str],
        sim_matrix: np.ndarray,
        times: List[datetime],
        iterations: List[int],
        categories: List[str],
    ) -> int:
        """Connect components, preserving at most one parent per node.

        Strategy:
        - Compute connected components of the undirected view.
        - Sort components by size (largest first).
        - MAIN = largest component (set of node ids).
        - For each other component C (largest -> smallest):
            * Phase A (if same_category_only=True):
                - Find best (src_idx, dst_idx, sim) connecting MAIN to C with:
                    + valid temporal order,
                    + same category,
                    + dst node has in-degree == 0,
                    + maximum similarity.
            * Phase B (fallback):
                - If Phase A fails, repeat ignoring category, still requiring
                  valid temporal order and dst in-degree == 0.
            - If a pair is found, add ONE directed edge and
              update MAIN to include nodes from C.
        - This adds at most (#components - 1) edges and never increases
          in-degree above 1 for any node.
        """
        undirected = self.graph.to_undirected()
        initial_components = list(nx.connected_components(undirected))

        if len(initial_components) <= 1:
            return 0  # already connected or single node

        # Sort components by size descending
        initial_components = sorted(initial_components, key=len, reverse=True)

        # MAIN = largest component
        main_nodes = set(initial_components[0])
        other_components = [set(c) for c in initial_components[1:]]

        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        times_arr = np.array([t.timestamp() for t in times], dtype=float)
        iters_arr = np.array(iterations, dtype=float)
        cats_arr = np.array(categories, dtype=object)

        def find_best_edge_for_component(
            comp_nodes: set,
            require_category: bool,
        ) -> Optional[Tuple[int, int, float]]:
            """Find best (src_idx, dst_idx, sim) connecting MAIN to comp_nodes.

            Constraints:
            - Temporal order must be clear.
            - If require_category=True, categories must match.
            - The newer node (dst) must have in-degree == 0.
            """
            best_sim = -1.0
            best: Optional[Tuple[int, int, float]] = None

            for nid_main in main_nodes:
                i = id_to_idx[nid_main]
                for nid_c in comp_nodes:
                    j = id_to_idx[nid_c]

                    order = self._is_older(
                        times_arr[i], times_arr[j], iters_arr[i], iters_arr[j]
                    )
                    if order is None:
                        continue  # no clear temporal relation

                    # Decide direction older -> newer
                    if order == (0, 1):
                        older_idx, newer_idx = i, j
                    else:
                        older_idx, newer_idx = j, i

                    older_id = node_ids[older_idx]
                    newer_id = node_ids[newer_idx]

                    # Ensure newer node has no parent yet
                    if self.graph.in_degree(newer_id) > 0:
                        continue

                    # Category constraint if required
                    if require_category and cats_arr[i] != cats_arr[j]:
                        continue

                    sim = float(sim_matrix[older_idx, newer_idx])
                    if sim > best_sim:
                        best_sim = sim
                        best = (older_idx, newer_idx, sim)

            return best

        new_edges = 0

        for comp_nodes in other_components:
            # If this component is already fully inside MAIN, skip
            if comp_nodes.issubset(main_nodes):
                continue

            pair: Optional[Tuple[int, int, float]] = None

            # Phase A: respect category if same_category_only=True
            if self.same_category_only:
                pair = find_best_edge_for_component(comp_nodes, require_category=True)

            # Phase B: relax category if needed
            if pair is None:
                pair = find_best_edge_for_component(comp_nodes, require_category=False)

            if pair is None:
                # Could not connect this component without violating
                # temporal or one-parent constraints.
                logger.warning(
                    "Could not connect component under current constraints.",
                    extra={
                        "extra_data": {
                            "component_size": len(comp_nodes),
                        }
                    },
                )
                continue

            older_idx, newer_idx, sim = pair
            src_id = node_ids[older_idx]
            dst_id = node_ids[newer_idx]

            if not self.graph.has_edge(src_id, dst_id):
                self.graph.add_edge(
                    src_id,
                    dst_id,
                    similarity=float(sim),
                    component_link=True,
                )
                new_edges += 1

            # Merge this component into MAIN
            main_nodes.update(comp_nodes)

        logger.info(
            "Component linking finished.",
            extra={"extra_data": {"new_edges": new_edges}},
        )
        return new_edges

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_graph(self) -> tuple[nx.DiGraph, Dict[str, Any]]:
        """Build the precursor event graph.

        Returns:
            (graph, stats) where:
                - graph is a networkx.DiGraph
                - stats is a dict with similarity statistics and edge counts.
        """
        if not self.paired_results:
            raise ValueError("No paired_results provided. Nothing to build.")

        # 1) Flatten nodes and add them to the graph
        node_ids: List[str] = []
        what_texts: List[str] = []
        times: List[datetime] = []
        categories: List[str] = []
        iterations: List[int] = []

        for pair in self.paired_results:
            ev = pair["event"]
            art = pair["article"]
            iteration = int(pair.get("iteration", 0))

            ev_id = ev.get("event_id")
            if not ev_id:
                ev_id = f"ev_{len(node_ids)}"

            t = self._parse_time(ev, art)
            cat = ev.get("category", "OTHERS")

            node_ids.append(ev_id)
            what_texts.append(ev.get("what", ""))
            times.append(t)
            categories.append(cat)
            iterations.append(iteration)

            self.graph.add_node(
                ev_id,
                event=ev,
                article=art,
                time=t,
                iteration=iteration,
                category=cat,
            )

        n = len(node_ids)
        logger.info(
            "Building precursor graph.",
            extra={"extra_data": {"num_events": n}},
        )

        # 2) Compute embeddings for all 'what' texts
        model = self._load_model()
        embeddings = model.encode(what_texts, convert_to_numpy=True)

        # 3) Compute cosine similarity matrix
        sim_matrix = cosine_similarity(embeddings, embeddings)

        # 4) Build mask for candidate pairs (A, B) where A is older than B
        times_arr = np.array([t.timestamp() for t in times], dtype=float)
        iters_arr = np.array(iterations, dtype=float)
        mask_not_diag = ~np.eye(n, dtype=bool)

        older_matrix = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                order = self._is_older(times_arr[i], times_arr[j], iters_arr[i], iters_arr[j])
                if order is None:
                    continue
                if order == (0, 1):
                    older_matrix[i, j] = True  # i older -> j

        candidate_mask = older_matrix & mask_not_diag

        # Enforce same category only in the threshold phase
        cats_arr = np.array(categories, dtype=object)
        if self.same_category_only:
            mask_cat = cats_arr[:, None] == cats_arr[None, :]
            candidate_mask &= mask_cat

        # 5) Extract candidate similarities (for stats and threshold)
        candidate_sims = sim_matrix[candidate_mask]
        if candidate_sims.size > 0:
            mean_sim = float(candidate_sims.mean())
            std_sim = float(candidate_sims.std())
            threshold = mean_sim + std_sim

            logger.info(
                "Similarity statistics.",
                extra={
                    "extra_data": {
                        "mean_similarity": mean_sim,
                        "std_similarity": std_sim,
                        "threshold": threshold,
                    }
                },
            )
        else:
            mean_sim = None
            std_sim = None
            threshold = None
            logger.warning(
                "No candidate pairs found for threshold phase. "
                "No edges will be added by the threshold rule.",
            )

        # 6) Threshold phase: for each potential child j, at most ONE best parent i
        edges_added_threshold = 0
        if threshold is not None:
            # Matrix of allowed edges above threshold
            mask_edge = candidate_mask & (sim_matrix >= threshold)

            for j in range(n):
                # All possible parents for j
                parent_indices = np.where(mask_edge[:, j])[0]
                if parent_indices.size == 0:
                    continue

                # Choose best parent by similarity
                sims = sim_matrix[parent_indices, j]
                best_idx = parent_indices[np.argmax(sims)]
                src_id = node_ids[best_idx]
                dst_id = node_ids[j]

                # Safety check: ensure no parent yet (should always be true here)
                if self.graph.in_degree(dst_id) > 0:
                    continue

                sim_val = float(sim_matrix[best_idx, j])
                self.graph.add_edge(
                    src_id,
                    dst_id,
                    similarity=sim_val,
                    component_link=False,
                )
                edges_added_threshold += 1

            logger.info(
                "Edges added by threshold rule.",
                extra={"extra_data": {"edges_threshold": edges_added_threshold}},
            )
        else:
            logger.info("Skipping threshold-based edges (no candidate pairs).")

        # 7) Component-link phase: connect components, keeping in-degree <= 1
        edges_added_components = 0
        if self.connect_components:
            edges_added_components = self._connect_until_single_component(
                node_ids=node_ids,
                sim_matrix=sim_matrix,
                times=times,
                iterations=iterations,
                categories=categories,
            )

        num_edges_total = edges_added_threshold + edges_added_components

        # Final number of connected components in undirected view
        num_components_final = nx.number_connected_components(self.graph.to_undirected())

        self.stats = {
            "num_events": n,
            "num_candidate_pairs": int(candidate_sims.size),
            "num_edges_threshold": int(edges_added_threshold),
            "num_edges_component_links": int(edges_added_components),
            "num_edges_total": int(num_edges_total),
            "mean_similarity": mean_sim,
            "std_similarity": std_sim,
            "threshold": threshold,
            "num_connected_components_final": num_components_final,
        }

        logger.info(
            "Graph construction finished.",
            extra={"extra_data": self.stats},
        )

        return self.graph, self.stats
