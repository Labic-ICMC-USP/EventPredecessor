
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List

import yaml
from tqdm.auto import tqdm

from .graph_builder import PrecursorEventGraphBuilder
from .graph_io import save_full_graph_json
from .llm import EventLLMExtractor, LLMExecutor
from .logging_utils import get_logger
from .news import Article, PrecursorNewsCollector

logger = get_logger(__name__)


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}
    return cfg


class EventPredecessorPipeline:
    """High-level pipeline orchestrator for the EventPredecessor package.

    It reads configuration from a YAML file, collects precursor news windows,
    extracts 5W1H events via an LLM, and builds a precursor graph.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "EventPredecessorPipeline":
        cfg = _load_config(path)
        return cls(cfg)

    def _build_news_collector(self) -> PrecursorNewsCollector:
        news_cfg = self.config.get("news", {})
        keywords: List[str] = news_cfg.get("keywords", [])
        if not keywords:
            raise ValueError("Config 'news.keywords' must not be empty.")

        logger.info(
            "Keywords for news collection loaded.",
            extra={"extra_data": {"query": keywords}},
        )
        reference_start_str = news_cfg.get("reference_start")
        reference_end_str = news_cfg.get("reference_end")
        if not reference_start_str or not reference_end_str:
            raise ValueError("Config 'news.reference_start' and 'news.reference_end' are required.")

        reference_start = datetime.fromisoformat(reference_start_str)
        reference_end = datetime.fromisoformat(reference_end_str)

        window = news_cfg.get("window", "m")
        searcher_kwargs = news_cfg.get("searcher", {})

        collector = PrecursorNewsCollector(
            keywords=keywords,
            reference_start=reference_start,
            reference_end=reference_end,
            window=window,
            searcher_kwargs=searcher_kwargs,
        )
        return collector

    def _build_llm_executor(self) -> LLMExecutor:
        llm_cfg = self.config.get("llm", {})
        api_key = llm_cfg.get("api_key")
        api_key_env_var = llm_cfg.get("api_key_env_var", "OPENROUTER_API_KEY")

        if not api_key:
            api_key = os.getenv(api_key_env_var)

        if not api_key:
            raise RuntimeError(
                "No API key configured for LLM. Set 'llm.api_key' or the environment "
                f"variable '{api_key_env_var}'."
            )

        exe = LLMExecutor()
        exe.set_api_key(api_key)
        exe.set_model(llm_cfg.get("model_name", "mistralai/mistral-nemo"))
        exe.set_base_url(llm_cfg.get("base_url", "https://openrouter.ai/api/v1"))
        exe.set_temperature(float(llm_cfg.get("temperature", 0.0)))
        max_tokens = llm_cfg.get("max_tokens")
        if max_tokens is not None:
            exe.set_max_tokens(int(max_tokens))

        return exe

    def _build_event_extractor(self, base_executor: LLMExecutor) -> EventLLMExtractor:
        categories_cfg = self.config.get("categories", {})
        if not isinstance(categories_cfg, dict) or not categories_cfg:
            raise ValueError(
                "Config 'categories' must be a non-empty mapping {label: description}."
            )
        extractor = EventLLMExtractor(base_llm_executor=base_executor, categories=categories_cfg)
        return extractor

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _collect_news(self) -> List[Dict[str, Any]]:
        news_cfg = self.config.get("news", {})
        collector = self._build_news_collector()

        max_iterations = news_cfg.get("max_iterations")
        stop_when_no_articles = news_cfg.get("stop_when_no_articles", True)

        results = collector.collect(
            max_iterations=max_iterations,
            stop_when_no_articles=stop_when_no_articles,
        )
        logger.info(
            "News collection finished.",
            extra={
                "extra_data": {
                    "iterations": len(results),
                    "articles_total": sum(len(b["articles"]) for b in results),
                }
            },
        )
        return results

    def _extract_events(
        self,
        results: List[Dict[str, Any]],
        extractor: EventLLMExtractor,
    ):
        # Flatten tasks: (iteration, local_index, article)
        tasks = []
        for block in results:
            iteration = block["iteration"]
            for idx, art in enumerate(block["articles"]):
                tasks.append((iteration, idx, art))

        logger.info(
            "Starting LLM extraction.",
            extra={"extra_data": {"num_articles": len(tasks)}},
        )

        extracted_events = []
        paired_results = []

        def worker(iteration: int, idx: int, article: Article):
            ev = extractor.extract_event(article)
            return iteration, idx, article, ev

        llm_cfg = self.config.get("llm", {})
        max_workers = int(llm_cfg.get("max_workers", 20))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(worker, iteration, idx, art)
                for (iteration, idx, art) in tasks
            ]

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="LLM extraction",
            ):
                iteration, idx, art, ev = fut.result()
                if ev is None:
                    continue

                # Create a stable event_id based on (iteration, idx)
                if not ev.event_id:
                    ev.event_id = f"ev_{iteration}_{idx}"

                # Store plain event
                extracted_events.append(ev)

                # Store paired structure (Article + Event)
                paired_results.append(
                    {
                        "iteration": iteration,
                        "article": {
                            "title": art.title,
                            "description": art.description,
                            "published": art.published.isoformat(),
                            "url": art.url,
                            "source": art.source,
                        },
                        "event": ev.model_dump(),
                    }
                )

        logger.info(
            "LLM extraction finished.",
            extra={
                "extra_data": {
                    "events_extracted": len(extracted_events),
                    "pairs": len(paired_results),
                }
            },
        )
        return extracted_events, paired_results

    def _build_graph(self, paired_results):
        builder_cfg = self.config.get("builder", {})
        builder = PrecursorEventGraphBuilder(
            paired_results=paired_results,
            embedding_model_name=builder_cfg.get(
                "embedding_model_name",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            same_category_only=builder_cfg.get("same_category_only", True),
            connect_components=builder_cfg.get("connect_components", True),
        )
        G, stats = builder.build_graph()
        return G, stats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run the full pipeline and return a dictionary with results."""
        results = self._collect_news()
        llm_exec = self._build_llm_executor()
        extractor = self._build_event_extractor(llm_exec)

        events, paired_results = self._extract_events(results, extractor)
        G, stats = self._build_graph(paired_results)

        output_cfg = self.config.get("output", {})
        json_path = output_cfg.get("full_graph_json_path", "events_full_graph.json")
        save_full_graph_json(G, json_path)

        logger.info(
            "Pipeline finished.",
            extra={
                "extra_data": {
                    "json_path": json_path,
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                }
            },
        )

        return {
            "articles_windows": results,
            "events": events,
            "paired_results": paired_results,
            "graph": G,
            "stats": stats,
            "full_graph_json_path": json_path,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the EventPredecessor pipeline from a YAML configuration file.",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_path",
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    args = parser.parse_args()

    pipeline = EventPredecessorPipeline.from_yaml(args.config_path)
    pipeline.run()


if __name__ == "__main__":
    main()
