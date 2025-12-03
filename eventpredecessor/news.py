
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta
from gnews import GNews

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Article:
    """Simple container for a news article returned by GNews.

    This object holds the minimal information we need from each news item.
    """
    title: str
    description: str
    published: datetime
    url: str
    source: str
    raw: Dict[str, Any]  # raw JSON/dict returned by GNews


class GNewsSearcher:
    """Small wrapper around the 'gnews' library for a given date interval."""

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        language: str = "pt",
        country: str = "BR",
        max_results: int = 10,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date

        # GNews expects (year, month, day) tuples for custom date ranges
        self.client = GNews(
            language=language,
            country=country,
            max_results=max_results,
            start_date=(start_date.year, start_date.month, start_date.day),
            end_date=(end_date.year, end_date.month, end_date.day),
        )

    def _parse_published(self, raw_date: Any) -> datetime:
        """Best-effort conversion of the 'published date' field to datetime."""
        if isinstance(raw_date, datetime):
            return raw_date
        if isinstance(raw_date, str):
            try:
                return dateparser.parse(raw_date)
            except Exception:
                pass
        # Fallback: current UTC time (naive)
        return datetime.utcnow()

    

    def search(self, keywords: Iterable[str]) -> List[Article]:
        from typing import Iterable, List, Set

        """Search news once per keyword and merge results, removing duplicates by URL."""
        articles: List[Article] = []
        seen_urls: Set[str] = set()


        for kw in keywords:
            logger.info(
                "GNews search for single keyword/query.",
                extra={"extra_data": {"query": kw}},
            )
            raw_results = self.client.get_news(kw)
            logger.info(
                "RAW results received from GNews.",
                extra={"extra_data": {"results": raw_results}},
            )

            for item in raw_results:
                title = item.get("title", "")
                description = item.get("description", "")
                published_raw = (
                    item.get("published date")
                    or item.get("published")
                    or ""
                )
                url = item.get("url", "")
                source_info = item.get("publisher") or item.get("source") or {}
                source_title = ""
                if isinstance(source_info, dict):
                    source_title = (
                        source_info.get("title", "")
                        or source_info.get("name", "")
                    )

                # pular se não tem URL ou se já vimos essa notícia
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                published = self._parse_published(published_raw)

                articles.append(
                    Article(
                        title=title,
                        description=description,
                        published=published,
                        url=url,
                        source=source_title,
                        raw=item,
                    )
                )

        logger.info(
            "Finished GNews search for all keywords.",
            extra={"extra_data": {"total_articles": len(articles)}},
        )
        return articles



class PrecursorNewsCollector:
    """Collect news for a reference window and precursor windows.

    Semantics of the 'window' parameter:

    - iteration 1:
        [reference_start, reference_end]  # exactly what the user asked
    - iteration 2:
        [reference_start - step, reference_start]
    - iteration 3:
        [reference_start - 2 * step, reference_start - step]
    - and so on...

    Where 'step' depends on `window`:
        'd' -> 1 day
        'w' -> 1 week
        'm' -> 1 calendar month (relativedelta(months=1))
        'y' -> 1 calendar year (relativedelta(years=1))
    """

    def __init__(
        self,
        keywords: Iterable[str],
        reference_start: datetime,
        reference_end: datetime,
        window: str,
        searcher_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.keywords = list(keywords)
        self.reference_start = reference_start
        self.reference_end = reference_end
        self.window = window.lower()

        if self.window not in {"d", "w", "m", "y"}:
            raise ValueError("window must be one of: 'd', 'w', 'm', 'y'.")

        if reference_end <= reference_start:
            raise ValueError("reference_end must be after reference_start.")

        # Define the step size for the precursor windows
        if self.window == "d":
            self.step_delta = timedelta(days=1)
        elif self.window == "w":
            self.step_delta = timedelta(weeks=1)
        elif self.window == "m":
            self.step_delta = relativedelta(months=1)
        else:  # 'y'
            self.step_delta = relativedelta(years=1)

        self.searcher_kwargs = searcher_kwargs or {}

    def collect(
        self,
        max_iterations: Optional[int] = None,
        stop_when_no_articles: bool = True,
    ) -> List[Dict[str, Any]]:
        """Collect articles across multiple iterations.

        Returns a list of dicts:
        [
            {
                "iteration": i,
                "start": datetime,
                "end": datetime,
                "articles": List[Article],
            },
            ...
        ]
        """
        all_results: List[Dict[str, Any]] = []

        # iteration 1: exactly the user-provided window
        iteration = 0

        # this anchor is always the *start* of the last window,
        # so that we move backward from it for subsequent windows.
        anchor_start = self.reference_start

        while True:
            iteration += 1

            if max_iterations is not None and iteration > max_iterations:
                logger.info(
                    "Max iterations reached; stopping.",
                    extra={"extra_data": {"max_iterations": max_iterations}},
                )
                break

            if iteration == 1:
                # First iteration: use [reference_start, reference_end]
                current_start = self.reference_start
                current_end = self.reference_end
            else:
                # For iteration >= 2:
                #   new_end   = previous anchor_start
                #   new_start = new_end - step_delta
                current_end = anchor_start
                current_start = current_end - self.step_delta
                # Update anchor_start for the next iteration
                anchor_start = current_start

            logger.info(
                "Searching news window.",
                extra={
                    "extra_data": {
                        "iteration": iteration,
                        "start": current_start.date().isoformat(),
                        "end": current_end.date().isoformat(),
                    }
                },
            )

            searcher = GNewsSearcher(
                start_date=current_start,
                end_date=current_end,
                **self.searcher_kwargs,
            )
            articles = searcher.search(self.keywords)
            logger.info(
                "Collected articles for window.",
                extra={
                    "extra_data": {
                        "iteration": iteration,
                        "articles_count": len(articles),
                    }
                },
            )

            all_results.append(
                {
                    "iteration": iteration,
                    "start": current_start,
                    "end": current_end,
                    "articles": articles,
                }
            )

            if stop_when_no_articles and len(articles) == 0:
                logger.info(
                    "No articles returned in this window; stopping early.",
                    extra={"extra_data": {"iteration": iteration}},
                )
                break

        return all_results
