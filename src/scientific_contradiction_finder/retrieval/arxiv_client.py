"""arXiv API client parsing Atom XML with politeness and caching."""

import asyncio
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import httpx

from scientific_contradiction_finder.retrieval.cache import SQLiteCache
from scientific_contradiction_finder.schemas import Paper

ARXIV_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivClient:
    """Async client for arXiv API with HTTPS, redirect support, caching, and politeness throttling."""

    def __init__(
        self,
        cache: Optional[SQLiteCache] = None,
        timeout_seconds: int = 15,
        max_retries: int = 3,
        offline_mode: bool = False,
    ):
        self.cache = cache or SQLiteCache()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.offline_mode = offline_mode
        self.base_url = "https://export.arxiv.org/api/query"

    async def search_with_diagnostics(
        self, query: str, limit: int = 20
    ) -> Tuple[List[Paper], int, str]:
        clean_query = query.strip()
        cache_key = self.cache.compute_key(
            "arxiv", "/query", {"query": clean_query, "limit": limit}
        )
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return [Paper(**p) for p in cached_data.get("papers", [])], 200, "CACHED"

        if self.offline_mode:
            return [], 0, "OFFLINE_MODE"

        terms = [t for t in clean_query.split() if t.isalnum()]
        search_expr = " AND ".join(f"all:{t}" for t in terms) if terms else f"all:{clean_query}"

        params = {"search_query": search_expr, "start": 0, "max_results": limit}

        last_status = 0
        last_error = ""

        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.get(self.base_url, params=params)
                    last_status = response.status_code
                    if response.status_code == 200:
                        papers = self._parse_atom_xml(response.text)
                        self.cache.set(
                            cache_key,
                            "arxiv",
                            "/query",
                            {"papers": [p.model_dump() for p in papers]},
                        )
                        return papers, 200, "OK"
                    elif response.status_code == 429:
                        last_error = "Rate limit (HTTP 429)"
                        await asyncio.sleep(3.0 * (2**attempt))
                    else:
                        last_error = f"HTTP {response.status_code}"
                        response.raise_for_status()
                except Exception as e:
                    last_error = f"{type(e).__name__}: {e}"
                    if attempt == self.max_retries - 1:
                        return [], last_status, last_error
                    await asyncio.sleep(2.0 * (2**attempt))

        return [], last_status, last_error

    async def search(self, query: str, limit: int = 20) -> List[Paper]:
        papers, _, _ = await self.search_with_diagnostics(query, limit)
        return papers

    def _parse_atom_xml(self, xml_text: str) -> List[Paper]:
        papers = []
        try:
            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", ARXIV_NAMESPACE):
                id_elem = entry.find("atom:id", ARXIV_NAMESPACE)
                title_elem = entry.find("atom:title", ARXIV_NAMESPACE)
                summary_elem = entry.find("atom:summary", ARXIV_NAMESPACE)
                published_elem = entry.find("atom:published", ARXIV_NAMESPACE)

                raw_id = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
                arxiv_id = raw_id.split("/abs/")[-1]
                title = (
                    " ".join(title_elem.text.split())
                    if title_elem is not None and title_elem.text
                    else "Untitled"
                )
                abstract = (
                    " ".join(summary_elem.text.split())
                    if summary_elem is not None and summary_elem.text
                    else ""
                )

                year = None
                if published_elem is not None and published_elem.text:
                    try:
                        year = int(published_elem.text[:4])
                    except Exception:
                        pass

                authors = []
                for author_elem in entry.findall("atom:author", ARXIV_NAMESPACE):
                    name_elem = author_elem.find("atom:name", ARXIV_NAMESPACE)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text.strip())

                if abstract:
                    papers.append(
                        Paper(
                            paper_id=f"arxiv:{arxiv_id}",
                            title=title,
                            abstract=abstract,
                            url=raw_id,
                            year=year,
                            authors=authors,
                            source_provider="arxiv",
                        )
                    )
        except Exception:
            return []

        return papers
