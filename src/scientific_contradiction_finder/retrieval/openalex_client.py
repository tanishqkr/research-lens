"""OpenAlex API client with inverted index decoding, backoff, and caching."""

import asyncio
import time
import httpx
from typing import List, Optional, Dict, Any
from scientific_contradiction_finder.schemas import Paper
from scientific_contradiction_finder.retrieval.cache import SQLiteCache


def decode_openalex_inverted_index(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Reconstructs full text abstract from OpenAlex abstract_inverted_index format."""
    if not inverted_index:
        return ""
    pos_map: Dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            pos_map[pos] = word

    sorted_indices = sorted(pos_map.keys())
    return " ".join(pos_map[i] for i in sorted_indices)


class OpenAlexClient:
    """Async client for OpenAlex works endpoint with rate-limiting, retries, and offline caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        mailto: Optional[str] = None,
        cache: Optional[SQLiteCache] = None,
        timeout_seconds: int = 15,
        max_retries: int = 3,
        offline_mode: bool = False
    ):
        self.api_key = api_key
        self.mailto = mailto
        self.cache = cache or SQLiteCache()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.offline_mode = offline_mode
        self.base_url = "https://api.openalex.org"

    async def search_works(self, query: str, limit: int = 25) -> List[Paper]:
        """Search OpenAlex works with query text."""
        params = {
            "search": query,
            "per_page": limit,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if self.mailto:
            params["mailto"] = self.mailto

        cache_key = self.cache.compute_key("openalex", "/works", {"search": query, "limit": limit})
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return self._parse_works_response(cached_data)

        if self.offline_mode:
            return []

        # Make HTTP request with bounded backoff
        headers = {"User-Agent": f"ScientificContradictionFinder/0.1.0 (mailto:{self.mailto or 'user@example.org'})"}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.get(
                        f"{self.base_url}/works",
                        params=params,
                        headers=headers
                    )
                    if response.status_code == 200:
                        data = response.json()
                        self.cache.set(cache_key, "openalex", "/works", data)
                        return self._parse_works_response(data)
                    elif response.status_code == 429:
                        wait_time = 1.5 * (2 ** attempt)
                        await asyncio.sleep(wait_time)
                    else:
                        response.raise_for_status()
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        # Graceful failure fallback
                        return []
                    await asyncio.sleep(1.0 * (2 ** attempt))

        return []

    def _parse_works_response(self, data: Dict[str, Any]) -> List[Paper]:
        results = data.get("results", [])
        papers = []
        for r in results:
            work_id = r.get("id", "").split("/")[-1]
            title = r.get("title") or "Untitled Document"
            abstract = decode_openalex_inverted_index(r.get("abstract_inverted_index"))
            doi = r.get("doi")
            url = r.get("doi") or r.get("id")
            year = r.get("publication_year")

            authors = []
            for authorship in r.get("authorships", []):
                author_name = authorship.get("author", {}).get("display_name")
                if author_name:
                    authors.append(author_name)

            if abstract:  # We require abstracts for evidence processing
                papers.append(Paper(
                    paper_id=work_id,
                    title=title,
                    abstract=abstract,
                    doi=doi,
                    url=url,
                    year=year,
                    authors=authors,
                    source_provider="openalex"
                ))
        return papers
