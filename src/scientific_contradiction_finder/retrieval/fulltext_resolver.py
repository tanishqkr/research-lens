"""Open access full-text resolver and extractor using free legitimate repositories and open web sources."""

import logging
import re
from typing import Optional

import httpx

from scientific_contradiction_finder.retrieval.cache import SQLiteCache

logger = logging.getLogger(__name__)


class OpenAccessFullTextResolver:
    """Fetches and extracts clean text from legitimate open-access URLs and open archives without paid APIs."""

    def __init__(self, cache: Optional[SQLiteCache] = None, timeout_seconds: int = 15):
        self.cache = cache or SQLiteCache()
        self.timeout_seconds = timeout_seconds

    async def resolve_full_text(
        self, oa_url: Optional[str], doi: Optional[str] = None
    ) -> Optional[str]:
        """Attempts to fetch open-access full text if an open URL is available."""
        if not oa_url and not doi:
            return None

        target_url = oa_url
        if not target_url and doi:
            target_url = f"https://doi.org/{doi}"

        cache_key = self.cache.compute_key("oa_fulltext", "resolve", {"url": target_url})
        cached = self.cache.get(cache_key)
        if cached:
            return cached.get("text")

        headers = {
            "User-Agent": "ScientificContradictionFinder/1.0 (mailto:researcher@example.org)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, follow_redirects=True
            ) as client:
                resp = await client.get(target_url, headers=headers)
                if resp.status_code == 200:
                    content_type = resp.headers.get("content-type", "").lower()
                    # If HTML, extract text body
                    if "text/html" in content_type:
                        raw_html = resp.text
                        # Strip scripts, styles, navigation
                        cleaned = re.sub(
                            r"<(script|style|nav|header|footer)[^>]*>.*?</\1>",
                            " ",
                            raw_html,
                            flags=re.DOTALL | re.IGNORECASE,
                        )
                        # Strip HTML tags
                        text = re.sub(r"<[^>]+>", " ", cleaned)
                        text = " ".join(text.split())
                        # Keep relevant portion (up to 12,000 chars)
                        if len(text) > 300:
                            self.cache.set(
                                cache_key, "oa_fulltext", "resolve", {"text": text[:12000]}
                            )
                            return text[:12000]
        except Exception as e:
            logger.debug(f"Full-text resolution failed for {target_url}: {e}")

        return None
