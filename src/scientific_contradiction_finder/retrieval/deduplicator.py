"""Deduplication and canonicalization across OpenAlex, arXiv, and local corpora."""

import re
import hashlib
from typing import List
from scientific_contradiction_finder.schemas import Paper


def normalize_title(title: str) -> str:
    """Normalizes titles by stripping punctuation, extra spaces, and lowercase."""
    cleaned = re.sub(r"[^\w\s]", "", title.lower())
    return " ".join(cleaned.split())


class DocumentDeduplicator:
    """Deduplicates retrieved documents using DOI matching followed by normalized title hashing."""

    @staticmethod
    def deduplicate(papers: List[Paper]) -> List[Paper]:
        seen_dois = set()
        seen_titles = set()
        unique_papers = []

        for p in papers:
            # 1. DOI check
            if p.doi:
                norm_doi = p.doi.lower().strip()
                if norm_doi in seen_dois:
                    continue
                seen_dois.add(norm_doi)

            # 2. Title hash check
            norm_t = normalize_title(p.title)
            if not norm_t:
                continue
            title_hash = hashlib.sha256(norm_t.encode("utf-8")).hexdigest()
            if title_hash in seen_titles:
                continue
            seen_titles.add(title_hash)

            unique_papers.append(p)

        return unique_papers
