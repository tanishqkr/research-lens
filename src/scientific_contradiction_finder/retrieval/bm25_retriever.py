"""BM25 Okapi lexical retriever over paper abstracts."""

import math
import re
from collections import Counter

from scientific_contradiction_finder.schemas import Paper


def tokenize_corpus_text(text: str) -> list[str]:
    """Tokenize and normalize text for lexical ranking."""
    text = text.lower()
    return re.findall(r"\b[a-z0-9_\-]+\b", text)


class BM25Retriever:
    """Robust lexical retriever with Robertson-Spärck Jones BM25 implementation.

    Uses standard positive-smoothed IDF to prevent zero or negative weights on small corpora:
    IDF(q) = ln(1 + (N - n + 0.5) / (n + 0.5))
    """

    def __init__(self, papers: list[Paper], k1: float = 1.5, b: float = 0.75):
        self.papers = papers
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = [
            tokenize_corpus_text(f"{p.title} {p.abstract}") for p in papers
        ]
        self.doc_lens = [len(dt) for dt in self.doc_tokens]
        self.avg_doc_len = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 1.0
        self.n_docs = len(self.doc_tokens)

        # Compute document frequencies
        self.df = Counter()
        for dt in self.doc_tokens:
            for term in set(dt):
                self.df[term] += 1

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        # Smoothed positive IDF
        return math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5))

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[Paper, float]]:
        if not self.papers:
            return []

        query_tokens = tokenize_corpus_text(query)
        scores = [0.0] * self.n_docs

        for term in query_tokens:
            idf = self._idf(term)
            if idf <= 0.0:
                continue

            for i, (dt, dlen) in enumerate(zip(self.doc_tokens, self.doc_lens)):
                tf = dt.count(term)
                if tf > 0:
                    denom = tf + self.k1 * (
                        1.0 - self.b + self.b * (dlen / (self.avg_doc_len or 1.0))
                    )
                    scores[i] += idf * (tf * (self.k1 + 1.0)) / denom

        scored_papers = [(self.papers[i], float(scores[i])) for i in range(self.n_docs)]
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        return scored_papers[:top_k]
