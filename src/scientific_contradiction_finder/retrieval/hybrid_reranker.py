"""Reciprocal Rank Fusion (RRF) combining lexical BM25 and dense bi-encoder results."""

from scientific_contradiction_finder.schemas import Paper


class ReciprocalRankFusion:
    """Combines multiple ranked result lists using Reciprocal Rank Fusion."""

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self, ranked_lists: list[list[tuple[Paper, float]]], top_n: int = 20
    ) -> list[tuple[Paper, float]]:
        """Fuses scored paper lists using RRF score = sum(1 / (k + rank))."""
        rrf_scores: dict[str, float] = {}
        paper_lookup: dict[str, Paper] = {}

        for ranked_list in ranked_lists:
            for rank, (paper, _) in enumerate(ranked_list, start=1):
                pid = paper.paper_id
                paper_lookup[pid] = paper
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (self.k + rank))

        sorted_pids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)
        return [(paper_lookup[pid], rrf_scores[pid]) for pid in sorted_pids[:top_n]]
