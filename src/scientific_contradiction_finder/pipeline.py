"""End-to-end contradiction analysis orchestrator."""

import asyncio
from typing import Any

from scientific_contradiction_finder.config import Config, default_config
from scientific_contradiction_finder.extraction.condition_tagger import ConditionTagger
from scientific_contradiction_finder.extraction.finding_extractor import FindingExtractor
from scientific_contradiction_finder.reasoning.classifier import ConditionAwareRelationClassifier
from scientific_contradiction_finder.reasoning.compatibility import ConditionCompatibilityEvaluator
from scientific_contradiction_finder.retrieval.arxiv_client import ArxivClient
from scientific_contradiction_finder.retrieval.bm25_retriever import BM25Retriever
from scientific_contradiction_finder.retrieval.deduplicator import DocumentDeduplicator
from scientific_contradiction_finder.retrieval.dense_retriever import DenseRetriever
from scientific_contradiction_finder.retrieval.hybrid_reranker import ReciprocalRankFusion
from scientific_contradiction_finder.retrieval.openalex_client import OpenAlexClient
from scientific_contradiction_finder.retrieval.query_parser import PolarityQueryExpander
from scientific_contradiction_finder.schemas import (
    Finding,
    FindingPair,
    Paper,
    Prediction,
)


class ContradictionFinderPipeline:
    """Orchestrates query expansion, multi-source retrieval, extraction, and condition-aware reasoning."""

    def __init__(self, config: Config | None = None):
        self.cfg = config or default_config
        self.expander = PolarityQueryExpander()
        self.openalex_client = OpenAlexClient(
            api_key=self.cfg.retrieval.openalex_api_key,
            mailto=self.cfg.retrieval.openalex_mailto,
            timeout_seconds=self.cfg.retrieval.request_timeout_seconds,
            offline_mode=self.cfg.app.offline_mode,
        )
        self.arxiv_client = ArxivClient(
            timeout_seconds=self.cfg.retrieval.request_timeout_seconds,
            offline_mode=self.cfg.app.offline_mode,
        )
        self.finding_extractor = FindingExtractor()
        self.condition_tagger = ConditionTagger()
        self.compat_evaluator = ConditionCompatibilityEvaluator()
        self.classifier = ConditionAwareRelationClassifier(compat_evaluator=self.compat_evaluator)

    async def search_papers(self, query: str, limit: int = 20) -> list[Paper]:
        """Expands query and retrieves relevant papers across OpenAlex and arXiv."""
        expanded = self.expander.expand(query)
        tasks = []
        for q in expanded.expanded_queries[:3]:
            tasks.append(self.openalex_client.search_works(q, limit=limit // 2))
            tasks.append(self.arxiv_client.search(q, limit=limit // 2))

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        raw_papers: list[Paper] = []
        for res in batch_results:
            if isinstance(res, list):
                raw_papers.extend(res)

        clean_papers = DocumentDeduplicator.deduplicate(raw_papers)
        if not clean_papers:
            return []

        # Rerank with BM25
        bm25 = BM25Retriever(clean_papers)
        bm25_ranked = bm25.retrieve(query, top_k=limit)

        # Dense Reranking
        dense = DenseRetriever(clean_papers)
        dense_ranked = dense.retrieve(query, top_k=limit)

        # RRF Fusion
        rrf = ReciprocalRankFusion(k=self.cfg.retrieval.rrf_k)
        fused = rrf.fuse([bm25_ranked, dense_ranked], top_n=limit)
        return [p for p, _ in fused]

    def extract_findings_from_papers(self, papers: list[Paper]) -> list[Finding]:
        """Extracts evidence rationales, condition slots, and directional outcomes from paper abstracts."""
        findings = []
        for paper in papers:
            evidence_spans = self.finding_extractor.extract_evidence_spans(paper.abstract)
            if not evidence_spans:
                continue

            top_span = evidence_spans[0]
            direction = self.finding_extractor.infer_direction(top_span.text)
            conditions = self.condition_tagger.extract_conditions(top_span.text, paper.abstract)

            finding = Finding(
                finding_id=f"find_{paper.paper_id}",
                paper_id=paper.paper_id,
                paper_title=paper.title,
                paper_doi=paper.doi,
                paper_url=paper.url,
                evidence=top_span,
                finding_claim=top_span.text,
                direction=direction,
                conditions=conditions,
            )
            findings.append(finding)
        return findings

    def compare_findings(self, findings: list[Finding]) -> list[tuple[FindingPair, Prediction]]:
        """Compares all pairs of extracted findings using condition-aware reasoning."""
        results = []
        for i in range(len(findings)):
            for j in range(i + 1, len(findings)):
                fa = findings[i]
                fb = findings[j]
                pair = FindingPair(
                    pair_id=f"pair_{fa.finding_id}_{fb.finding_id}", finding_a=fa, finding_b=fb
                )
                pred = self.classifier.classify_pair(pair)
                results.append((pair, pred))
        return results

    async def run(self, query: str, limit: int = 15) -> dict[str, Any]:
        """Runs the end-to-end pipeline: search -> extract -> condition tag -> compare."""
        papers = await self.search_papers(query, limit=limit)
        findings = self.extract_findings_from_papers(papers)
        comparisons = self.compare_findings(findings)

        return {
            "query": query,
            "papers_found": len(papers),
            "findings_extracted": len(findings),
            "comparisons": comparisons,
            "papers": papers,
            "findings": findings,
        }
