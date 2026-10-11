"""End-to-end contradiction analysis orchestrator with rich stage diagnostics."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

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
from scientific_contradiction_finder.schemas import Finding, FindingPair, Paper, Prediction


class ContradictionFinderPipeline:
    """Orchestrates query expansion, multi-source retrieval, extraction, and condition-aware reasoning."""

    def __init__(self, config: Optional[Config] = None):
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

    async def search_papers_with_diagnostics(
        self, query: str, limit: int = 20
    ) -> Tuple[List[Paper], Dict[str, Any]]:
        """Expands query, queries OpenAlex and arXiv, and collects detailed stage diagnostics."""
        expanded = self.expander.expand(query)
        diagnostics = {
            "query_raw": query,
            "expanded_queries": expanded.expanded_queries,
            "keyword_queries": expanded.keyword_queries,
            "openalex": {"status": None, "results_count": 0, "error": None},
            "arxiv": {"status": None, "results_count": 0, "error": None},
            "raw_total": 0,
            "usable_abstracts": 0,
            "after_dedup": 0,
        }

        # Build tasks: use keyword query and first polarity variant
        search_queries = expanded.expanded_queries[:3]
        oa_tasks = [
            self.openalex_client.search_works_with_diagnostics(q, limit=limit)
            for q in search_queries
        ]
        ax_tasks = [
            self.arxiv_client.search_with_diagnostics(q, limit=limit) for q in search_queries
        ]

        oa_results = await asyncio.gather(*oa_tasks, return_exceptions=True)
        ax_results = await asyncio.gather(*ax_tasks, return_exceptions=True)

        raw_papers: List[Paper] = []

        # Aggregate OpenAlex
        for res in oa_results:
            if isinstance(res, tuple):
                papers, status, err = res
                diagnostics["openalex"]["status"] = status
                diagnostics["openalex"]["error"] = err
                diagnostics["openalex"]["results_count"] += len(papers)
                raw_papers.extend(papers)
            elif isinstance(res, Exception):
                diagnostics["openalex"]["error"] = str(res)

        # Aggregate arXiv
        for res in ax_results:
            if isinstance(res, tuple):
                papers, status, err = res
                diagnostics["arxiv"]["status"] = status
                diagnostics["arxiv"]["error"] = err
                diagnostics["arxiv"]["results_count"] += len(papers)
                raw_papers.extend(papers)
            elif isinstance(res, Exception):
                diagnostics["arxiv"]["error"] = str(res)

        diagnostics["raw_total"] = len(raw_papers)
        diagnostics["usable_abstracts"] = sum(
            1 for p in raw_papers if p.abstract and len(p.abstract.strip()) > 30
        )

        clean_papers = DocumentDeduplicator.deduplicate(raw_papers)
        diagnostics["after_dedup"] = len(clean_papers)

        if not clean_papers:
            return [], diagnostics

        # Rerank with BM25
        bm25 = BM25Retriever(clean_papers)
        # Use keywords query for lexical ranking if available
        ranking_query = expanded.keyword_queries[0] if expanded.keyword_queries else query
        bm25_ranked = bm25.retrieve(ranking_query, top_k=limit)

        # Dense Reranking
        dense = DenseRetriever(clean_papers)
        dense_ranked = dense.retrieve(ranking_query, top_k=limit)

        # RRF Fusion
        rrf = ReciprocalRankFusion(k=self.cfg.retrieval.rrf_k)
        fused = rrf.fuse([bm25_ranked, dense_ranked], top_n=limit)
        return [p for p, _ in fused], diagnostics

    async def search_papers(self, query: str, limit: int = 20) -> List[Paper]:
        papers, _ = await self.search_papers_with_diagnostics(query, limit)
        return papers

    def extract_findings_from_papers(self, papers: List[Paper]) -> List[Finding]:
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

    def compare_findings(self, findings: List[Finding]) -> List[Tuple[FindingPair, Prediction]]:
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

    async def run(self, query: str, limit: int = 15) -> Dict[str, Any]:
        """Runs the end-to-end pipeline: search -> extract -> condition tag -> compare."""
        papers, diag = await self.search_papers_with_diagnostics(query, limit=limit)
        findings = self.extract_findings_from_papers(papers)
        comparisons = self.compare_findings(findings)

        return {
            "query": query,
            "papers_found": len(papers),
            "findings_extracted": len(findings),
            "comparisons": comparisons,
            "papers": papers,
            "findings": findings,
            "diagnostics": diag,
        }
