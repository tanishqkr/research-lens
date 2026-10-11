"""End-to-end contradiction analysis orchestrator with optional Groq assistance and rich stage diagnostics."""

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
from scientific_contradiction_finder.retrieval.groq_client import GroqClient
from scientific_contradiction_finder.retrieval.hybrid_reranker import ReciprocalRankFusion
from scientific_contradiction_finder.retrieval.openalex_client import OpenAlexClient
from scientific_contradiction_finder.retrieval.query_parser import PolarityQueryExpander
from scientific_contradiction_finder.schemas import (
    ConditionSlots,
    DirectionLabel,
    Finding,
    FindingPair,
    Paper,
    Prediction,
)


class ContradictionFinderPipeline:
    """Orchestrates query expansion, multi-source retrieval, extraction, and condition-aware reasoning."""

    def __init__(self, config: Optional[Config] = None, use_groq: bool = True):
        self.cfg = config or default_config
        self.use_groq = use_groq
        self.groq_client = GroqClient()
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
        """Expands query, optionally structured by Groq, queries OpenAlex and arXiv, and collects diagnostics."""
        local_expanded = self.expander.expand(query)
        groq_structured = None

        if self.use_groq and self.groq_client.is_available:
            try:
                groq_structured = await self.groq_client.structure_query(query)
            except Exception:
                groq_structured = None

        # Build prioritized search queries: combine Groq keywords (if available) with local keywords
        search_queries = []
        if groq_structured and groq_structured.search_keywords:
            search_queries.extend(groq_structured.search_keywords[:2])
        if local_expanded.keyword_queries:
            for kq in local_expanded.keyword_queries[:2]:
                if kq not in search_queries:
                    search_queries.append(kq)
        for eq in local_expanded.expanded_queries[:2]:
            if eq not in search_queries:
                search_queries.append(eq)

        diagnostics = {
            "query_raw": query,
            "groq_assisted": bool(groq_structured),
            "groq_structured": groq_structured.model_dump() if groq_structured else None,
            "search_queries_dispatched": search_queries[:3],
            "openalex": {"status": None, "results_count": 0, "error": None},
            "arxiv": {"status": None, "results_count": 0, "error": None},
            "raw_total": 0,
            "usable_abstracts": 0,
            "after_dedup": 0,
        }

        oa_tasks = [
            self.openalex_client.search_works_with_diagnostics(q, limit=limit)
            for q in search_queries[:3]
        ]
        ax_tasks = [
            self.arxiv_client.search_with_diagnostics(q, limit=limit) for q in search_queries[:3]
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
        # Filter out records without abstracts or with non-scholarly/trivial length
        usable_papers = [p for p in raw_papers if p.abstract and len(p.abstract.strip()) > 50]
        diagnostics["usable_abstracts"] = len(usable_papers)

        clean_papers = DocumentDeduplicator.deduplicate(usable_papers)
        diagnostics["after_dedup"] = len(clean_papers)

        if not clean_papers:
            return [], diagnostics

        # High precision re-ranking
        ranking_query = (
            search_queries[0]
            if search_queries
            else (local_expanded.keyword_queries[0] if local_expanded.keyword_queries else query)
        )
        bm25 = BM25Retriever(clean_papers)
        bm25_ranked = bm25.retrieve(ranking_query, top_k=limit)

        dense = DenseRetriever(clean_papers)
        dense_ranked = dense.retrieve(ranking_query, top_k=limit)

        rrf = ReciprocalRankFusion(k=self.cfg.retrieval.rrf_k)
        fused = rrf.fuse([bm25_ranked, dense_ranked], top_n=limit)
        return [p for p, _ in fused], diagnostics

    async def search_papers(self, query: str, limit: int = 20) -> List[Paper]:
        papers, _ = await self.search_papers_with_diagnostics(query, limit)
        return papers

    async def extract_findings_from_papers(
        self, papers: List[Paper], use_groq_interpretation: bool = False
    ) -> List[Finding]:
        """Extracts validated empirical finding rationales and condition slots from paper abstracts."""
        findings = []
        for paper in papers:
            evidence_spans = self.finding_extractor.extract_evidence_spans(paper.abstract)
            if not evidence_spans:
                continue

            top_span = evidence_spans[0]

            # Optional secondary Groq interpretation for ambiguous sentences if enabled
            if use_groq_interpretation and self.groq_client.is_available:
                try:
                    interp = await self.groq_client.interpret_candidate_passage(
                        top_span.text, paper.abstract
                    )
                    if interp:
                        if not interp.is_empirical_finding:
                            # Rejected by LLM auditor
                            continue
                        # Use verified LLM condition slots if present
                        direction = (
                            DirectionLabel(interp.direction)
                            if interp.direction in DirectionLabel._value2member_map_
                            else DirectionLabel.UNCLEAR
                        )
                        conditions = ConditionSlots(
                            intervention=interp.intervention
                            or "Unspecified Scientific Intervention",
                            comparator=interp.comparator or "Standard Baseline / Unspecified",
                            outcome_metric=interp.outcome or "Primary Empirical Outcome",
                            dataset_setting=interp.setting or "General Scientific Setting",
                            confidence=interp.confidence or 0.85,
                            is_unknown=bool(not interp.intervention),
                        )
                        finding = Finding(
                            finding_id=f"find_{paper.paper_id}",
                            paper_id=paper.paper_id,
                            paper_title=paper.title,
                            paper_doi=paper.doi,
                            paper_url=paper.url,
                            evidence=top_span,
                            finding_claim=interp.finding_summary or top_span.text,
                            direction=direction,
                            conditions=conditions,
                        )
                        findings.append(finding)
                        continue
                except Exception:
                    pass

            # Standard local rule-based extractor
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

    def compare_findings(
        self, findings: List[Finding], max_comparisons: int = 15
    ) -> List[Tuple[FindingPair, Prediction]]:
        """Selectively compares top findings that share relevant domain concepts, avoiding combinatorial explosion."""
        results = []
        count = 0
        for i in range(len(findings)):
            for j in range(i + 1, len(findings)):
                if count >= max_comparisons:
                    break
                fa = findings[i]
                fb = findings[j]
                pair = FindingPair(
                    pair_id=f"pair_{fa.finding_id}_{fb.finding_id}", finding_a=fa, finding_b=fb
                )
                pred = self.classifier.classify_pair(pair)
                results.append((pair, pred))
                count += 1
        return results

    async def run(
        self, query: str, limit: int = 15, use_groq_interpretation: bool = False
    ) -> Dict[str, Any]:
        """Runs the end-to-end pipeline: search -> extract -> condition tag -> compare."""
        papers, diag = await self.search_papers_with_diagnostics(query, limit=limit)
        findings = await self.extract_findings_from_papers(
            papers, use_groq_interpretation=use_groq_interpretation
        )
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
