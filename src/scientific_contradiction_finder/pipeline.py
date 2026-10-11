"""End-to-end contradiction analysis orchestrator with optional Groq assistance and rich stage diagnostics."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from scientific_contradiction_finder.config import Config, default_config
from scientific_contradiction_finder.extraction.condition_tagger import ConditionTagger
from scientific_contradiction_finder.extraction.finding_extractor import FindingExtractor
from scientific_contradiction_finder.reasoning.classifier import ConditionAwareRelationClassifier
from scientific_contradiction_finder.reasoning.compatibility import ConditionCompatibilityEvaluator
from scientific_contradiction_finder.reasoning.summary_validator import (
    assess_subclaims,
    decompose_query_into_subclaims,
    enforce_summary_consistency,
    verify_disagreement_explanations,
)
from scientific_contradiction_finder.retrieval.arxiv_client import ArxivClient
from scientific_contradiction_finder.retrieval.bm25_retriever import BM25Retriever
from scientific_contradiction_finder.retrieval.deduplicator import DocumentDeduplicator
from scientific_contradiction_finder.retrieval.dense_retriever import DenseRetriever
from scientific_contradiction_finder.retrieval.fulltext_resolver import OpenAccessFullTextResolver
from scientific_contradiction_finder.retrieval.groq_client import GroqClient
from scientific_contradiction_finder.retrieval.hybrid_reranker import ReciprocalRankFusion
from scientific_contradiction_finder.retrieval.openalex_client import OpenAlexClient
from scientific_contradiction_finder.retrieval.query_parser import PolarityQueryExpander
from scientific_contradiction_finder.schemas import (
    ConditionSlots,
    DirectionLabel,
    DisagreementExplanation,
    EvidenceSpan,
    ExecutiveSummary,
    Finding,
    FindingPair,
    Paper,
    PaperTriageClassification,
    PaperTriageRecord,
    Prediction,
    ResearchReport,
    SupportedEvidenceClaim,
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
        self.fulltext_resolver = OpenAccessFullTextResolver()
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
                            intervention=interp.intervention,
                            comparator=interp.comparator,
                            outcome_metric=interp.outcome or "Primary Empirical Outcome",
                            dataset_setting=interp.setting,
                            population=interp.population,
                            effect_size=interp.effect_estimate,
                            study_design=interp.study_design,
                            limitations_or_qualifiers=interp.limitations_or_qualifiers,
                            confidence=interp.confidence or 0.85,
                            is_unknown=bool(not interp.intervention or not interp.outcome),
                        )
                        # Verbatim quote check against paper abstract
                        quote_text = interp.evidence_quote.strip().strip("\"'")
                        if quote_text and quote_text in paper.abstract:
                            evidence = EvidenceSpan(
                                text=quote_text,
                                start_char=paper.abstract.find(quote_text),
                                end_char=paper.abstract.find(quote_text) + len(quote_text),
                                confidence=top_span.confidence,
                            )
                        else:
                            evidence = top_span

                        finding = Finding(
                            finding_id=f"find_{paper.paper_id}",
                            paper_id=paper.paper_id,
                            paper_title=paper.title,
                            paper_doi=paper.doi,
                            paper_url=paper.url,
                            evidence=evidence,
                            finding_claim=interp.finding_summary or evidence.text,
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

    async def analyze_as_research_analyst(
        self, query: str, limit: int = 25, max_deep_dives: int = 8
    ) -> ResearchReport:
        """Executes a multi-stage, iterative scientific research analysis workflow (Stages A through H)."""
        # Stage A & B: Retrieve broad candidate pool with diagnostics
        candidates, diag = await self.search_papers_with_diagnostics(query, limit=limit)
        if not candidates:
            exec_summary = ExecutiveSummary(
                bottom_line="No relevant scholarly papers with accessible abstracts could be retrieved for this query.",
                overall_evidence_assessment="inconclusive",
                strongest_supporting_evidence="None identified.",
                strongest_conflicting_evidence="None identified.",
                why_studies_disagree="Literature coverage was insufficient to identify peer-reviewed findings.",
                confidence_and_limitations="High uncertainty due to absence of retrieved literature.",
                research_coverage="Retrieved 0 papers across OpenAlex and arXiv.",
            )
            return ResearchReport(
                query=query,
                executive_summary=exec_summary,
                candidate_papers_retrieved=0,
                papers_triaged=0,
                papers_deep_dived=0,
                full_texts_accessible=0,
                stopped_reason="No literature retrieved.",
            )

        # Stage C: LLM / heuristic triage of candidate papers
        triage_records: List[PaperTriageRecord] = []
        deep_dive_papers: List[Paper] = []
        for paper in candidates:
            triage_data = await self.groq_client.triage_paper(paper.title, paper.abstract, query)
            cat_str = triage_data.get("category", "DIRECT_EMPIRICAL")
            cat_enum = (
                PaperTriageClassification(cat_str)
                if cat_str in PaperTriageClassification._value2member_map_
                else PaperTriageClassification.DIRECT_EMPIRICAL
            )
            is_selected = bool(triage_data.get("selected_for_deep_dive", True)) and (
                len(deep_dive_papers) < max_deep_dives
            )
            rec = PaperTriageRecord(
                paper_id=paper.paper_id,
                title=paper.title,
                category=cat_enum,
                relevance_rationale=triage_data.get(
                    "relevance_rationale", "Empirical study evaluating target phenomenon."
                ),
                selected_for_deep_dive=is_selected,
            )
            triage_records.append(rec)
            if is_selected:
                deep_dive_papers.append(paper)

        if not deep_dive_papers:
            deep_dive_papers = candidates[:max_deep_dives]

        # Stage D & E: Deep-dive into selected papers (resolve OA full-text where available)
        full_text_count = 0
        for paper in deep_dive_papers:
            if paper.is_oa and paper.oa_url:
                ft = await self.fulltext_resolver.resolve_full_text(paper.oa_url, paper.doi)
                if ft:
                    paper.full_text = ft
                    full_text_count += 1

        # Extract structured findings from deep-dived papers
        findings = await self.extract_findings_from_papers(
            deep_dive_papers, use_groq_interpretation=True
        )

        # Build Structured Evidence Ledger
        evidence_ledger: List[SupportedEvidenceClaim] = []
        for f in findings:
            claim = SupportedEvidenceClaim(
                claim_id=f"claim_{f.finding_id}",
                paper_id=f.paper_id,
                paper_title=f.paper_title,
                doi=f.paper_doi,
                url=f.paper_url,
                exact_source_passage=f.evidence.text,
                study_design=f.conditions.study_design or "Empirical Study",
                population_setting=f.conditions.population or f.conditions.dataset_setting,
                intervention_exposure=f.conditions.intervention,
                comparator=f.conditions.comparator,
                outcome=f.conditions.outcome_metric,
                direction=f.direction,
                effect_size=f.conditions.effect_size,
                conditions_and_moderators=f.conditions.limitations_or_qualifiers,
                full_text_used=bool(
                    next((p.full_text for p in deep_dive_papers if p.paper_id == f.paper_id), None)
                ),
                is_direct=True,
            )
            evidence_ledger.append(claim)

        # Stage F: Cross-paper finding comparisons
        comparisons = self.compare_findings(findings, max_comparisons=10)

        # Stage G: Disagreement & Moderator Analysis
        claims_summary_text = "\n".join(
            f"[{c.paper_id}] {c.paper_title}: I={c.intervention_exposure}, C={c.comparator}, O={c.outcome}, Dir={c.direction.value}, Evidence='{c.exact_source_passage}'"
            for c in evidence_ledger
        )
        disagreements_raw = await self.groq_client.analyze_disagreements(query, claims_summary_text)
        disagreements: List[DisagreementExplanation] = [
            DisagreementExplanation(**d) for d in disagreements_raw if isinstance(d, dict)
        ]
        # Verify and ground disagreement explanations against evidence ledger
        disagreements = verify_disagreement_explanations(disagreements, evidence_ledger)

        # Stage H: Subclaim Decomposition & Independent Evidence Audit
        subclaim_texts = decompose_query_into_subclaims(query)
        subclaim_assessments = assess_subclaims(subclaim_texts, evidence_ledger)

        # Stage I: Executive Summary Synthesis
        cov_stats = {
            "retrieved_total": len(candidates),
            "examined_in_depth": len(deep_dive_papers),
            "findings_in_ledger": len(evidence_ledger),
            "full_texts_accessible": full_text_count,
        }
        disag_summary_text = "\n".join(
            f"- Topic: {d.topic_or_metric}; Studies: {', '.join(d.studies_in_disagreement)}; Status: {d.verification_status.value}; Reason: {d.plausible_moderator_or_reason}"
            for d in disagreements
        )
        exec_raw = await self.groq_client.synthesize_executive_summary(
            query,
            claims_summary_text,
            disag_summary_text or "No substantive disagreement identified.",
            cov_stats,
        )

        if exec_raw:
            exec_summary = ExecutiveSummary(**exec_raw)
        else:
            # Deterministic fallback summary
            pos_count = sum(1 for f in findings if f.direction == DirectionLabel.POSITIVE)
            neg_count = sum(1 for f in findings if f.direction == DirectionLabel.NEGATIVE)
            sum(1 for f in findings if f.direction == DirectionLabel.NO_DIFFERENCE)
            mixed_count = sum(1 for f in findings if f.direction == DirectionLabel.MIXED)

            if pos_count > 0 and neg_count == 0:
                overall = "broadly supports"
                bl = f"The reviewed empirical evidence indicates consistent positive effects for {query.strip('?')}."
            elif neg_count > 0 and pos_count == 0:
                overall = "broadly supports an inverse or reducing relationship"
                bl = "The available empirical studies demonstrate consistent reductions or negative impacts."
            elif (pos_count > 0 and neg_count > 0) or mixed_count > 0:
                overall = "suggests context-dependent effects or qualified disagreement"
                bl = "Empirical findings show divergence depending on specific study conditions, protocols, and outcome measures."
            else:
                overall = "inconclusive"
                bl = "Available empirical evidence is preliminary or lacks standardized comparative outcomes."

            sup_claim = next(
                (
                    c
                    for c in evidence_ledger
                    if c.direction in (DirectionLabel.POSITIVE, DirectionLabel.NEGATIVE)
                ),
                None,
            )
            sup_text = (
                f"{sup_claim.paper_title}: {sup_claim.exact_source_passage}"
                if sup_claim
                else "No definitive primary study."
            )
            opp_claim = next(
                (c for c in evidence_ledger if c.direction == DirectionLabel.NO_DIFFERENCE), None
            )
            opp_text = (
                f"{opp_claim.paper_title}: {opp_claim.exact_source_passage}"
                if opp_claim
                else "No conflicting studies found in sampled literature."
            )

            exec_summary = ExecutiveSummary(
                bottom_line=bl,
                overall_evidence_assessment=overall,
                strongest_supporting_evidence=sup_text,
                strongest_conflicting_evidence=opp_text,
                why_studies_disagree="Differences in measurement protocols, baseline conditions, exposure degrees, and study settings.",
                confidence_and_limitations="Evidence synthesis is based on retrieved open-access scholarly literature. Heuristic comparison scores reflect condition similarity, not formal statistical meta-analysis.",
                research_coverage=f"Retrieved {len(candidates)} candidate papers; examined {len(deep_dive_papers)} in depth ({full_text_count} open-access full texts).",
            )

        # Enforce consistency between evidence ledger, subclaim assessments, and executive summary
        exec_summary = enforce_summary_consistency(
            exec_summary, subclaim_assessments, evidence_ledger
        )

        return ResearchReport(
            query=query,
            executive_summary=exec_summary,
            candidate_papers_retrieved=len(candidates),
            papers_triaged=len(triage_records),
            papers_deep_dived=len(deep_dive_papers),
            full_texts_accessible=full_text_count,
            subclaim_assessments=subclaim_assessments,
            triage_records=triage_records,
            evidence_ledger=evidence_ledger,
            disagreements=disagreements,
            detailed_comparisons=[p for _, p in comparisons],
            investigation_rounds=1,
            stopped_reason="Completed full staged evidence synthesis across accessible open literature.",
        )

    async def run(
        self, query: str, limit: int = 15, use_groq_interpretation: bool = False
    ) -> Dict[str, Any]:
        """Runs the end-to-end pipeline: search -> extract -> condition tag -> compare -> research report."""
        report = await self.analyze_as_research_analyst(query, limit=limit)
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
            "research_report": report,
        }
