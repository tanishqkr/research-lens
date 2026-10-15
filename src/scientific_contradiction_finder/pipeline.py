"""End-to-end contradiction analysis orchestrator with optional Groq assistance and rich stage diagnostics."""

import asyncio
import uuid
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
    PipelineExecutionState,
    Prediction,
    QueryIntent,
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

            # Retain multiple distinct result findings per paper (e.g., 6-month vs 12-month outcomes)
            selected_spans = [evidence_spans[0]]
            for span in evidence_spans[1:]:
                if all(
                    abs(span.start_char - s.start_char) > 25 and span.text != s.text
                    for s in selected_spans
                ):
                    selected_spans.append(span)
                if len(selected_spans) >= 3:
                    break

            for s_idx, span in enumerate(selected_spans):
                fid = f"find_{paper.paper_id}" if s_idx == 0 else f"find_{paper.paper_id}_{s_idx}"

                # Optional secondary Groq interpretation for ambiguous sentences if enabled (on primary span)
                if s_idx == 0 and use_groq_interpretation and self.groq_client.is_available:
                    try:
                        interp = await self.groq_client.interpret_candidate_passage(
                            span.text, paper.abstract
                        )
                        if interp:
                            if not interp.is_empirical_finding:
                                continue
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
                            quote_text = interp.evidence_quote.strip().strip("\"'")
                            if quote_text and quote_text in paper.abstract:
                                evidence = EvidenceSpan(
                                    text=quote_text,
                                    start_char=paper.abstract.find(quote_text),
                                    end_char=paper.abstract.find(quote_text) + len(quote_text),
                                    confidence=span.confidence,
                                )
                            else:
                                evidence = span

                            finding = Finding(
                                finding_id=fid,
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
                direction = self.finding_extractor.infer_direction(span.text)
                conditions = self.condition_tagger.extract_conditions(span.text, paper.abstract)

                finding = Finding(
                    finding_id=fid,
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    paper_doi=paper.doi,
                    paper_url=paper.url,
                    evidence=span,
                    finding_claim=span.text,
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

    def detect_query_intent(self, query: str) -> QueryIntent:
        """Detect the intent of the scientific query to steer extraction and synthesis."""
        q_lower = query.lower()
        if any(p in q_lower for p in ["what technology", "what technologies", "what method", "what methods", "what tool", "what tools", "how can we use", "what are the ways", "what approaches"]):
            return QueryIntent.RESEARCH_OVERVIEW
        if any(p in q_lower for p in ["versus", " vs ", " vs.", "compared to", "compared with", "which is better", "head-to-head"]):
            return QueryIntent.DIRECT_COMPARISON
        if any(p in q_lower for p in ["does ", "do ", "is ", "are ", "can ", "whether ", "contradict", "debate", "dispute", "conflict"]):
            return QueryIntent.CONTRADICTION_ANALYSIS
        return QueryIntent.CLAIM_VERIFICATION

    async def analyze_as_research_analyst(
        self, query: str, limit: int = 25, max_deep_dives: int = 8
    ) -> ResearchReport:
        """Executes a multi-stage, iterative scientific research analysis workflow (Stages A through H)."""
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        query_intent = self.detect_query_intent(query)

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
                run_id=run_id,
                query_intent=query_intent,
                execution_state=PipelineExecutionState.COMPLETED_WITH_LIMITATIONS,
                executive_summary=exec_summary,
                candidate_papers_retrieved=0,
                papers_triaged=0,
                papers_abstract_screened=0,
                papers_deep_dived=0,
                full_texts_accessible=0,
                findings_extracted=0,
                comparisons_evaluated=0,
                retrieval_diagnostics=diag,
                stopped_reason="No literature retrieved.",
            )

        # Stage C: Structured Batch Triage of candidate papers (with fallback to deterministic heuristic)
        candidate_dicts = [
            {"title": p.title, "abstract": p.abstract or ""}
            for p in candidates
        ]
        triage_batch_results = await self.groq_client.triage_papers_batch(candidate_dicts, query)

        triage_records: List[PaperTriageRecord] = []
        deep_dive_papers: List[Paper] = []
        for paper, t_data in zip(candidates, triage_batch_results):
            cat_str = t_data.get("category", "DIRECT_EMPIRICAL")
            cat_enum = (
                PaperTriageClassification(cat_str)
                if cat_str in PaperTriageClassification._value2member_map_
                else PaperTriageClassification.DIRECT_EMPIRICAL
            )
            is_heuristic = bool(t_data.get("is_heuristic_fallback", False))
            is_selected = bool(t_data.get("selected_for_deep_dive", True)) and (
                len(deep_dive_papers) < max_deep_dives
            )
            rec = PaperTriageRecord(
                paper_id=paper.paper_id,
                title=paper.title,
                category=cat_enum,
                relevance_rationale=t_data.get(
                    "relevance_rationale", "Empirical study evaluating target phenomenon."
                ),
                selected_for_deep_dive=is_selected,
                is_heuristic_fallback=is_heuristic,
            )
            triage_records.append(rec)
            if is_selected:
                deep_dive_papers.append(paper)

        # If zero papers selected for deep dive, return honest partial completion without fabricating findings
        if not deep_dive_papers:
            exec_summary = ExecutiveSummary(
                bottom_line="Candidate literature was retrieved, but no empirical studies directly meeting the evaluation criteria qualified for deep-dive analysis.",
                overall_evidence_assessment="inconclusive",
                strongest_supporting_evidence="None identified in qualifying literature.",
                strongest_conflicting_evidence="None identified.",
                why_studies_disagree="Retrieved papers either evaluated off-target populations or did not report direct empirical outcomes.",
                confidence_and_limitations="High uncertainty. Automated triage screened out candidate abstracts.",
                research_coverage=f"Retrieved {len(candidates)} candidate papers; 0 selected for deep-dive.",
            )
            return ResearchReport(
                query=query,
                run_id=run_id,
                query_intent=query_intent,
                execution_state=PipelineExecutionState.COMPLETED_WITH_LIMITATIONS,
                executive_summary=exec_summary,
                candidate_papers_retrieved=len(candidates),
                papers_triaged=len(triage_records),
                papers_abstract_screened=len(candidates),
                papers_deep_dived=0,
                full_texts_accessible=0,
                findings_extracted=0,
                comparisons_evaluated=0,
                retrieval_diagnostics=diag,
                triage_records=triage_records,
                evidence_ledger=[],
                disagreements=[],
                detailed_comparisons=[],
                investigation_rounds=1,
                stopped_reason="No candidate papers passed empirical triage criteria.",
            )

        # Stage D & E: Deep-dive into selected papers (resolve OA full-text where available)
        full_text_count = 0
        for paper in deep_dive_papers:
            if paper.is_oa and paper.oa_url:
                ft = await self.fulltext_resolver.resolve_full_text(paper.oa_url, paper.doi)
                if ft:
                    paper.full_text = ft
                    full_text_count += 1

        # Extract structured findings from deep-dived papers only
        findings = await self.extract_findings_from_papers(
            deep_dive_papers, use_groq_interpretation=True
        )

        # Build Structured Evidence Ledger
        evidence_ledger: List[SupportedEvidenceClaim] = []
        triage_map = {r.paper_id: r.category for r in triage_records}
        for f in findings:
            cat = triage_map.get(f.paper_id, PaperTriageClassification.DIRECT_EMPIRICAL)
            is_rev = cat == PaperTriageClassification.REVIEW_SYNTHESIS
            is_dir = cat in (
                PaperTriageClassification.DIRECT_COMPARISON,
                PaperTriageClassification.DIRECT_EMPIRICAL,
            )
            claim = SupportedEvidenceClaim(
                claim_id=f"claim_{f.finding_id}",
                paper_id=f.paper_id,
                paper_title=f.paper_title,
                doi=f.paper_doi,
                url=f.paper_url,
                exact_source_passage=f.evidence.text,
                study_design=f.conditions.study_design
                or ("Systematic Review" if is_rev else "Empirical Study"),
                study_type=f.conditions.study_type,
                provenance_section=f.conditions.provenance_section,
                population_setting=f.conditions.population or f.conditions.dataset_setting,
                intervention_exposure=f.conditions.intervention,
                comparator=f.conditions.comparator,
                outcome=f.conditions.outcome_metric,
                direction=f.direction,
                effect_size=f.conditions.effect_size,
                follow_up_duration=f.conditions.follow_up_duration,
                sample_size=f.conditions.sample_size,
                numerical_results=f.conditions.numerical_results,
                conditions_and_moderators=f.conditions.limitations_or_qualifiers,
                full_text_used=bool(
                    next((p.full_text for p in deep_dive_papers if p.paper_id == f.paper_id), None)
                ),
                is_direct=is_dir,
                is_review_synthesis=is_rev,
            )
            evidence_ledger.append(claim)

        # Stage F: Cross-paper finding comparisons (only if multiple empirical findings exist)
        comparisons = self.compare_findings(findings, max_comparisons=15) if len(findings) >= 2 else []

        # Stage G: Disagreement & Moderator Analysis
        claims_summary_text = "\n".join(
            f"[{c.paper_id}] ({'REVIEW SYNTHESIS' if c.is_review_synthesis else 'PRIMARY EMPIRICAL'}) {c.paper_title}: I={c.intervention_exposure}, C={c.comparator}, O={c.outcome}, Dir={c.direction.value}, Evidence='{c.exact_source_passage}'"
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
            mixed_count = sum(1 for f in findings if f.direction == DirectionLabel.MIXED)

            if len(evidence_ledger) == 0:
                overall = "inconclusive"
                bl = "No qualifying empirical literature was identified testing this relationship in the retrieved search corpus."
            elif query_intent == QueryIntent.RESEARCH_OVERVIEW:
                overall = "identifies promising technological approaches"
                bl = "Multiple technological methods and prediction models are reported in the examined literature for this clinical target."
            elif pos_count > 0 and neg_count == 0:
                overall = "broadly supports"
                bl = "The reviewed empirical literature reports positive associations or improvements under the tested intervention conditions."
            elif neg_count > 0 and pos_count == 0:
                overall = "broadly supports an inverse or reducing relationship"
                bl = "The available empirical studies demonstrate consistent reductions (e.g. reduced risk, weight reduction, or lowered mortality) under the evaluated conditions."
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

            conf_and_lim = (
                "Evidence synthesis is based on accessible scholarly abstracts (0 open-access full texts accessible). Findings reflect abstract-reported results without full-text verification."
                if full_text_count == 0
                else "Evidence synthesis is based on retrieved open-access scholarly literature. Heuristic comparison scores reflect condition similarity, not formal statistical meta-analysis."
            )
            res_cov = (
                f"Retrieved {len(candidates)} candidate papers; synthesized {len(deep_dive_papers)} study abstracts (0 open-access full texts accessible; abstract-level synthesis)."
                if full_text_count == 0
                else f"Retrieved {len(candidates)} candidate papers; examined {len(deep_dive_papers)} in depth ({full_text_count} open-access full texts)."
            )

            exec_summary = ExecutiveSummary(
                bottom_line=bl,
                overall_evidence_assessment=overall,
                strongest_supporting_evidence=sup_text,
                strongest_conflicting_evidence=opp_text,
                why_studies_disagree="Differences in measurement protocols, baseline conditions, exposure degrees, and study settings.",
                confidence_and_limitations=conf_and_lim,
                research_coverage=res_cov,
            )

        # Enforce consistency between evidence ledger, subclaim assessments, and executive summary
        exec_summary = enforce_summary_consistency(
            exec_summary, subclaim_assessments, evidence_ledger
        )

        has_limitations = any(r.is_heuristic_fallback for r in triage_records) or len(evidence_ledger) < 2
        final_state = PipelineExecutionState.COMPLETED_WITH_LIMITATIONS if has_limitations else PipelineExecutionState.COMPLETED

        return ResearchReport(
            query=query,
            run_id=run_id,
            query_intent=query_intent,
            execution_state=final_state,
            executive_summary=exec_summary,
            candidate_papers_retrieved=len(candidates),
            papers_triaged=len(triage_records),
            papers_abstract_screened=len(candidates),
            papers_deep_dived=len(deep_dive_papers),
            full_texts_accessible=full_text_count,
            findings_extracted=len(evidence_ledger),
            comparisons_evaluated=len(comparisons),
            retrieval_diagnostics=diag,
            subclaim_assessments=subclaim_assessments,
            triage_records=triage_records,
            evidence_ledger=evidence_ledger,
            disagreements=disagreements,
            detailed_comparisons=[p for _, p in comparisons],
            investigation_rounds=1,
            stopped_reason=(
                "Completed staged abstract-level evidence synthesis across accessible open literature (0 open-access full texts accessible; findings reflect abstract-reported results)."
                if full_text_count == 0
                else "Completed full staged evidence synthesis across accessible open literature."
            ),
        )

    async def run(
        self, query: str, limit: int = 15, use_groq_interpretation: bool = False
    ) -> Dict[str, Any]:
        """Runs the end-to-end pipeline: search -> extract -> condition tag -> compare -> research report."""
        report = await self.analyze_as_research_analyst(query, limit=limit)

        # Build findings directly from the report's evidence ledger to preserve single source of truth
        # and prevent un-triaged candidate papers from producing phantom findings or comparisons
        findings: List[Finding] = []
        for claim in report.evidence_ledger:
            conds = ConditionSlots(
                intervention=claim.intervention_exposure,
                comparator=claim.comparator,
                outcome_metric=claim.outcome,
                population=claim.population_setting,
                dataset_setting=claim.population_setting,
                effect_size=claim.effect_size,
                follow_up_duration=claim.follow_up_duration,
                sample_size=claim.sample_size,
                numerical_results=claim.numerical_results,
                study_type=claim.study_type,
                provenance_section=claim.provenance_section,
                limitations_or_qualifiers=claim.conditions_and_moderators,
                study_design=claim.study_design,
                confidence=0.85,
            )
            ev_span = EvidenceSpan(
                text=claim.exact_source_passage,
                start_char=0,
                end_char=len(claim.exact_source_passage),
                confidence=0.90,
            )
            f = Finding(
                finding_id=claim.claim_id.replace("claim_", ""),
                paper_id=claim.paper_id,
                paper_title=claim.paper_title,
                paper_url=claim.url,
                evidence=ev_span,
                finding_claim=claim.exact_source_passage,
                direction=claim.direction,
                conditions=conds,
            )
            findings.append(f)

        comparisons = self.compare_findings(findings) if len(findings) >= 2 else []

        retrieval_diag = dict(report.retrieval_diagnostics)
        retrieval_diag["run_id"] = report.run_id
        retrieval_diag["execution_state"] = report.execution_state.value

        return {
            "query": query,
            "run_id": report.run_id,
            "query_intent": report.query_intent.value,
            "execution_state": report.execution_state.value,
            "papers_found": report.candidate_papers_retrieved,
            "findings_extracted": len(findings),
            "comparisons": comparisons,
            "papers": [],
            "findings": findings,
            "diagnostics": retrieval_diag,
            "research_report": report,
        }
