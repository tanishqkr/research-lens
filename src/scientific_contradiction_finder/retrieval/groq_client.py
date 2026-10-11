"""Groq API client for scientific query structuring and optional candidate sentence interpretation."""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from scientific_contradiction_finder.retrieval.cache import SQLiteCache

load_dotenv()
logger = logging.getLogger(__name__)


class StructuredScientificQuery(BaseModel):
    core_concepts: List[str] = Field(
        default_factory=list,
        description="Primary domain terms (e.g. ['lithium-ion battery', 'high temperature'])",
    )
    intervention_exposure: Optional[str] = Field(
        None, description="Target intervention, exposure, or method"
    )
    comparator_baseline: Optional[str] = Field(
        None, description="Comparator or baseline condition if explicitly inferable"
    )
    outcome_metric: Optional[str] = Field(
        None, description="Outcome variable or metric under study"
    )
    population_setting: Optional[str] = Field(
        None, description="Target setting, benchmark, or population"
    )
    search_keywords: List[str] = Field(
        default_factory=list,
        description="Curated high-precision keyword combinations for API queries",
    )
    expanded_phrases: List[str] = Field(
        default_factory=list, description="Synonym / facet query variants"
    )


class StructuredFindingInterpretation(BaseModel):
    is_empirical_finding: bool = Field(
        ...,
        description="True only if passage reports an observation, measured result, effect, or quantitative finding",
    )
    evidence_quote: Optional[str] = Field(
        default="",
        description="Exact verbatim substring from the source passage supporting the finding",
    )
    finding_summary: Optional[str] = Field(
        default="", description="Concise statement of the finding"
    )
    direction: Optional[str] = Field(
        default="UNCLEAR",
        description="POSITIVE (increases/benefits outcome), NEGATIVE (decreases/worsens outcome), NO_DIFFERENCE, MIXED, or UNCLEAR",
    )
    intervention: Optional[str] = Field(
        None,
        description="Intervention, method, exposure or manipulated condition, null if unspecified",
    )
    comparator: Optional[str] = Field(
        None, description="Control group, baseline condition, or comparator, null if unspecified"
    )
    outcome: Optional[str] = Field(
        None, description="Measured, assessed, or dependent variable, null if unspecified"
    )
    population: Optional[str] = Field(
        None, description="Cohort, subjects, species, or demographic group, null if unspecified"
    )
    setting: Optional[str] = Field(
        None,
        description="Experimental environment, city, climate, benchmark, or context, null if unspecified",
    )
    effect_estimate: Optional[str] = Field(
        None,
        description="Numerical effect size, percentage, or statistics if reported, null otherwise",
    )
    study_design: Optional[str] = Field(
        None,
        description="Study design (RCT, observational, simulation, benchmark), null if unstated",
    )
    limitations_or_qualifiers: Optional[str] = Field(
        None, description="Reported threshold, duration, dose, or qualification, null if unstated"
    )
    confidence: Optional[float] = Field(
        default=0.8, description="Extraction confidence between 0.0 and 1.0"
    )
    rejection_reason: Optional[str] = Field(None, description="Reason if rejected as non-empirical")


class GroqClient:
    """Client for Groq LLM API with strict schema validation, offline fallback, and persistent caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen/qwen3.8-27b",
        cache: Optional[SQLiteCache] = None,
        timeout_seconds: int = 15,
        max_retries: int = 2,
    ):
        self.api_key = api_key or os.getenv("GROQ_API")
        self.model = model
        self.cache = cache or SQLiteCache()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    @property
    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.startswith("gsk_"))

    async def structure_query(self, query: str) -> Optional[StructuredScientificQuery]:
        """Structures a natural language scientific question into validated search terms."""
        if not self.is_available:
            return None

        cache_key = self.cache.compute_key(
            "groq", "structure_query", {"query": query, "model": self.model}
        )
        cached = self.cache.get(cache_key)
        if cached:
            try:
                return StructuredScientificQuery(**cached)
            except Exception:
                pass

        system_prompt = (
            "You are a scientific literature retrieval expert. Convert the user's scientific question into a structured JSON query object.\n"
            "Rules:\n"
            "1. Extract ONLY terms explicitly present or clearly implied by the question.\n"
            "2. Do NOT invent paper titles, authors, citations, or synthetic empirical results.\n"
            "3. search_keywords must contain 2-4 concise search phrases (2 to 5 words each) optimized for OpenAlex and arXiv title/abstract search.\n"
            "4. Return strict JSON matching this schema:\n"
            "{\n"
            '  "core_concepts": ["concept1", "concept2"],\n'
            '  "intervention_exposure": "string or null",\n'
            '  "comparator_baseline": "string or null",\n'
            '  "outcome_metric": "string or null",\n'
            '  "population_setting": "string or null",\n'
            '  "search_keywords": ["phrase 1", "phrase 2"],\n'
            '  "expanded_phrases": ["phrase 1"]\n'
            "}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Scientific question: {query}"},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for attempt in range(self.max_retries):
                try:
                    res = await client.post(self.base_url, headers=headers, json=payload)
                    if res.status_code == 200:
                        content = res.json()["choices"][0]["message"]["content"]
                        data = json.loads(content)
                        structured = StructuredScientificQuery(**data)
                        self.cache.set(
                            cache_key, "groq", "structure_query", structured.model_dump()
                        )
                        return structured
                    elif res.status_code == 429:
                        break  # Rate limited, fallback immediately
                except Exception as e:
                    logger.warning(f"Groq query structuring attempt {attempt + 1} failed: {e}")

        return None

    async def interpret_candidate_passage(
        self, passage: str, context: Optional[str] = None
    ) -> Optional[StructuredFindingInterpretation]:
        """Interprets a candidate empirical sentence with strict verbatim quote verification."""
        if not self.is_available or not passage:
            return None

        cache_key = self.cache.compute_key(
            "groq", "interpret_passage", {"passage": passage, "model": self.model}
        )
        cached = self.cache.get(cache_key)
        if cached:
            try:
                item = StructuredFindingInterpretation(**cached)
                # Verify substring against passage
                if (
                    item.is_empirical_finding
                    and item.evidence_quote.strip().strip("\"'") not in passage
                ):
                    return None
                return item
            except Exception:
                pass

        system_prompt = (
            "You are an empirical evidence auditor for scientific literature. Analyze the supplied passage.\n"
            "Determine if it reports an actual EMPIRICAL FINDING (an observation, measured result, effect, or quantitative/qualitative finding).\n"
            "Reject generic background statements ('chronic sleep restriction is common'), methodology descriptions ('participants were administered the interview'),\n"
            "review intentions ('this paper aims to'), speculation, recommendations, or generic promise statements ('demonstrated great promise').\n"
            "Rules:\n"
            "1. evidence_quote MUST be an EXACT, VERBATIM substring of the supplied passage.\n"
            "2. If not an empirical finding, set is_empirical_finding=false, set rejection_reason, and leave slots null.\n"
            "3. Never substitute generic phrases such as 'A Result', 'The Action', 'The Researcher', 'The Day', 'Our Findings', or 'The Effect' for scientific variables.\n"
            "4. direction must be one of: 'POSITIVE' (intervention increases/benefits outcome), 'NEGATIVE' (intervention decreases/worsens outcome or causes impairment), 'NO_DIFFERENCE' (no significant effect), 'MIXED' (opposing effects across measures), or 'UNCLEAR'.\n"
            "5. Return strict JSON with keys: is_empirical_finding, evidence_quote, finding_summary, direction, intervention, comparator, outcome, population, setting, effect_estimate, study_design, limitations_or_qualifiers, confidence (float 0.0-1.0), rejection_reason."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Passage to analyze:\n"""{passage}"""'},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for attempt in range(self.max_retries):
                try:
                    res = await client.post(self.base_url, headers=headers, json=payload)
                    if res.status_code == 200:
                        content = res.json()["choices"][0]["message"]["content"]
                        data = json.loads(content)
                        if data.get("confidence") is None:
                            data["confidence"] = 0.8
                        parsed = StructuredFindingInterpretation(**data)
                        # Verify exact verbatim containment
                        if parsed.is_empirical_finding:
                            cleaned_quote = parsed.evidence_quote.strip().strip("\"'")
                            if cleaned_quote and cleaned_quote not in passage:
                                parsed.is_empirical_finding = False
                                parsed.rejection_reason = (
                                    "Quotation failed verbatim substring verification"
                                )
                        self.cache.set(cache_key, "groq", "interpret_passage", parsed.model_dump())
                        return parsed
                except Exception as e:
                    logger.warning(f"Groq passage interpretation attempt {attempt + 1} failed: {e}")

        return None

    async def triage_paper(self, paper_title: str, abstract: str, query: str) -> Dict[str, Any]:
        """Triage a candidate paper into DIRECT_EMPIRICAL, INDIRECT_EMPIRICAL, REVIEW_SYNTHESIS, BACKGROUND_THEORY, or WEAK_OR_IRRELEVANT."""
        if not self.is_available:
            # Fallback heuristic
            return {
                "category": "DIRECT_EMPIRICAL"
                if any(
                    w in abstract.lower()
                    for w in ["result", "show", "found", "trial", "experiment"]
                )
                else "INDIRECT_EMPIRICAL",
                "relevance_rationale": "Categorized via local heuristic keyword filter.",
                "selected_for_deep_dive": True,
            }

        cache_key = self.cache.compute_key(
            "groq", "triage_paper", {"title": paper_title, "query": query}
        )
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        prompt = (
            f"Question: {query}\n"
            f"Paper Title: {paper_title}\n"
            f"Abstract: {abstract[:1500]}\n\n"
            "Assess this candidate paper for the specific research question.\n"
            "Classify category into exactly one of:\n"
            "- 'DIRECT_COMPARISON': Directly compares the specific interventions/conditions requested in the question.\n"
            "- 'DIRECT_EMPIRICAL': Reports original empirical data directly on the target intervention and outcome.\n"
            "- 'INDIRECT_EVIDENCE': Investigates related variables, single-arm trials, or distinct mechanisms.\n"
            "- 'REVIEW_SYNTHESIS': Meta-analysis, systematic review, or literature synthesis.\n"
            "- 'BACKGROUND': Theoretical framing or contextual background.\n"
            "- 'IRRELEVANT': Studies an unrelated topic, different species, or incidental keyword match.\n"
            "Rules for selected_for_deep_dive:\n"
            "Set true ONLY if category is 'DIRECT_COMPARISON' or 'DIRECT_EMPIRICAL' or highly relevant 'INDIRECT_EVIDENCE'.\n"
            "If the paper is 'IRRELEVANT' or 'BACKGROUND', set selected_for_deep_dive=false.\n"
            'Return JSON: {"category": "...", "relevance_rationale": "...", "selected_for_deep_dive": true/false}'
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a scientific literature triage analyst. Return strict JSON. Strictly reject papers that do not study the queried exposure and outcome.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    res = await client.post(self.base_url, headers=headers, json=payload)
                    if res.status_code == 200:
                        data = json.loads(res.json()["choices"][0]["message"]["content"])
                        self.cache.set(cache_key, "groq", "triage_paper", data)
                        return data
                    elif res.status_code == 429:
                        await asyncio.sleep(1.0 * (attempt + 1))
            except Exception as e:
                logger.warning(f"Groq triage attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))

        # Safe fail-closed fallback: never classify as DIRECT_EMPIRICAL on failure
        return {
            "category": "IRRELEVANT",
            "relevance_rationale": "Automated triage unavailable; defaulting to irrelevant to prevent unverified ingestion.",
            "selected_for_deep_dive": False,
        }

    async def analyze_disagreements(self, query: str, claims_summary: str) -> List[Dict[str, Any]]:
        """Reasons across empirical claims to identify genuine disagreements, moderators, and explanations."""
        if not self.is_available or not claims_summary:
            return []

        cache_key = self.cache.compute_key(
            "groq", "disagreements", {"query": query, "summary": claims_summary[:2000]}
        )
        cached = self.cache.get(cache_key)
        if cached:
            return cached.get("disagreements", [])

        prompt = (
            f"Scientific Question: {query}\n\n"
            f"Evidence Ledger:\n{claims_summary}\n\n"
            "Analyze whether these studies agree, disagree, or report context-dependent effects.\n"
            "Identify: 1) What topic/metric exhibits divergence, 2) Which studies diverge, 3) The most plausible scientific explanations/moderators (e.g. population, measurement method, dose/duration, urban density, climate), 4) Suggested follow-up investigation.\n"
            'Return JSON: {"disagreements": [{"topic_or_metric": "...", "studies_in_disagreement": ["Study A", "Study B"], "plausible_moderator_or_reason": "...", "suggested_follow_up_investigation": "..."}]}'
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a rigorous scientific research synthesist. Identify genuine conflicts vs condition differences.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                res = await client.post(self.base_url, headers=headers, json=payload)
                if res.status_code == 200:
                    data = json.loads(res.json()["choices"][0]["message"]["content"])
                    disagreements = data.get("disagreements", [])
                    self.cache.set(
                        cache_key, "groq", "disagreements", {"disagreements": disagreements}
                    )
                    return disagreements
        except Exception as e:
            logger.warning(f"Groq disagreement analysis failed: {e}")

        return []

    async def synthesize_executive_summary(
        self,
        query: str,
        evidence_summary: str,
        disagreement_summary: str,
        coverage_stats: Dict[str, Any],
    ) -> Optional[Dict[str, str]]:
        """Generates an evidence-grounded executive summary answering the question directly in 15-30 seconds."""
        if not self.is_available:
            return None

        cache_key = self.cache.compute_key(
            "groq", "exec_summary", {"query": query, "evidence": evidence_summary[:2000]}
        )
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        prompt = (
            f"Original Question: {query}\n\n"
            f"Evidence Ledger:\n{evidence_summary}\n\n"
            f"Disagreement & Moderator Analysis:\n{disagreement_summary}\n\n"
            f"Coverage: {coverage_stats}\n\n"
            "Synthesize a rigorous, evidence-grounded executive summary.\n"
            "Rules:\n"
            "1. bottom_line: A direct, unambiguous scientific answer in 1-2 sentences. Never lead with 'INSUFFICIENT_EVIDENCE'.\n"
            "   IMPORTANT: Do NOT claim that an effect varies with conditions (e.g. climate or city density) unless direct multi-setting trials in the ledger explicitly verified it. If the ledger studies are single-city, explicitly qualify that cross-climate variation is not directly established.\n"
            "2. overall_evidence_assessment: State clearly whether evidence 'broadly supports', 'broadly challenges', 'shows mixed results', 'suggests context-dependent effects', or 'remains inconclusive'.\n"
            "3. strongest_supporting_evidence: Summarize the most definitive supporting paper(s) and what they measured.\n"
            "4. strongest_conflicting_evidence: Summarize studies that genuinely challenge or qualify the leading interpretation (or note if evidence is uniformly consistent).\n"
            "5. why_studies_disagree: Explain scientific causes of divergence (metrics, baseline, dose, duration, setting).\n"
            "6. confidence_and_limitations: State evidence robustness honestly without claiming uncalibrated probabilities. Explicitly list any unverified subclaims (e.g., cross-climate variation or partial vs total sleep comparisons).\n"
            "7. research_coverage: Summarize coverage counts (retrieved vs examined, abstract vs full-text).\n"
            "Return JSON matching keys: bottom_line, overall_evidence_assessment, strongest_supporting_evidence, strongest_conflicting_evidence, why_studies_disagree, confidence_and_limitations, research_coverage."
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a master scientific research analyst writing an executive briefing.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    res = await client.post(self.base_url, headers=headers, json=payload)
                    if res.status_code == 200:
                        data = json.loads(res.json()["choices"][0]["message"]["content"])
                        self.cache.set(cache_key, "groq", "exec_summary", data)
                        return data
                    elif res.status_code == 429:
                        await asyncio.sleep(1.0 * (attempt + 1))
            except Exception as e:
                logger.warning(f"Groq executive summary synthesis attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))

        return None
