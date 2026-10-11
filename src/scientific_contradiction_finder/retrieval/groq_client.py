"""Groq API client for scientific query structuring and optional candidate sentence interpretation."""

import json
import logging
import os
from typing import List, Optional

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
    evidence_quote: str = Field(
        ..., description="Exact verbatim substring from the source passage supporting the finding"
    )
    finding_summary: str = Field(..., description="Concise statement of the finding")
    direction: str = Field(
        default="UNCLEAR", description="POSITIVE, NEGATIVE, NO_DIFFERENCE, or UNCLEAR"
    )
    intervention: Optional[str] = Field(
        None, description="Intervention or exposure evaluated, null if unspecified"
    )
    comparator: Optional[str] = Field(
        None, description="Control or reference condition, null if unspecified"
    )
    outcome: Optional[str] = Field(
        None, description="Outcome metric or variable, null if unspecified"
    )
    setting: Optional[str] = Field(
        None, description="Population, benchmark, or setting, null if unspecified"
    )
    effect_estimate: Optional[str] = Field(
        None, description="Numerical effect or statistics if reported"
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
            "Determine if it reports an actual EMPIRICAL FINDING (an observation, measured result, effect, or quantitative finding).\n"
            "Reject generic promises ('models show great promise'), background introductions, study aims, biographies, or boilerplate.\n"
            "Rules:\n"
            "1. evidence_quote MUST be an EXACT, VERBATIM substring of the supplied passage.\n"
            "2. If not an empirical finding, set is_empirical_finding=false, set rejection_reason, and do not invent conditions.\n"
            "3. direction must be one of: 'POSITIVE', 'NEGATIVE', 'NO_DIFFERENCE', 'UNCLEAR'.\n"
            "4. Return strict JSON with keys: is_empirical_finding, evidence_quote, finding_summary, direction, intervention, comparator, outcome, setting, effect_estimate, confidence (number between 0.0 and 1.0), rejection_reason."
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
