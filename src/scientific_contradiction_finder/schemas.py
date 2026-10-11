"""Data contracts and schemas for Scientific Contradiction Finder."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RelationLabel(str, Enum):
    SUPPORTS_OR_CONSISTENT = "SUPPORTS_OR_CONSISTENT"
    CONTRADICTS = "CONTRADICTS"
    QUALIFIES_OR_PARTIAL_DISAGREEMENT = "QUALIFIES_OR_PARTIAL_DISAGREEMENT"
    INCOMPARABLE_CONDITIONS = "INCOMPARABLE_CONDITIONS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class DirectionLabel(str, Enum):
    POSITIVE = "POSITIVE"  # Intervention increases outcome / superior to comparator
    NEGATIVE = "NEGATIVE"  # Intervention decreases outcome / inferior to comparator
    NO_DIFFERENCE = "NO_DIFFERENCE"  # No statistically significant difference
    UNCLEAR = "UNCLEAR"


class Paper(BaseModel):
    paper_id: str = Field(
        ..., description="Unique paper identifier (DOI, OpenAlex ID, or arXiv ID)"
    )
    title: str = Field(..., description="Title of the paper")
    abstract: str = Field(..., description="Full text abstract")
    doi: str | None = None
    url: str | None = None
    venue: str | None = None
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    source_provider: str = Field(
        ..., description="Provider source (e.g., 'openalex', 'arxiv', 'benchmark')"
    )


class EvidenceSpan(BaseModel):
    text: str = Field(..., description="Verbatim extracted evidence sentence or passage")
    start_char: int | None = None
    end_char: int | None = None
    sentence_idx: int | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def verify_against_source(self, source_text: str) -> bool:
        """Verify that the evidence span is an exact substring of the source text."""
        return self.text.strip() in source_text


class ConditionSlots(BaseModel):
    intervention: str = Field(
        ..., description="Intervention, method, model, or exposure under test"
    )
    comparator: str | None = Field(
        default="Standard Baseline / Unspecified",
        description="Comparator, control group, or reference baseline",
    )
    outcome_metric: str = Field(..., description="Target outcome, evaluation metric, or phenotype")
    dataset_setting: str | None = Field(
        default="General Scientific Setting",
        description="Dataset, benchmark, cohort, animal model, or experimental setting",
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    is_unknown: bool = Field(
        default=False, description="True if condition parameters could not be reliably extracted"
    )


class Finding(BaseModel):
    finding_id: str
    paper_id: str
    paper_title: str
    paper_doi: str | None = None
    paper_url: str | None = None
    evidence: EvidenceSpan
    finding_claim: str = Field(
        ..., description="Normalized natural language assertion of the finding"
    )
    direction: DirectionLabel = DirectionLabel.UNCLEAR
    conditions: ConditionSlots


class ConditionDifference(BaseModel):
    intervention_match: bool
    intervention_similarity: float = Field(..., ge=0.0, le=1.0)
    comparator_match: bool
    comparator_similarity: float = Field(..., ge=0.0, le=1.0)
    outcome_match: bool
    outcome_similarity: float = Field(..., ge=0.0, le=1.0)
    setting_match: bool
    setting_similarity: float = Field(..., ge=0.0, le=1.0)
    overall_compatibility: float = Field(..., ge=0.0, le=1.0)
    summary_of_differences: str


class FindingPair(BaseModel):
    pair_id: str
    finding_a: Finding
    finding_b: Finding
    ground_truth_relation: RelationLabel | None = None
    domain: str | None = None
    notes: str | None = None


class Prediction(BaseModel):
    pair_id: str
    predicted_relation: RelationLabel
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    unconditioned_relation: RelationLabel | None = None
    condition_differences: ConditionDifference
    provenance_verified: bool = Field(
        default=True,
        description="True if evidence spans for both findings verified against source texts",
    )
    explanation: str


class EvaluationResult(BaseModel):
    dataset_name: str
    total_pairs: int
    evaluated_pairs: int
    abstained_count: int
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]
    metadata: dict[str, Any] = Field(default_factory=dict)
