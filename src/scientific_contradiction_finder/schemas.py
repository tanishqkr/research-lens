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
    NO_DIFFERENCE = "NO_DIFFERENCE"  # No statistically significant or detectable difference
    MIXED = "MIXED"  # Opposing effects across measures or conditions
    UNCLEAR = "UNCLEAR"  # Insufficient detail or unstated direction


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
    is_oa: bool = Field(default=False, description="Whether paper is Open Access")
    oa_url: str | None = Field(
        default=None, description="Direct URL to open access full text / PDF"
    )
    full_text: str | None = Field(default=None, description="Extracted full text if accessible")


class EvidenceSpan(BaseModel):
    text: str = Field(..., description="Verbatim extracted evidence sentence or passage")
    start_char: int | None = None
    end_char: int | None = None
    sentence_idx: int | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def verify_against_source(self, source_text: str) -> bool:
        """Verify that the evidence span is an exact substring of the source text."""
        return self.text.strip() in source_text

class StudyDesignType(str, Enum):
    INTERVENTIONAL_TRIAL = "INTERVENTIONAL_TRIAL"
    OBSERVATIONAL_STUDY = "OBSERVATIONAL_STUDY"
    PREDICTION_MODEL = "PREDICTION_MODEL"
    SYSTEMATIC_REVIEW_META_ANALYSIS = "SYSTEMATIC_REVIEW_META_ANALYSIS"
    MECHANISTIC_EXPLORATORY = "MECHANISTIC_EXPLORATORY"
    UNKNOWN = "UNKNOWN"


class ProvenanceSection(str, Enum):
    RESULTS = "RESULTS"
    CONCLUSION = "CONCLUSION"
    METHODS = "METHODS"
    BACKGROUND = "BACKGROUND"


class NumericalResult(BaseModel):
    arm_name: str | None = Field(default=None, description="Intervention arm or group name")
    comparator_arm: str | None = Field(default=None, description="Comparator or control arm name")
    endpoint: str | None = Field(default=None, description="Associated clinical outcome or metric endpoint")
    point_estimate: float | None = None
    metric_name: str | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = None
    unit: str | None = None
    follow_up_horizon: str | None = Field(default=None, description="Associated follow-up duration (e.g. '6 months')")
    sample_size: int | None = Field(default=None, description="Associated cohort or arm sample size")
    statistical_uncertainty: str | None = Field(default=None, description="Formulated uncertainty string (e.g. '95% CI 1.12 to 1.88, p=0.002')")
    raw_text: str | None = None


class ConditionSlots(BaseModel):
    intervention: str | None = Field(
        default=None, description="Intervention, method, model, or exposure under test"
    )
    comparator: str | None = Field(
        default="Standard Baseline / Unspecified",
        description="Comparator, control group, or reference baseline",
    )
    outcome_metric: str | None = Field(
        default=None, description="Target outcome, evaluation metric, or assessed variable"
    )
    dataset_setting: str | None = Field(
        default="General Scientific Setting",
        description="Dataset, benchmark, cohort, animal model, or experimental setting",
    )
    population: str | None = Field(
        default=None, description="Target population, species, or subjects"
    )
    effect_size: str | None = Field(
        default=None, description="Explicit quantitative or statistical effect reported"
    )
    follow_up_duration: str | None = Field(
        default=None, description="Reported duration of follow-up (e.g. '6 months', '12 months')"
    )
    sample_size: int | None = Field(
        default=None, description="Reported cohort sample size or trial participants count"
    )
    numerical_results: list[NumericalResult] = Field(
        default_factory=list, description="Structured numerical measurements extracted verbatim"
    )
    study_design: str | None = Field(
        default=None, description="Study design (e.g. RCT, observational, simulation, benchmark)"
    )
    study_type: StudyDesignType = Field(
        default=StudyDesignType.UNKNOWN, description="Classified scientific study design type"
    )
    provenance_section: ProvenanceSection = Field(
        default=ProvenanceSection.RESULTS, description="Abstract or full text section"
    )
    limitations_or_qualifiers: str | None = Field(
        default=None, description="Stated qualifiers, thresholds, or condition limits"
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
    heuristic_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Transparent heuristic comparison support score (based on evidence certainty and condition alignment). Not an uncalibrated probability.",
    )
    confidence_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Alias for heuristic_score for backward compatibility.",
    )
    unconditioned_relation: RelationLabel | None = None
    condition_differences: ConditionDifference
    provenance_verified: bool = Field(
        default=True,
        description="True if evidence spans for both findings verified against source texts",
    )
    explanation: str

    def model_post_init(self, __context: Any) -> None:
        if self.heuristic_score is None and self.confidence_score is not None:
            self.heuristic_score = self.confidence_score
        elif self.confidence_score is None and self.heuristic_score is not None:
            self.confidence_score = self.heuristic_score
        elif self.heuristic_score is None and self.confidence_score is None:
            self.heuristic_score = 0.80
            self.confidence_score = 0.80


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


class QueryIntent(str, Enum):
    RESEARCH_OVERVIEW = "RESEARCH_OVERVIEW"  # Method/technology discovery, landscape overview
    CONTRADICTION_ANALYSIS = "CONTRADICTION_ANALYSIS"  # Dispute, conflict, debate, or whether X causes/improves Y
    DIRECT_COMPARISON = "DIRECT_COMPARISON"  # Head-to-head evaluation (A vs B)
    CLAIM_VERIFICATION = "CLAIM_VERIFICATION"  # Checking specific factual or numerical claim


class PipelineExecutionState(str, Enum):
    INITIALIZED = "INITIALIZED"
    RETRIEVED = "RETRIEVED"
    TRIAGED = "TRIAGED"
    DEEP_DIVED = "DEEP_DIVED"
    EXTRACTED = "EXTRACTED"
    COMPARED = "COMPARED"
    SYNTHESIZED = "SYNTHESIZED"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_LIMITATIONS = "COMPLETED_WITH_LIMITATIONS"
    FAILED = "FAILED"


class PredictionModelDetails(BaseModel):
    model_method: str | None = Field(default=None, description="Prediction model, architecture, or algorithm name")
    input_features: list[str] = Field(default_factory=list, description="Input variables or biomarker features")
    target_condition: str | None = Field(default=None, description="Target condition or disease to predict")
    prediction_horizon: str | None = Field(default=None, description="Time window of prediction (e.g. 4-6 hours prior to onset)")
    comparator_score: str | None = Field(default=None, description="Comparator clinical score (e.g. SIRS, SOFA, pSOFA, PELOD)")
    metrics_and_values: dict[str, str] = Field(default_factory=dict, description="Evaluation metrics (AUROC, Sensitivity, Specificity, AUPRC)")
    validation_design: str | None = Field(default=None, description="Validation type (e.g. internal cross-validation, temporal, external cohort)")
    population_setting: str | None = Field(default=None, description="Cohort description, setting (e.g. PICU, ED), sample size")
    limitations: str | None = Field(default=None, description="Reported model limitations or dataset bias")


class PaperTriageClassification(str, Enum):
    DIRECT_COMPARISON = "DIRECT_COMPARISON"
    DIRECT_EMPIRICAL = "DIRECT_EMPIRICAL"
    INDIRECT_EVIDENCE = "INDIRECT_EVIDENCE"
    INDIRECT_EMPIRICAL = "INDIRECT_EMPIRICAL"
    REVIEW_SYNTHESIS = "REVIEW_SYNTHESIS"
    BACKGROUND_THEORY = "BACKGROUND_THEORY"
    BACKGROUND = "BACKGROUND"
    IRRELEVANT = "IRRELEVANT"
    WEAK_OR_IRRELEVANT = "WEAK_OR_IRRELEVANT"
    TRIAGE_UNAVAILABLE = "TRIAGE_UNAVAILABLE"
    UNCERTAIN = "UNCERTAIN"


class PaperTriageRecord(BaseModel):
    paper_id: str
    title: str
    category: PaperTriageClassification
    relevance_rationale: str
    selected_for_deep_dive: bool = False
    is_heuristic_fallback: bool = False


class SupportedEvidenceClaim(BaseModel):
    claim_id: str
    paper_id: str
    paper_title: str
    doi: str | None = None
    url: str | None = None
    exact_source_passage: str
    study_design: str | None = None
    study_type: StudyDesignType = StudyDesignType.UNKNOWN
    provenance_section: ProvenanceSection = ProvenanceSection.RESULTS
    population_setting: str | None = None
    intervention_exposure: str | None = None
    comparator: str | None = None
    outcome: str | None = None
    direction: DirectionLabel = DirectionLabel.UNCLEAR
    effect_size: str | None = None
    follow_up_duration: str | None = None
    sample_size: int | None = None
    numerical_results: list[NumericalResult] = Field(default_factory=list)
    conditions_and_moderators: str | None = None
    limitations: str | None = None
    prediction_model_details: PredictionModelDetails | None = None
    full_text_used: bool = False
    is_direct: bool = True
    is_review_synthesis: bool = False


class SubclaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    QUALIFIED = "QUALIFIED"
    UNRESOLVED = "UNRESOLVED"


class ExplanationVerificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PLAUSIBLE_UNVERIFIED = "PLAUSIBLE_UNVERIFIED"
    UNRESOLVED = "UNRESOLVED"


class SubclaimAssessment(BaseModel):
    subclaim_text: str = Field(
        ..., description="Specific subclaim decomposed from the original query"
    )
    status: SubclaimStatus = Field(
        ..., description="Evidence status: SUPPORTED, QUALIFIED, or UNRESOLVED"
    )
    rationale: str = Field(
        ..., description="Grounded explanation strictly referencing verified evidence"
    )
    evidence_paper_ids: list[str] = Field(
        default_factory=list,
        description="IDs of papers in the ledger supporting/qualifying this subclaim",
    )


class DisagreementExplanation(BaseModel):
    topic_or_metric: str
    studies_in_disagreement: list[str]
    plausible_moderator_or_reason: str
    verification_status: ExplanationVerificationStatus = Field(
        default=ExplanationVerificationStatus.PLAUSIBLE_UNVERIFIED,
        description="Whether this explanation is empirically SUPPORTED by cited trials, PLAUSIBLE_UNVERIFIED hypothesis, or UNRESOLVED.",
    )
    supporting_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence claim IDs directly demonstrating this moderating factor",
    )
    suggested_follow_up_investigation: str | None = None


class ExecutiveSummary(BaseModel):
    bottom_line: str = Field(..., description="Direct concise scientific answer to the question")
    overall_evidence_assessment: str = Field(
        ...,
        description="broadly supports, broadly challenges, mixed results, context-dependent, or inconclusive",
    )
    subclaim_assessments: list[SubclaimAssessment] = Field(
        default_factory=list, description="Independent evaluations for each decomposed subclaim"
    )
    strongest_supporting_evidence: str = Field(
        ..., description="Most informative supporting study/group and findings"
    )
    strongest_conflicting_evidence: str = Field(
        ..., description="Studies challenging or qualifying the leading interpretation"
    )
    why_studies_disagree: str = Field(
        ..., description="Plausible explanations (population, protocol, dose, duration, metric)"
    )
    confidence_and_limitations: str = Field(
        ..., description="Assessment of evidence strength and uncalibrated limitations"
    )
    research_coverage: str = Field(
        ..., description="Counts of papers found, examined in depth, and OA full-text coverage"
    )


class ResearchReport(BaseModel):
    query: str
    run_id: str | None = None
    query_intent: QueryIntent = QueryIntent.CONTRADICTION_ANALYSIS
    execution_state: PipelineExecutionState = PipelineExecutionState.COMPLETED
    executive_summary: ExecutiveSummary
    candidate_papers_retrieved: int
    papers_triaged: int
    papers_abstract_screened: int = 0
    papers_deep_dived: int
    full_texts_accessible: int
    findings_extracted: int = 0
    comparisons_evaluated: int = 0
    retrieval_diagnostics: dict[str, Any] = Field(default_factory=dict)
    subclaim_assessments: list[SubclaimAssessment] = Field(default_factory=list)
    triage_records: list[PaperTriageRecord] = Field(default_factory=list)
    evidence_ledger: list[SupportedEvidenceClaim] = Field(default_factory=list)
    disagreements: list[DisagreementExplanation] = Field(default_factory=list)
    detailed_comparisons: list[Prediction] = Field(default_factory=list)
    investigation_rounds: int = 1
    stopped_reason: str = "Investigation completed with adequate evidence."
