/**
 * Strict TypeScript types matching the backend Pydantic contracts
 * in src/scientific_contradiction_finder/schemas.py
 */

export type RelationLabel =
  | "SUPPORTS_OR_CONSISTENT"
  | "CONTRADICTS"
  | "QUALIFIES_OR_PARTIAL_DISAGREEMENT"
  | "INCOMPARABLE_CONDITIONS"
  | "INSUFFICIENT_EVIDENCE";

export type DirectionLabel =
  | "POSITIVE"
  | "NEGATIVE"
  | "NO_DIFFERENCE"
  | "MIXED"
  | "UNCLEAR";

export type StudyDesignType =
  | "INTERVENTIONAL_TRIAL"
  | "OBSERVATIONAL_STUDY"
  | "PREDICTION_MODEL"
  | "SYSTEMATIC_REVIEW_META_ANALYSIS"
  | "MECHANISTIC_EXPLORATORY"
  | "UNKNOWN";

export type ProvenanceSection =
  | "RESULTS"
  | "CONCLUSION"
  | "METHODS"
  | "BACKGROUND";

export type QueryIntent =
  | "RESEARCH_OVERVIEW"
  | "CONTRADICTION_ANALYSIS"
  | "DIRECT_COMPARISON"
  | "CLAIM_VERIFICATION";

export type PipelineExecutionState =
  | "INITIALIZED"
  | "RETRIEVED"
  | "TRIAGED"
  | "DEEP_DIVED"
  | "EXTRACTED"
  | "COMPARED"
  | "SYNTHESIZED"
  | "COMPLETED"
  | "COMPLETED_WITH_LIMITATIONS"
  | "FAILED";

export type PaperTriageClassification =
  | "DIRECT_COMPARISON"
  | "DIRECT_EMPIRICAL"
  | "INDIRECT_EVIDENCE"
  | "INDIRECT_EMPIRICAL"
  | "REVIEW_SYNTHESIS"
  | "BACKGROUND_THEORY"
  | "BACKGROUND"
  | "IRRELEVANT"
  | "WEAK_OR_IRRELEVANT"
  | "TRIAGE_UNAVAILABLE"
  | "UNCERTAIN";

export type SubclaimStatus = "SUPPORTED" | "QUALIFIED" | "UNRESOLVED";

export type ExplanationVerificationStatus =
  | "SUPPORTED"
  | "PLAUSIBLE_UNVERIFIED"
  | "UNRESOLVED";

export interface EvidenceSpan {
  text: string;
  start_char?: number | null;
  end_char?: number | null;
  sentence_idx?: number | null;
  confidence: number;
}

export interface NumericalResult {
  arm_name?: string | null;
  comparator_arm?: string | null;
  endpoint?: string | null;
  point_estimate?: number | null;
  metric_name?: string | null;
  ci_lower?: number | null;
  ci_upper?: number | null;
  p_value?: number | null;
  unit?: string | null;
  follow_up_horizon?: string | null;
  sample_size?: number | null;
  statistical_uncertainty?: string | null;
  raw_text?: string | null;
}

export interface PredictionModelDetails {
  model_method?: string | null;
  input_features?: string[];
  target_condition?: string | null;
  prediction_horizon?: string | null;
  comparator_score?: string | null;
  metrics_and_values?: Record<string, string>;
  validation_design?: string | null;
  population_setting?: string | null;
  limitations?: string | null;
}

export interface ConditionSlots {
  intervention?: string | null;
  comparator?: string | null;
  outcome_metric?: string | null;
  dataset_setting?: string | null;
  population?: string | null;
  effect_size?: string | null;
  follow_up_duration?: string | null;
  sample_size?: number | null;
  numerical_results: NumericalResult[];
  study_design?: string | null;
  study_type: StudyDesignType;
  provenance_section: ProvenanceSection;
  limitations_or_qualifiers?: string | null;
  confidence: number;
  is_unknown: boolean;
}

export interface Finding {
  finding_id: string;
  paper_id: string;
  paper_title: string;
  paper_doi?: string | null;
  paper_url?: string | null;
  evidence: EvidenceSpan;
  finding_claim: string;
  direction: DirectionLabel;
  conditions: ConditionSlots;
}

export interface ConditionDifference {
  intervention_match: boolean;
  intervention_similarity: number;
  comparator_match: boolean;
  comparator_similarity: number;
  outcome_match: boolean;
  outcome_similarity: number;
  setting_match: boolean;
  setting_similarity: number;
  overall_compatibility: number;
  summary_of_differences: string;
}

export interface Prediction {
  pair_id: string;
  predicted_relation: RelationLabel;
  heuristic_score?: number | null;
  confidence_score?: number | null;
  unconditioned_relation?: RelationLabel | null;
  condition_differences: ConditionDifference;
  provenance_verified: boolean;
  explanation: string;
}

export interface FindingPair {
  pair_id: string;
  finding_a: Finding;
  finding_b: Finding;
  ground_truth_relation?: RelationLabel | null;
  domain?: string | null;
  notes?: string | null;
}

export interface SupportedEvidenceClaim {
  claim_id: string;
  paper_id: string;
  paper_title: string;
  doi?: string | null;
  url?: string | null;
  exact_source_passage: string;
  study_design?: string | null;
  study_type: StudyDesignType;
  provenance_section: ProvenanceSection;
  population_setting?: string | null;
  intervention_exposure?: string | null;
  comparator?: string | null;
  outcome?: string | null;
  direction: DirectionLabel;
  effect_size?: string | null;
  follow_up_duration?: string | null;
  sample_size?: number | null;
  numerical_results: NumericalResult[];
  conditions_and_moderators?: string | null;
  limitations?: string | null;
  prediction_model_details?: PredictionModelDetails | null;
  full_text_used: boolean;
  is_direct: boolean;
  is_review_synthesis: boolean;
}

export interface SubclaimAssessment {
  subclaim_text: string;
  status: SubclaimStatus;
  rationale: string;
  evidence_paper_ids: string[];
}

export interface DisagreementExplanation {
  topic_or_metric: string;
  studies_in_disagreement: string[];
  plausible_moderator_or_reason: string;
  verification_status: ExplanationVerificationStatus;
  supporting_evidence_ids: string[];
  suggested_follow_up_investigation?: string | null;
}

export interface ExecutiveSummary {
  bottom_line: string;
  overall_evidence_assessment: string;
  subclaim_assessments?: SubclaimAssessment[];
  strongest_supporting_evidence: string;
  strongest_conflicting_evidence: string;
  why_studies_disagree: string;
  confidence_and_limitations: string;
  research_coverage: string;
}

export interface PaperTriageRecord {
  paper_id: string;
  title: string;
  category: PaperTriageClassification;
  relevance_rationale: string;
  selected_for_deep_dive: boolean;
  is_heuristic_fallback: boolean;
}

export interface ProviderDiagnostic {
  status: string | number | null;
  results_count: number;
  error: string | null;
}

export interface RetrievalDiagnostics {
  query_raw?: string;
  groq_assisted?: boolean;
  groq_structured?: {
    search_keywords?: string[];
    population?: string;
    intervention?: string;
    comparator?: string;
    outcome?: string;
  } | null;
  search_queries_dispatched?: string[];
  openalex?: ProviderDiagnostic;
  arxiv?: ProviderDiagnostic;
  raw_total?: number;
  usable_abstracts?: number;
  after_dedup?: number;
  run_id?: string;
  execution_state?: string;
}

export interface ResearchReport {
  query: string;
  run_id?: string | null;
  query_intent: QueryIntent;
  execution_state: PipelineExecutionState;
  executive_summary: ExecutiveSummary;
  candidate_papers_retrieved: number;
  papers_triaged: number;
  papers_abstract_screened: number;
  papers_deep_dived: number;
  full_texts_accessible: number;
  findings_extracted: number;
  comparisons_evaluated: number;
  retrieval_diagnostics: RetrievalDiagnostics;
  subclaim_assessments: SubclaimAssessment[];
  triage_records: PaperTriageRecord[];
  evidence_ledger: SupportedEvidenceClaim[];
  disagreements: DisagreementExplanation[];
  detailed_comparisons: Prediction[];
  investigation_rounds: number;
  stopped_reason: string;
}

export interface FindingComparisonItem {
  pair: FindingPair;
  prediction: Prediction;
}

export interface AnalysisResponse {
  query: string;
  run_id?: string | null;
  query_intent: QueryIntent;
  execution_state: PipelineExecutionState;
  papers_found: number;
  findings_extracted: number;
  comparisons: FindingComparisonItem[];
  findings: Finding[];
  diagnostics: RetrievalDiagnostics;
  research_report: ResearchReport;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  providers: {
    openalex: boolean;
    arxiv: boolean;
    groq: boolean;
  };
  offline_mode: boolean;
}

export interface PresetsResponse {
  presets: string[];
}

export interface BenchmarkPairsResponse {
  total_pairs: number;
  pairs: FindingPair[];
}

export interface BenchmarkClassifyResponse {
  prediction: Prediction;
}
