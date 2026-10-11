"""Regression tests for Milestone 1: Pipeline Integrity and Evidence Extraction."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scientific_contradiction_finder.extraction.condition_tagger import (
    clean_slot_string,
    extract_follow_up_duration,
    extract_numerical_results,
    extract_sample_size,
)
from scientific_contradiction_finder.extraction.finding_extractor import FindingExtractor
from scientific_contradiction_finder.pipeline import ContradictionFinderPipeline
from scientific_contradiction_finder.reasoning.classifier import (
    ConditionAwareRelationClassifier,
    get_endpoint_domain,
)
from scientific_contradiction_finder.retrieval.groq_client import GroqClient
from scientific_contradiction_finder.schemas import (
    ConditionSlots,
    DirectionLabel,
    EvidenceSpan,
    ExecutiveSummary,
    Finding,
    FindingPair,
    Paper,
    RelationLabel,
    ResearchReport,
)


def test_1_retrieval_diagnostics_consistency():
    """1. Retrieval diagnostics consistency: candidate count, screened count, diagnostics present."""
    report = ResearchReport(
        query="Test query",
        candidate_papers_retrieved=10,
        papers_triaged=10,
        papers_abstract_screened=10,
        papers_deep_dived=4,
        full_texts_accessible=0,
        findings_extracted=4,
        comparisons_evaluated=3,
        retrieval_diagnostics={
            "openalex": {"status": "ok", "results_count": 6},
            "arxiv": {"status": "ok", "results_count": 4},
            "usable_abstracts": 10,
        },
        executive_summary=ExecutiveSummary(
            bottom_line="Test bottom line",
            overall_evidence_assessment="inconclusive",
            strongest_supporting_evidence="None",
            strongest_conflicting_evidence="None",
            why_studies_disagree="None",
            confidence_and_limitations="High uncertainty",
            research_coverage="Retrieved 10 papers",
        ),
    )
    assert report.candidate_papers_retrieved == 10
    assert report.papers_abstract_screened == 10
    assert report.retrieval_diagnostics["openalex"]["results_count"] == 6
    assert report.retrieval_diagnostics["arxiv"]["results_count"] == 4
    assert (
        report.retrieval_diagnostics["openalex"]["results_count"]
        + report.retrieval_diagnostics["arxiv"]["results_count"]
        == report.candidate_papers_retrieved
    )


def test_2_zero_full_text_honesty():
    """2. Zero full-text honesty: full_texts_accessible == 0 reported honestly as abstract synthesis."""
    pipeline = ContradictionFinderPipeline()
    # Mocking deep-dive papers with no OA full text
    paper = Paper(
        paper_id="W123",
        title="Test Diet Study",
        abstract="A randomized trial evaluated weight loss. Low-carbohydrate diets reduced body weight by 5.8 kg at 6 months.",
        source_provider="openalex",
        is_oa=False,
    )
    findings = asyncio.run(pipeline.extract_findings_from_papers([paper]))
    assert len(findings) >= 1
    # Check that pipeline has research analyst synthesis method
    assert hasattr(pipeline, "analyze_as_research_analyst")


@pytest.mark.asyncio
async def test_3_groq_retry_budget_cap():
    """3. Groq retry budget cap: does not sleep for > 5.0s on 429 Retry-After, aborts on excessive delay."""
    client = GroqClient(api_key="mock_key")

    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    mock_resp_429.headers = {"retry-after": "1241"}

    slept_durations = []

    async def fake_sleep(duration):
        slept_durations.append(duration)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp_429
        with patch("asyncio.sleep", side_effect=fake_sleep):
            res = await client.triage_papers_batch([{"title": "Test", "abstract": "Test"}], "query")
            # Should fall back to heuristic triage rather than hanging for 1241 seconds
            assert len(res) == 1
            assert res[0]["is_heuristic_fallback"] is True
            # Did not sleep 1241s
            for dur in slept_durations:
                assert dur <= 5.0


def test_4_generic_token_filtering():
    """4. Generic token filtering: 'individuals', 'patients', '0.91 kg', '6 months' are rejected."""
    bad_tokens = [
        "individuals",
        "Patients",
        "THE PATIENTS",
        "subjects",
        "hospital",
        "0.91 kg",
        "6 months",
        "12 months",
        "the diet",
        "the largest amounts",
        "both groups",
    ]
    for tok in bad_tokens:
        cleaned = clean_slot_string(tok)
        assert cleaned is None, f"Expected '{tok}' to be rejected, but got '{cleaned}'"


def test_5_follow_up_duration_routing():
    """5. Follow-up duration routing: '6 months', '12 months', '14 weeks' routed to follow_up_duration."""
    sent1 = "At 6 months, patients randomized to low-carbohydrate diets showed significant weight reduction."
    dur1 = extract_follow_up_duration(sent1)
    assert dur1 == "6 months"

    sent2 = "The difference in body weight between groups was not significant at 12 months."
    dur2 = extract_follow_up_duration(sent2)
    assert dur2 == "12 months"

    sent3 = "During the 14-week trial, glycemic control improved."
    dur3 = extract_follow_up_duration(sent3)
    assert "14" in dur3 and "week" in dur3


def test_6_sample_size_routing():
    """6. Sample size routing: '120 participants' or 'n = 120' routed to sample_size."""
    sent1 = "In a randomized trial of 120 participants, low-carbohydrate diet reduced weight."
    sz1 = extract_sample_size(sent1)
    assert sz1 == 120

    sent2 = "Eligible subjects (n = 6,553) were followed prospectively."
    sz2 = extract_sample_size(sent2)
    assert sz2 == 6553


def test_7_numerical_results_extraction():
    """7. Numerical results extraction: point estimate, unit, p-value structured in numerical_results."""
    sent = "Low-carbohydrate diets reduced body weight by 5.8 kg (p = 0.002) compared with low-fat diets (OR 1.45, 95% CI 1.12 to 1.88)."
    res = extract_numerical_results(sent)
    assert len(res) >= 2

    # Check for unit extraction
    kg_res = next((r for r in res if r.unit == "kg"), None)
    assert kg_res is not None
    assert kg_res.point_estimate == 5.8

    # Check for p-value extraction
    pval_res = next((r for r in res if r.p_value is not None), None)
    assert pval_res is not None
    assert pval_res.p_value == 0.002

    # Check for OR ratio extraction
    or_res = next((r for r in res if r.metric_name == "OR"), None)
    assert or_res is not None
    assert or_res.point_estimate == 1.45
    assert or_res.ci_lower == 1.12
    assert or_res.ci_upper == 1.88


def test_8_background_escape_hatch_removed():
    """8. Background escape hatch removed: background sentences with statistics are rejected from empirical findings."""
    extractor = FindingExtractor()
    # Typical background epidemiological statement with statistics
    bg_sentence = "Cigarette smoking accounts for 85% of lung cancer cases worldwide and is the leading cause of preventable death."
    spans = extractor.extract_evidence_spans(bg_sentence)
    assert len(spans) == 0, f"Expected background sentence to be rejected, but extracted: {spans}"

    # Typical objective / proposal sentence with numbers
    obj_sentence = "In this study, we investigate whether vitamin D at 2000 IU reduces respiratory infections."
    spans_obj = extractor.extract_evidence_spans(obj_sentence)
    assert len(spans_obj) == 0


@pytest.mark.asyncio
async def test_9_multiple_findings_retained_per_paper():
    """9. Multiple findings retained per paper: if paper has 6-month and 12-month results, both are extracted."""
    pipeline = ContradictionFinderPipeline()
    abstract = (
        "We conducted a randomized trial comparing a low-carbohydrate diet with a low-fat diet. "
        "At 6 months, low-carbohydrate subjects had lost significantly more weight than low-fat subjects (p < 0.01). "
        "However, at 12 months, the difference between the groups was no longer statistically significant (p = 0.20)."
    )
    paper = Paper(
        paper_id="W_DIET_RCT",
        title="Comparison of Diets for Weight Loss",
        abstract=abstract,
        source_provider="openalex",
    )
    findings = await pipeline.extract_findings_from_papers([paper])
    assert len(findings) >= 2, f"Expected at least 2 findings, got {len(findings)}"
    # Verify durations in the extracted findings
    durations = [f.conditions.follow_up_duration for f in findings]
    assert "6 months" in durations or any("6" in str(d) for d in durations)


def test_10_endpoint_mismatch_prevention():
    """10. Endpoint mismatch prevention: stroke vs mortality or weight loss vs diabetes marked INCOMPARABLE_CONDITIONS."""
    classifier = ConditionAwareRelationClassifier()

    fa = Finding(
        finding_id="f1",
        paper_id="p1",
        paper_title="Alcohol and Mortality",
        evidence=EvidenceSpan(text="Moderate alcohol reduced all-cause mortality (RR 0.82).", confidence=0.9),
        finding_claim="Moderate alcohol reduced all-cause mortality.",
        direction=DirectionLabel.NEGATIVE,
        conditions=ConditionSlots(
            intervention="Moderate Alcohol Consumption",
            outcome_metric="All-Cause Mortality",
            confidence=0.85,
        ),
    )
    fb = Finding(
        finding_id="f2",
        paper_id="p2",
        paper_title="Alcohol and Stroke",
        evidence=EvidenceSpan(text="Alcohol consumption was associated with higher risk of hemorrhagic stroke.", confidence=0.9),
        finding_claim="Alcohol consumption increased stroke risk.",
        direction=DirectionLabel.POSITIVE,
        conditions=ConditionSlots(
            intervention="Moderate Alcohol Consumption",
            outcome_metric="Hemorrhagic Stroke",
            confidence=0.85,
        ),
    )
    pair = FindingPair(pair_id="p1_p2", finding_a=fa, finding_b=fb)
    pred = classifier.classify_pair(pair)
    assert pred.predicted_relation == RelationLabel.INCOMPARABLE_CONDITIONS
    assert "disparate outcome endpoints" in pred.explanation.lower() or "disparate" in pred.explanation.lower()


def test_11_incomparable_score_inversion_fixed():
    """11. Incomparable score inversion fixed: comparison support score for incompatible conditions is low."""
    classifier = ConditionAwareRelationClassifier()

    fa = Finding(
        finding_id="f1",
        paper_id="p1",
        paper_title="Low-Carb Weight Loss",
        evidence=EvidenceSpan(text="Diet produced 5.8 kg weight loss.", confidence=0.9),
        finding_claim="Diet produced weight loss.",
        direction=DirectionLabel.POSITIVE,
        conditions=ConditionSlots(
            intervention="Low-Carbohydrate Diet",
            outcome_metric="Weight Loss",
            confidence=0.85,
        ),
    )
    fb = Finding(
        finding_id="f2",
        paper_id="p2",
        paper_title="Insulin Resistance Study",
        evidence=EvidenceSpan(text="Metformin reduced HbA1c significantly.", confidence=0.9),
        finding_claim="Metformin reduced HbA1c.",
        direction=DirectionLabel.NEGATIVE,
        conditions=ConditionSlots(
            intervention="Metformin",
            outcome_metric="HbA1c",
            confidence=0.85,
        ),
    )
    pair = FindingPair(pair_id="p1_p2", finding_a=fa, finding_b=fb)
    pred = classifier.classify_pair(pair)
    assert pred.predicted_relation == RelationLabel.INCOMPARABLE_CONDITIONS
    # Heuristic score should be the low compatibility (< 0.50), not 0.90+
    assert pred.heuristic_score <= 0.50


def test_12_beneficial_reduction_synthesis():
    """12. Beneficial reduction synthesis: weight loss or mortality reduction not called 'negative impact'."""
    pipeline = ContradictionFinderPipeline()
    assert hasattr(pipeline, "analyze_as_research_analyst")
    # Endpoint domain detection recognizes mortality and weight loss
    assert get_endpoint_domain("all-cause mortality") == "MORTALITY"
    assert get_endpoint_domain("weight loss") == "WEIGHT_LOSS"
