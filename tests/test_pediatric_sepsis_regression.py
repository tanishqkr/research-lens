"""Regression tests for Pediatric Sepsis queries: rate limit resilience, heuristic triage fallback,
run isolation, prediction model extraction, and clinical endpoint compatibility gating.
"""

from unittest.mock import AsyncMock, patch

import pytest

from scientific_contradiction_finder.extraction.condition_tagger import (
    clean_slot_string,
)
from scientific_contradiction_finder.pipeline import ContradictionFinderPipeline
from scientific_contradiction_finder.reasoning.classifier import ConditionAwareRelationClassifier
from scientific_contradiction_finder.retrieval.groq_client import GroqClient
from scientific_contradiction_finder.schemas import (
    ConditionSlots,
    DirectionLabel,
    EvidenceSpan,
    Finding,
    FindingPair,
    Paper,
    PipelineExecutionState,
    QueryIntent,
    RelationLabel,
)


@pytest.fixture
def pipeline():
    return ContradictionFinderPipeline(use_groq=False)


def test_clean_slot_string_rejects_garbage_noun_chunks():
    """Verify clean_slot_string and invalid slots prevent generic placeholder noun chunks."""
    assert clean_slot_string("algorithm") is None
    assert clean_slot_string("The Prediction") is None
    assert clean_slot_string("6553 eligible children") is None
    assert clean_slot_string("8 times that of children") is None
    assert clean_slot_string("The Model") is None
    assert clean_slot_string("methods") is None

    # Valid scientific entities should be preserved
    assert clean_slot_string("Random Forest Model") == "Random Forest Model"
    assert clean_slot_string("Pediatric Sepsis Onset") == "Pediatric Sepsis Onset"


def test_heuristic_triage_resilience_on_sepsis_queries():
    """Verify that deterministic heuristic triage identifies relevant sepsis papers and rejects unrelated ones."""
    groq = GroqClient()
    query = "What technology can we use to predict early onset of pediatric sepsis?"

    # Relevant sepsis paper
    rel_triage = groq.heuristic_triage(
        paper_title="Machine learning for early prediction of pediatric sepsis in intensive care units",
        abstract="We developed a gradient boosting model predicting sepsis onset 4 hours before clinical deterioration in pediatric patients (AUROC 0.89).",
        query=query,
    )
    assert rel_triage["selected_for_deep_dive"] is True
    assert rel_triage["is_heuristic_fallback"] is True
    assert rel_triage["category"] in ("DIRECT_EMPIRICAL", "DIRECT_COMPARISON")

    # Completely off-target paper (e.g. crop agriculture)
    irrel_triage = groq.heuristic_triage(
        paper_title="Photosynthetic efficiency of maize under drought stress",
        abstract="Maize yield was evaluated under varied irrigation schedules across experimental agricultural fields.",
        query=query,
    )
    assert irrel_triage["selected_for_deep_dive"] is False
    assert irrel_triage["is_heuristic_fallback"] is True
    assert irrel_triage["category"] == "IRRELEVANT"


@pytest.mark.asyncio
async def test_batch_triage_rate_limit_fallback():
    """Verify that when Groq API returns 429 rate limit repeatedly, batch triage falls back to heuristic triage."""
    groq = GroqClient(api_key="gsk_dummy_test_key")
    papers = [
        {
            "title": "Machine learning prediction of pediatric sepsis onset",
            "abstract": "We evaluate an automated machine learning alert for early onset of sepsis in 2000 pediatric patients.",
        },
        {
            "title": "Soil microbial dynamics in forest ecosystems",
            "abstract": "Bacterial and fungal diversity was sequenced from temperate forest soil samples.",
        },
    ]

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # Simulate 429 response on every attempt
        mock_response = AsyncMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "0"}
        mock_post.return_value = mock_response

        results = await groq.triage_papers_batch(papers, "What technology can we use to predict early onset of pediatric sepsis?")
        assert len(results) == 2
        assert results[0]["is_heuristic_fallback"] is True
        assert results[0]["selected_for_deep_dive"] is True
        assert results[1]["is_heuristic_fallback"] is True
        assert results[1]["selected_for_deep_dive"] is False


def test_query_intent_detection(pipeline):
    """Verify that query intent is accurately routed."""
    overview_q1 = "What technology can we use to predict early onset of pediatric sepsis?"
    overview_q2 = "What technology can we use to predict early onset of pediatric sepsis using machine learning?"
    contradiction_q = "Does sleep restriction impair working memory in adults?"
    comparison_q = "Intermittent fasting versus continuous calorie restriction for weight loss"

    assert pipeline.detect_query_intent(overview_q1) == QueryIntent.RESEARCH_OVERVIEW
    assert pipeline.detect_query_intent(overview_q2) == QueryIntent.RESEARCH_OVERVIEW
    assert pipeline.detect_query_intent(contradiction_q) == QueryIntent.CONTRADICTION_ANALYSIS
    assert pipeline.detect_query_intent(comparison_q) == QueryIntent.DIRECT_COMPARISON


def test_incompatible_clinical_endpoints_rejection():
    """Verify that comparing sepsis onset prediction with acute kidney injury or mortality is rejected as INCOMPARABLE_CONDITIONS."""
    classifier = ConditionAwareRelationClassifier()

    finding_sepsis = Finding(
        finding_id="f_sepsis",
        paper_id="p1",
        paper_title="Early Sepsis Detection",
        evidence=EvidenceSpan(text="The XGBoost model predicted pediatric sepsis onset 4 hours in advance with AUROC 0.91.", start_char=0, end_char=80, confidence=0.9),
        finding_claim="XGBoost predicts pediatric sepsis onset",
        direction=DirectionLabel.POSITIVE,
        conditions=ConditionSlots(
            intervention="XGBoost Model",
            outcome_metric="Pediatric Sepsis Onset",
            dataset_setting="Pediatric Intensive Care Unit",
            confidence=0.9,
        ),
    )

    finding_aki = Finding(
        finding_id="f_aki",
        paper_id="p2",
        paper_title="Acute Kidney Injury Prediction",
        evidence=EvidenceSpan(text="Random Forest predicted acute kidney injury in hospitalized children with AUROC 0.84.", start_char=0, end_char=81, confidence=0.9),
        finding_claim="Random forest predicts acute kidney injury",
        direction=DirectionLabel.POSITIVE,
        conditions=ConditionSlots(
            intervention="Random Forest Model",
            outcome_metric="Acute Kidney Injury",
            dataset_setting="Pediatric Inpatient Unit",
            confidence=0.9,
        ),
    )

    pair = FindingPair(pair_id="pair_test", finding_a=finding_sepsis, finding_b=finding_aki)
    prediction = classifier.classify_pair(pair)

    assert prediction.predicted_relation == RelationLabel.INCOMPARABLE_CONDITIONS
    # Compatibility score should reflect actual low compatibility (diff.overall_compatibility), NOT 90.6%
    assert prediction.heuristic_score < 0.60
    assert "disjoint target conditions" in prediction.explanation.lower()


@pytest.mark.asyncio
async def test_zero_deep_dive_run_isolation(pipeline):
    """Verify that when 0 papers qualify for deep dive, run() returns 0 findings and 0 comparisons."""
    dummy_papers = [
        Paper(
            paper_id="off_target_1",
            title="Non-relevant study on plant cells",
            abstract="Study analyzing chloroplast membrane potential in Arabidopsis.",
            year=2024,
            source_provider="openalex",
            url="https://example.com/plant",
        )
    ]

    with patch.object(pipeline, "search_papers_with_diagnostics", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = (dummy_papers, {"query_raw": "test", "usable_abstracts": 1})

        res = await pipeline.run("What technology can we use to predict early onset of pediatric sepsis?")
        assert res["findings_extracted"] == 0
        assert len(res["comparisons"]) == 0
        assert res["execution_state"] == PipelineExecutionState.COMPLETED_WITH_LIMITATIONS.value
        assert res["research_report"].papers_deep_dived == 0
        assert len(res["research_report"].evidence_ledger) == 0
