"""Comprehensive regression test suite for scientific extraction, condition tagging, and relation classification."""

import pytest

from scientific_contradiction_finder.extraction.condition_tagger import ConditionTagger
from scientific_contradiction_finder.extraction.finding_extractor import FindingExtractor
from scientific_contradiction_finder.pipeline import ContradictionFinderPipeline
from scientific_contradiction_finder.reasoning.classifier import ConditionAwareRelationClassifier
from scientific_contradiction_finder.schemas import (
    ConditionSlots,
    DirectionLabel,
    EvidenceSpan,
    Finding,
    FindingPair,
    Paper,
    RelationLabel,
)


def test_rejection_of_background_and_methodology():
    extractor = FindingExtractor()

    # 1. Background statements
    bg_1 = "Chronic sleep restriction during adolescence is common."
    assert len(extractor.extract_evidence_spans(bg_1)) == 0, (
        "Background sleep claim should be rejected!"
    )

    # 2. Methodology statements
    meth_1 = (
        "Ninety college students were administered the structured clinical interview for DSM-IV."
    )
    assert len(extractor.extract_evidence_spans(meth_1)) == 0, (
        "Clinical interview method should be rejected!"
    )

    # 3. Review intentions
    rev_1 = "In this review, we examine the literature regarding tree canopy and microclimates."
    assert len(extractor.extract_evidence_spans(rev_1)) == 0, (
        "Review introduction should be rejected!"
    )

    # 4. Generic promises
    prom_1 = "Vision Transformer models have demonstrated great promise across computer vision benchmarks."
    assert len(extractor.extract_evidence_spans(prom_1)) == 0, "Generic promise should be rejected!"


def test_acceptance_of_empirical_findings_without_pvalues():
    extractor = FindingExtractor()

    # Qualitative/empirical result without p-value
    sent = "The results revealed that increasing tree canopy cover provided consistent cooling across all monitored urban neighborhoods."
    spans = extractor.extract_evidence_spans(sent)
    assert len(spans) > 0, "Valid empirical finding without p-value should be accepted!"


def test_clean_scientific_condition_slots():
    tagger = ConditionTagger()

    # Test "The researcher found..." does not become the intervention
    sent_res = "After conducting the trial, the researchers found that 4 hours of sleep restriction impaired working memory accuracy."
    cond_res = tagger.extract_conditions(sent_res)
    assert cond_res.intervention not in (
        "The Researcher",
        "The Researchers",
        "The Results",
        "The Study",
    )
    assert "sleep restriction" in cond_res.intervention.lower()

    # Test measurement method does not become the outcome
    sent_bike = "Bike-mounted temperature sensors revealed daytime air temperature reductions of 1.5°C under high canopy cover."
    cond_bike = tagger.extract_conditions(sent_bike)
    assert "sensors" not in cond_bike.outcome_metric.lower()
    assert "air temperature" in cond_bike.outcome_metric.lower()


def test_direction_semantics_and_mixed_effects():
    extractor = FindingExtractor()

    # 1. Opposing measure / mixed: reaction time increases without affecting accuracy
    sent_mixed = "Reaction times during verbal working memory tasks increased significantly (p < 0.01), without affecting accuracy."
    assert extractor.infer_direction(sent_mixed) == DirectionLabel.MIXED

    # 2. Cooling effect: tree canopy decreases temperature
    sent_cool = "Urban tree canopy decreased daytime air temperature by 2.4°C."
    assert extractor.infer_direction(sent_cool) == DirectionLabel.NEGATIVE

    # 3. Impervious cover increases temperature
    sent_heat = "Daytime air temperature increased linearly with increasing impervious cover."
    assert extractor.infer_direction(sent_heat) == DirectionLabel.POSITIVE


def test_preservation_of_scientific_conditions():
    tagger = ConditionTagger()

    # Urban heat conditions
    sent_heat = "Daytime air temperature was reduced by 2.1°C when canopy cover exceeded 40% in high density urban areas."
    cond_heat = tagger.extract_conditions(sent_heat)
    assert "canopy cover" in cond_heat.intervention.lower()
    assert "air temperature" in cond_heat.outcome_metric.lower()
    assert cond_heat.limitations_or_qualifiers is not None
    assert "40%" in cond_heat.limitations_or_qualifiers

    # Sleep conditions
    sent_sleep = (
        "Total sleep deprivation impaired visual working memory in healthy adults after 24 hours."
    )
    cond_sleep = tagger.extract_conditions(sent_sleep)
    assert "sleep deprivation" in cond_sleep.intervention.lower()
    assert "working memory" in cond_sleep.outcome_metric.lower()
    assert cond_sleep.population == "Healthy Adults"


def test_verdict_distinction_and_support_score():
    classifier = ConditionAwareRelationClassifier()

    span_a = EvidenceSpan(
        text="Urban tree canopy cover decreased daytime air temperature by 2.0°C in dense urban centers.",
        confidence=0.95,
    )
    span_b_contradict = EvidenceSpan(
        text="Urban tree canopy cover increased daytime air temperature by 0.5°C in dense urban centers.",
        confidence=0.95,
    )
    span_b_qualify = EvidenceSpan(
        text="Urban tree canopy cover showed no significant effect on daytime air temperature in arid desert climates.",
        confidence=0.95,
    )

    finding_a = Finding(
        finding_id="f_a",
        paper_id="p_a",
        paper_title="Study A",
        evidence=span_a,
        finding_claim=span_a.text,
        direction=DirectionLabel.NEGATIVE,
        conditions=ConditionSlots(
            intervention="Urban Tree Canopy Cover",
            comparator="Open Urban Baseline",
            outcome_metric="Daytime Air Temperature",
            dataset_setting="Dense Urban Centers",
            confidence=0.90,
        ),
    )

    finding_b_contra = Finding(
        finding_id="f_b_contra",
        paper_id="p_b_contra",
        paper_title="Study B Contra",
        evidence=span_b_contradict,
        finding_claim=span_b_contradict.text,
        direction=DirectionLabel.POSITIVE,
        conditions=ConditionSlots(
            intervention="Urban Tree Canopy Cover",
            comparator="Open Urban Baseline",
            outcome_metric="Daytime Air Temperature",
            dataset_setting="Dense Urban Centers",
            confidence=0.90,
        ),
    )

    finding_b_qual = Finding(
        finding_id="f_b_qual",
        paper_id="p_b_qual",
        paper_title="Study B Qual",
        evidence=span_b_qualify,
        finding_claim=span_b_qualify.text,
        direction=DirectionLabel.NO_DIFFERENCE,
        conditions=ConditionSlots(
            intervention="Urban Tree Canopy Cover",
            comparator="Open Urban Baseline",
            outcome_metric="Daytime Air Temperature",
            dataset_setting="Arid Desert Climates",
            confidence=0.90,
        ),
    )

    # 1. Direct contradiction
    pair_contra = FindingPair(pair_id="p_contra", finding_a=finding_a, finding_b=finding_b_contra)
    pred_contra = classifier.classify_pair(pair_contra)
    assert pred_contra.predicted_relation == RelationLabel.CONTRADICTS
    assert pred_contra.heuristic_score > 0.70

    # 2. Qualified disagreement due to climate difference
    pair_qual = FindingPair(pair_id="p_qual", finding_a=finding_a, finding_b=finding_b_qual)
    pred_qual = classifier.classify_pair(pair_qual)
    assert pred_qual.predicted_relation == RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT

    # 3. Incomparable conditions
    finding_c_incomp = Finding(
        finding_id="f_c",
        paper_id="p_c",
        paper_title="Study C",
        evidence=EvidenceSpan(
            text="Lithium-ion batteries experienced 20% capacity fade during thermal cycling.",
            confidence=0.9,
        ),
        finding_claim="Lithium-ion batteries experienced 20% capacity fade",
        direction=DirectionLabel.NEGATIVE,
        conditions=ConditionSlots(
            intervention="Thermal Cycling",
            comparator="Ambient Baseline",
            outcome_metric="Capacity Fade",
            dataset_setting="Lithium-Ion Cells",
            confidence=0.90,
        ),
    )
    pair_incomp = FindingPair(pair_id="p_incomp", finding_a=finding_a, finding_b=finding_c_incomp)
    pred_incomp = classifier.classify_pair(pair_incomp)
    assert pred_incomp.predicted_relation == RelationLabel.INCOMPARABLE_CONDITIONS


@pytest.mark.asyncio
async def test_no_synthetic_findings_on_empty_evidence():
    pipeline = ContradictionFinderPipeline()
    p_empty = Paper(
        paper_id="empty_1",
        title="Editorial Notes",
        abstract="The editors thank all reviewers for their contributions during the past year. No experimental data is reported.",
        source_provider="openalex",
    )
    findings = await pipeline.extract_findings_from_papers([p_empty])
    assert len(findings) == 0, (
        "No synthetic findings should be generated when abstract has no empirical evidence!"
    )
