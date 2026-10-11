import pytest
from scientific_contradiction_finder.schemas import (
    Finding,
    FindingPair,
    EvidenceSpan,
    ConditionSlots,
    DirectionLabel,
    RelationLabel
)
from scientific_contradiction_finder.reasoning.compatibility import ConditionCompatibilityEvaluator
from scientific_contradiction_finder.reasoning.classifier import ConditionAwareRelationClassifier


def test_compatibility_evaluator():
    evaluator = ConditionCompatibilityEvaluator()

    c1 = ConditionSlots(
        intervention="Metformin 1000mg",
        comparator="Placebo",
        outcome_metric="HbA1c",
        dataset_setting="Type 2 Diabetes"
    )
    c2 = ConditionSlots(
        intervention="Metformin 1000mg",
        comparator="Placebo",
        outcome_metric="HbA1c",
        dataset_setting="Type 2 Diabetes"
    )
    diff = evaluator.evaluate_difference(c1, c2)
    assert diff.intervention_match is True
    assert diff.overall_compatibility >= 0.8


def test_classifier_distinguishes_direct_vs_qualified():
    classifier = ConditionAwareRelationClassifier()

    # Direct contradiction: identical setting, opposite results
    fa_contra = Finding(
        finding_id="1", paper_id="p1", paper_title="T1",
        evidence=EvidenceSpan(text="Drug X reduced mortality significantly"),
        finding_claim="Drug X reduced mortality significantly",
        direction=DirectionLabel.NEGATIVE,
        conditions=ConditionSlots(intervention="Drug X", comparator="Placebo", outcome_metric="Mortality", dataset_setting="ICU Patients")
    )
    fb_contra = Finding(
        finding_id="2", paper_id="p2", paper_title="T2",
        evidence=EvidenceSpan(text="Drug X did not reduce mortality"),
        finding_claim="Drug X did not reduce mortality",
        direction=DirectionLabel.NO_DIFFERENCE,
        conditions=ConditionSlots(intervention="Drug X", comparator="Placebo", outcome_metric="Mortality", dataset_setting="ICU Patients")
    )
    pair_contra = FindingPair(pair_id="c1", finding_a=fa_contra, finding_b=fb_contra)
    pred_contra = classifier.classify_pair(pair_contra)
    assert pred_contra.predicted_relation == RelationLabel.CONTRADICTS

    # Qualified disagreement: different settings explains opposing results
    fb_qual = Finding(
        finding_id="3", paper_id="p3", paper_title="T3",
        evidence=EvidenceSpan(text="Drug X did not reduce mortality in outpatient cohort"),
        finding_claim="Drug X did not reduce mortality in outpatient cohort",
        direction=DirectionLabel.NO_DIFFERENCE,
        conditions=ConditionSlots(intervention="Drug X", comparator="Placebo", outcome_metric="Mortality", dataset_setting="Outpatient Mild Cohort")
    )
    pair_qual = FindingPair(pair_id="q1", finding_a=fa_contra, finding_b=fb_qual)
    pred_qual = classifier.classify_pair(pair_qual)
    assert pred_qual.predicted_relation == RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT
