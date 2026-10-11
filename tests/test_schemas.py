from scientific_contradiction_finder.schemas import (
    ConditionDifference,
    ConditionSlots,
    DirectionLabel,
    EvidenceSpan,
    Finding,
    FindingPair,
    Paper,
    Prediction,
    RelationLabel,
)


def test_paper_schema_valid():
    paper = Paper(
        paper_id="W123456",
        title="Impact of Metformin on Glycemic Control",
        abstract="In this double-blind trial, metformin significantly reduced HbA1c levels compared to placebo.",
        doi="10.1001/jama.2021.001",
        year=2021,
        source_provider="openalex",
    )
    assert paper.paper_id == "W123456"
    assert "metformin" in paper.abstract


def test_evidence_span_verification():
    source_text = "Metformin reduced HbA1c by 1.2% across diabetic cohorts."
    span = EvidenceSpan(text="Metformin reduced HbA1c by 1.2%", start_char=0, end_char=31)
    assert span.verify_against_source(source_text) is True

    fake_span = EvidenceSpan(
        text="Metformin showed no effect on glucose levels", start_char=0, end_char=44
    )
    assert fake_span.verify_against_source(source_text) is False


def test_condition_slots_defaults():
    conditions = ConditionSlots(
        intervention="Metformin 1000mg daily",
        outcome_metric="HbA1c reduction",
    )
    assert conditions.comparator == "Standard Baseline / Unspecified"
    assert conditions.dataset_setting == "General Scientific Setting"
    assert conditions.is_unknown is False


def test_finding_pair_and_prediction():
    span_a = EvidenceSpan(text="Metformin lowered HbA1c by 1.1%", start_char=0, end_char=31)
    cond_a = ConditionSlots(
        intervention="Metformin",
        comparator="Placebo",
        outcome_metric="HbA1c",
        dataset_setting="Type 2 Diabetes Cohort",
    )
    fa = Finding(
        finding_id="f1",
        paper_id="p1",
        paper_title="Study A",
        evidence=span_a,
        finding_claim="Metformin reduces HbA1c compared to placebo in T2D.",
        direction=DirectionLabel.POSITIVE,
        conditions=cond_a,
    )

    span_b = EvidenceSpan(
        text="Metformin did not alter HbA1c significantly", start_char=0, end_char=43
    )
    cond_b = ConditionSlots(
        intervention="Metformin",
        comparator="Placebo",
        outcome_metric="HbA1c",
        dataset_setting="Prediabetic Cohort",
    )
    fb = Finding(
        finding_id="f2",
        paper_id="p2",
        paper_title="Study B",
        evidence=span_b,
        finding_claim="Metformin does not significantly alter HbA1c in prediabetes.",
        direction=DirectionLabel.NO_DIFFERENCE,
        conditions=cond_b,
    )

    pair = FindingPair(
        pair_id="pair_001",
        finding_a=fa,
        finding_b=fb,
        ground_truth_relation=RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT,
        domain="clinical_medicine",
    )
    assert pair.ground_truth_relation == RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT

    diff = ConditionDifference(
        intervention_match=True,
        intervention_similarity=1.0,
        comparator_match=True,
        comparator_similarity=1.0,
        outcome_match=True,
        outcome_similarity=1.0,
        setting_match=False,
        setting_similarity=0.45,
        overall_compatibility=0.72,
        summary_of_differences="Divergent patient cohorts: T2D vs Prediabetes",
    )

    pred = Prediction(
        pair_id="pair_001",
        predicted_relation=RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT,
        confidence_score=0.88,
        condition_differences=diff,
        provenance_verified=True,
        explanation="Study A observed efficacy in T2D while Study B tested prediabetic individuals.",
    )
    assert pred.predicted_relation == RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT
    assert pred.provenance_verified is True
