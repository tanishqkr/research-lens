"""Comprehensive regression tests for directional conflict classification.

Verifies:
1. Symmetric conflict classification under both finding orders (fa vs fb and fb vs fa).
2. NO_DIFFERENCE vs POSITIVE and NO_DIFFERENCE vs NEGATIVE in both orders.
3. MIXED vs POSITIVE and MIXED vs NEGATIVE in both orders.
4. POSITIVE vs NEGATIVE in both orders.
5. Concordant / compatible findings (POSITIVE vs POSITIVE, NEGATIVE vs NEGATIVE, NO_DIFF vs NO_DIFF).
6. Safe abstention on low confidence, unknown conditions, or UNCLEAR direction.
7. Endpoint domain partitioning (disjoint endpoints -> INCOMPARABLE_CONDITIONS).
8. Causal design guardrail (Observational vs Interventional -> QUALIFIES_OR_PARTIAL_DISAGREEMENT).
"""

import pytest

from scientific_contradiction_finder.reasoning.classifier import ConditionAwareRelationClassifier
from scientific_contradiction_finder.schemas import (
    ConditionSlots,
    DirectionLabel,
    EvidenceSpan,
    Finding,
    FindingPair,
    RelationLabel,
    StudyDesignType,
)


def _make_finding(
    fid: str,
    direction: DirectionLabel,
    claim: str = "Intervention outcome claim",
    intervention: str = "Drug X",
    comparator: str = "Placebo",
    outcome: str = "All-Cause Mortality",
    setting: str = "Adult ICU Patients",
    confidence: float = 0.9,
    is_unknown: bool = False,
    study_type: StudyDesignType = StudyDesignType.INTERVENTIONAL_TRIAL,
    duration: str = "12 months",
) -> Finding:
    return Finding(
        finding_id=fid,
        paper_id=f"paper_{fid}",
        paper_title=f"Title {fid}",
        evidence=EvidenceSpan(
            text=claim,
            start_char=0,
            end_char=len(claim),
            confidence=confidence,
        ),
        finding_claim=claim,
        direction=direction,
        conditions=ConditionSlots(
            intervention=intervention,
            comparator=comparator,
            outcome_metric=outcome,
            dataset_setting=setting,
            confidence=confidence,
            is_unknown=is_unknown,
            study_type=study_type,
            follow_up_duration=duration,
        ),
    )


@pytest.fixture
def classifier():
    return ConditionAwareRelationClassifier()


@pytest.mark.parametrize(
    "dir_a,dir_b,expected_relation",
    [
        # POSITIVE vs NEGATIVE (direct contradiction)
        (DirectionLabel.POSITIVE, DirectionLabel.NEGATIVE, RelationLabel.CONTRADICTS),
        (DirectionLabel.NEGATIVE, DirectionLabel.POSITIVE, RelationLabel.CONTRADICTS),
        # Active vs NO_DIFFERENCE (both orders)
        (DirectionLabel.POSITIVE, DirectionLabel.NO_DIFFERENCE, RelationLabel.CONTRADICTS),
        (DirectionLabel.NO_DIFFERENCE, DirectionLabel.POSITIVE, RelationLabel.CONTRADICTS),
        (DirectionLabel.NEGATIVE, DirectionLabel.NO_DIFFERENCE, RelationLabel.CONTRADICTS),
        (DirectionLabel.NO_DIFFERENCE, DirectionLabel.NEGATIVE, RelationLabel.CONTRADICTS),
        # Active vs MIXED (both orders)
        (DirectionLabel.POSITIVE, DirectionLabel.MIXED, RelationLabel.CONTRADICTS),
        (DirectionLabel.MIXED, DirectionLabel.POSITIVE, RelationLabel.CONTRADICTS),
        (DirectionLabel.NEGATIVE, DirectionLabel.MIXED, RelationLabel.CONTRADICTS),
        (DirectionLabel.MIXED, DirectionLabel.NEGATIVE, RelationLabel.CONTRADICTS),
        # Concordant / compatible
        (DirectionLabel.POSITIVE, DirectionLabel.POSITIVE, RelationLabel.SUPPORTS_OR_CONSISTENT),
        (DirectionLabel.NEGATIVE, DirectionLabel.NEGATIVE, RelationLabel.SUPPORTS_OR_CONSISTENT),
        (DirectionLabel.NO_DIFFERENCE, DirectionLabel.NO_DIFFERENCE, RelationLabel.SUPPORTS_OR_CONSISTENT),
    ],
)
def test_directional_conflict_symmetry_and_polarities(classifier, dir_a, dir_b, expected_relation):
    fa = _make_finding("a", dir_a, claim="Finding A statement")
    fb = _make_finding("b", dir_b, claim="Finding B statement")

    pair = FindingPair(pair_id="p1", finding_a=fa, finding_b=fb)
    pred = classifier.classify_pair(pair)

    assert pred.predicted_relation == expected_relation, (
        f"Failed for order ({dir_a.value}, {dir_b.value}): expected {expected_relation.value}, "
        f"got {pred.predicted_relation.value}"
    )


def test_no_difference_order_invariance(classifier):
    """Specifically verifies the fixed bug where fa=NO_DIFF and fb=ACTIVE was previously broken."""
    fa_null = _make_finding("null", DirectionLabel.NO_DIFFERENCE, claim="Drug X showed no significant effect on mortality")
    fb_active = _make_finding("active", DirectionLabel.NEGATIVE, claim="Drug X significantly reduced mortality")

    # Order 1: Null first, Active second
    pair_1 = FindingPair(pair_id="order1", finding_a=fa_null, finding_b=fb_active)
    pred_1 = classifier.classify_pair(pair_1)
    assert pred_1.predicted_relation == RelationLabel.CONTRADICTS

    # Order 2: Active first, Null second
    pair_2 = FindingPair(pair_id="order2", finding_a=fb_active, finding_b=fa_null)
    pred_2 = classifier.classify_pair(pair_2)
    assert pred_2.predicted_relation == RelationLabel.CONTRADICTS


def test_mixed_order_invariance(classifier):
    """Verifies that MIXED vs ACTIVE is detected symmetrically in both finding orders."""
    fa_mixed = _make_finding("mixed", DirectionLabel.MIXED, claim="Drug X showed mixed results across cohorts")
    fb_pos = _make_finding("pos", DirectionLabel.POSITIVE, claim="Drug X showed strong positive efficacy")

    # Order 1: Mixed first
    pair_1 = FindingPair(pair_id="m1", finding_a=fa_mixed, finding_b=fb_pos)
    pred_1 = classifier.classify_pair(pair_1)
    assert pred_1.predicted_relation == RelationLabel.CONTRADICTS

    # Order 2: Positive first
    pair_2 = FindingPair(pair_id="m2", finding_a=fb_pos, finding_b=fa_mixed)
    pred_2 = classifier.classify_pair(pair_2)
    assert pred_2.predicted_relation == RelationLabel.CONTRADICTS


def test_safe_abstention_guardrails(classifier):
    """Low evidence confidence, unknown condition flag, or UNCLEAR direction must trigger INSUFFICIENT_EVIDENCE."""
    fa_good = _make_finding("g", DirectionLabel.POSITIVE)

    # 1. Low evidence confidence
    fa_low_conf = _make_finding("low", DirectionLabel.POSITIVE, confidence=0.4)
    pred_low = classifier.classify_pair(FindingPair(pair_id="ab1", finding_a=fa_low_conf, finding_b=fa_good))
    assert pred_low.predicted_relation == RelationLabel.INSUFFICIENT_EVIDENCE

    # 2. Unknown conditions flag
    fa_unk = _make_finding("unk", DirectionLabel.NEGATIVE, is_unknown=True)
    pred_unk = classifier.classify_pair(FindingPair(pair_id="ab2", finding_a=fa_good, finding_b=fa_unk))
    assert pred_unk.predicted_relation == RelationLabel.INSUFFICIENT_EVIDENCE

    # 3. UNCLEAR direction
    fa_unclear = _make_finding("unc", DirectionLabel.UNCLEAR)
    pred_unc = classifier.classify_pair(FindingPair(pair_id="ab3", finding_a=fa_good, finding_b=fa_unclear))
    assert pred_unc.predicted_relation == RelationLabel.INSUFFICIENT_EVIDENCE


def test_endpoint_partitioning_guardrail(classifier):
    """Disparate target clinical endpoints must be partitioned as INCOMPARABLE_CONDITIONS."""
    fa_mortality = _make_finding("m", DirectionLabel.NEGATIVE, outcome="All-Cause Mortality")
    fb_cancer = _make_finding("c", DirectionLabel.NEGATIVE, outcome="Cervical Cancer")

    pred = classifier.classify_pair(FindingPair(pair_id="incomp1", finding_a=fa_mortality, finding_b=fb_cancer))
    assert pred.predicted_relation == RelationLabel.INCOMPARABLE_CONDITIONS
    assert "disparate outcome endpoints" in pred.explanation.lower()


def test_causal_design_divergence_guardrail(classifier):
    """Observational cohort vs Randomized trial with opposing directions must be QUALIFIED, not direct CONTRADICTS."""
    fa_obs = _make_finding(
        "obs",
        DirectionLabel.POSITIVE,
        claim="Observational cohort shows benefit",
        study_type=StudyDesignType.OBSERVATIONAL_STUDY,
    )
    fb_rct = _make_finding(
        "rct",
        DirectionLabel.NEGATIVE,
        claim="Randomized controlled trial shows no benefit or harm",
        study_type=StudyDesignType.INTERVENTIONAL_TRIAL,
    )

    pred = classifier.classify_pair(FindingPair(pair_id="qual1", finding_a=fa_obs, finding_b=fb_rct))
    assert pred.predicted_relation == RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT
    assert "study design divergence" in pred.explanation.lower()


def test_weight_loss_contradiction_not_falsely_concordant(classifier):
    """When one study claims weight loss (POSITIVE) and another claims reduced weight loss / inferiority (NEGATIVE),
    they must be classified as CONTRADICTS, not falsely suppressed as concordant.
    """
    fa = _make_finding(
        "wl1",
        DirectionLabel.POSITIVE,
        claim="LCDs produced significant weight loss of 5 kg.",
        outcome="Weight Loss",
    )
    fb = _make_finding(
        "wl2",
        DirectionLabel.NEGATIVE,
        claim="LCDs reduced weight loss compared to controls.",
        outcome="Weight Loss",
    )
    pred = classifier.classify_pair(FindingPair(pair_id="wl_conflict", finding_a=fa, finding_b=fb))
    assert pred.predicted_relation == RelationLabel.CONTRADICTS


def test_anatomical_outcome_divergence_gating(classifier):
    """Opposing directions on divergent anatomical endpoints (e.g. ICH vs GI bleeding) must be QUALIFIED."""
    fa = _make_finding(
        "ich",
        DirectionLabel.NEGATIVE,
        claim="DOACs significantly reduced intracranial hemorrhage.",
        outcome="Intracranial Hemorrhage Rate",
        intervention="Direct Oral Anticoagulants",
        comparator="Warfarin",
    )
    fb = _make_finding(
        "gib",
        DirectionLabel.POSITIVE,
        claim="DOACs significantly increased gastrointestinal bleeding.",
        outcome="Gastrointestinal Bleeding Rate",
        intervention="Direct Oral Anticoagulants",
        comparator="Warfarin",
    )
    pred = classifier.classify_pair(FindingPair(pair_id="bleed_div", finding_a=fa, finding_b=fb))
    assert pred.predicted_relation == RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT
    assert "outcome metric differs" in pred.explanation.lower() or "condition divergence" in pred.explanation.lower()


def test_disparate_endpoint_domain_incomparability(classifier):
    """Statin on mortality vs Statin on cataract extraction surgery must be INCOMPARABLE_CONDITIONS."""
    fa = _make_finding(
        "mort",
        DirectionLabel.NEGATIVE,
        claim="Simvastatin reduced all-cause mortality.",
        outcome="All-Cause Mortality",
        intervention="Simvastatin",
    )
    fb = _make_finding(
        "cat",
        DirectionLabel.NO_DIFFERENCE,
        claim="Statin therapy showed no effect on incident cataract surgery.",
        outcome="Incident Cataract Extraction Surgery",
        intervention="Statin",
    )
    pred = classifier.classify_pair(FindingPair(pair_id="disparate_incomp", finding_a=fa, finding_b=fb))
    assert pred.predicted_relation == RelationLabel.INCOMPARABLE_CONDITIONS
