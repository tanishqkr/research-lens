from scientific_contradiction_finder.reasoning.classifier import ConditionAwareRelationClassifier
from scientific_contradiction_finder.schemas import (
    ConditionSlots,
    DirectionLabel,
    EvidenceSpan,
    Finding,
    FindingPair,
)


def test_confidence_not_constant():
    clf = ConditionAwareRelationClassifier()

    # Pair 1: High quality evidence, identical conditions
    f1 = Finding(
        finding_id="1",
        paper_id="p1",
        paper_title="T1",
        evidence=EvidenceSpan(text="Treatment A reduced mortality significantly", confidence=0.95),
        finding_claim="Treatment A reduced mortality",
        direction=DirectionLabel.NEGATIVE,
        conditions=ConditionSlots(
            intervention="Treatment A",
            comparator="Placebo",
            outcome_metric="Mortality",
            dataset_setting="ICU cohort",
        ),
    )
    f2 = Finding(
        finding_id="2",
        paper_id="p2",
        paper_title="T2",
        evidence=EvidenceSpan(text="Treatment A reduced mortality across trials", confidence=0.90),
        finding_claim="Treatment A reduced mortality",
        direction=DirectionLabel.NEGATIVE,
        conditions=ConditionSlots(
            intervention="Treatment A",
            comparator="Placebo",
            outcome_metric="Mortality",
            dataset_setting="ICU cohort",
        ),
    )
    pred1 = clf.classify_pair(FindingPair(pair_id="p1", finding_a=f1, finding_b=f2))

    # Pair 2: Lower evidence confidence, qualified disagreement
    f3 = Finding(
        finding_id="3",
        paper_id="p3",
        paper_title="T3",
        evidence=EvidenceSpan(
            text="Treatment A showed no mortality effect in mild outpatients", confidence=0.65
        ),
        finding_claim="Treatment A did not reduce mortality",
        direction=DirectionLabel.NO_DIFFERENCE,
        conditions=ConditionSlots(
            intervention="Treatment A",
            comparator="Placebo",
            outcome_metric="Mortality",
            dataset_setting="Mild outpatient cohort",
        ),
    )
    pred2 = clf.classify_pair(FindingPair(pair_id="p2", finding_a=f1, finding_b=f3))

    print(f"Pred 1 Conf: {pred1.confidence_score}, Pred 2 Conf: {pred2.confidence_score}")
    assert pred1.confidence_score != pred2.confidence_score
    assert 0.0 <= pred1.confidence_score <= 1.0
    assert 0.0 <= pred2.confidence_score <= 1.0


if __name__ == "__main__":
    test_confidence_not_constant()
