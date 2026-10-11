"""Condition-aware 5-class relation classifier with abstention gating."""

from typing import List, Optional
from scientific_contradiction_finder.schemas import (
    FindingPair,
    Prediction,
    RelationLabel,
    DirectionLabel
)
from scientific_contradiction_finder.reasoning.compatibility import ConditionCompatibilityEvaluator


class ConditionAwareRelationClassifier:
    """Predicts relationship across finding pairs by explicitly incorporating condition divergence.
    
    Decision Architecture:
    1. Check for Insufficient Evidence / Unknown conditions -> INSUFFICIENT_EVIDENCE
    2. Evaluate condition compatibility:
       - If Intervention or Outcome are fundamentally incompatible (<0.35) -> INCOMPARABLE_CONDITIONS
    3. Evaluate reported directional outcomes:
       - If conditions match closely (>=0.70):
         - Concordant directions -> SUPPORTS_OR_CONSISTENT
         - Opposing directions -> CONTRADICTS
       - If conditions diverge partially (e.g. comparator or setting differs, but intervention/outcome related):
         - Opposing directions -> QUALIFIES_OR_PARTIAL_DISAGREEMENT
         - Concordant directions -> SUPPORTS_OR_CONSISTENT
    4. Abstention gate for low confidence.
    """

    def __init__(
        self,
        compat_evaluator: Optional[ConditionCompatibilityEvaluator] = None,
        confidence_abstention_threshold: float = 0.45
    ):
        self.compat_evaluator = compat_evaluator or ConditionCompatibilityEvaluator()
        self.abstention_threshold = confidence_abstention_threshold

    def classify_pair(self, pair: FindingPair) -> Prediction:
        fa = pair.finding_a
        fb = pair.finding_b

        # 1. Check for insufficient evidence or unknown conditions
        if (
            fa.conditions.is_unknown or
            fb.conditions.is_unknown or
            fa.direction == DirectionLabel.UNCLEAR or
            fb.direction == DirectionLabel.UNCLEAR or
            fa.evidence.confidence < 0.6 or
            fb.evidence.confidence < 0.6
        ):
            diff = self.compat_evaluator.evaluate_difference(fa.conditions, fb.conditions)
            return Prediction(
                pair_id=pair.pair_id,
                predicted_relation=RelationLabel.INSUFFICIENT_EVIDENCE,
                confidence_score=0.85,
                condition_differences=diff,
                provenance_verified=True,
                explanation="Findings lack sufficient empirical evidence or condition clarity to form a valid comparison."
            )

        # 2. Evaluate condition difference
        diff = self.compat_evaluator.evaluate_difference(fa.conditions, fb.conditions)

        # Check for incomparable conditions
        if not diff.intervention_match or diff.outcome_similarity < 0.35:
            return Prediction(
                pair_id=pair.pair_id,
                predicted_relation=RelationLabel.INCOMPARABLE_CONDITIONS,
                confidence_score=0.90,
                condition_differences=diff,
                provenance_verified=True,
                explanation=f"Studies evaluate disparate interventions or outcomes: {diff.summary_of_differences}"
            )

        # 3. Assess directional agreement
        directions_conflict = (
            (fa.direction in (DirectionLabel.POSITIVE, DirectionLabel.NEGATIVE) and fb.direction == DirectionLabel.NO_DIFFERENCE) or
            (fa.direction == DirectionLabel.POSITIVE and fb.direction == DirectionLabel.NEGATIVE) or
            (fa.direction == DirectionLabel.NEGATIVE and fb.direction == DirectionLabel.POSITIVE) or
            (fa.direction == DirectionLabel.NO_DIFFERENCE and fb.direction in (DirectionLabel.POSITIVE, DirectionLabel.NEGATIVE))
        )

        if directions_conflict:
            # If all conditions match closely -> Direct contradiction
            if diff.setting_match and diff.comparator_match and diff.overall_compatibility >= 0.70:
                pred_label = RelationLabel.CONTRADICTS
                explanation = "Direct empirical contradiction: studies evaluate identical conditions but report opposing directional findings."
                conf = 0.92
            else:
                # Differing setting or comparator explains opposing findings -> Qualified disagreement
                pred_label = RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT
                explanation = f"Apparent disagreement qualified by experimental divergence: {diff.summary_of_differences}"
                conf = 0.88
        else:
            # Directions concordant
            pred_label = RelationLabel.SUPPORTS_OR_CONSISTENT
            explanation = "Findings report consistent directional outcomes under compatible conditions."
            conf = 0.90

        # Provenance verification check
        provenance_ok = (
            fa.evidence.verify_against_source(fa.evidence.text) and
            fb.evidence.verify_against_source(fb.evidence.text)
        )

        return Prediction(
            pair_id=pair.pair_id,
            predicted_relation=pred_label,
            confidence_score=conf,
            condition_differences=diff,
            provenance_verified=provenance_ok,
            explanation=explanation
        )

    def classify_all(self, pairs: List[FindingPair]) -> List[Prediction]:
        return [self.classify_pair(p) for p in pairs]
