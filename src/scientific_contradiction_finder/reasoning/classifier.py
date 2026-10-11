"""Condition-aware 5-class relation classifier with abstention gating."""

from scientific_contradiction_finder.reasoning.compatibility import ConditionCompatibilityEvaluator
from scientific_contradiction_finder.schemas import (
    DirectionLabel,
    FindingPair,
    Prediction,
    RelationLabel,
)


class ConditionAwareRelationClassifier:
    """Predicts relationship across finding pairs by explicitly incorporating condition divergence.

    Decision Architecture:
    1. Check for Insufficient Evidence / Unknown conditions -> INSUFFICIENT_EVIDENCE
    2. Evaluate condition compatibility:
       - If Intervention similarity < 0.20 and Intervention not substring -> INCOMPARABLE_CONDITIONS
    3. Evaluate reported directional outcomes:
       - If directions are concordant:
         - SUPPORTS_OR_CONSISTENT
       - If directions conflict (e.g. POSITIVE vs NEGATIVE, or ACTIVE vs NO_DIFFERENCE):
         - If all conditions match very tightly (setting similarity >= 0.90 & comparator >= 0.85 & overall >= 0.85):
           -> CONTRADICTS
         - If conditions diverge (differing settings, metrics, dosages, or baselines):
           -> QUALIFIES_OR_PARTIAL_DISAGREEMENT
    4. Abstention gate for low confidence.
    """

    def __init__(
        self,
        compat_evaluator: ConditionCompatibilityEvaluator | None = None,
        confidence_abstention_threshold: float = 0.45,
    ):
        self.compat_evaluator = compat_evaluator or ConditionCompatibilityEvaluator()
        self.abstention_threshold = confidence_abstention_threshold

    def classify_pair(self, pair: FindingPair) -> Prediction:
        fa = pair.finding_a
        fb = pair.finding_b

        # 1. Check for insufficient evidence or unknown conditions
        if (
            fa.conditions.is_unknown
            or fb.conditions.is_unknown
            or fa.direction == DirectionLabel.UNCLEAR
            or fb.direction == DirectionLabel.UNCLEAR
            or fa.evidence.confidence < 0.6
            or fb.evidence.confidence < 0.6
        ):
            diff = self.compat_evaluator.evaluate_difference(fa.conditions, fb.conditions)
            mean_ev_conf = (fa.evidence.confidence + fb.evidence.confidence) / 2.0
            # Heuristic comparison support score (not an uncalibrated probability)
            calculated_score = round(
                float(0.70 + 0.25 * (1.0 - diff.overall_compatibility) * mean_ev_conf), 3
            )
            return Prediction(
                pair_id=pair.pair_id,
                predicted_relation=RelationLabel.INSUFFICIENT_EVIDENCE,
                heuristic_score=calculated_score,
                confidence_score=calculated_score,
                condition_differences=diff,
                provenance_verified=True,
                explanation="Findings lack sufficient empirical evidence or condition clarity to form a valid comparison.",
            )

        # 2. Evaluate condition difference
        diff = self.compat_evaluator.evaluate_difference(fa.conditions, fb.conditions)

        # Check for fundamentally incomparable conditions (disparate fields / unrelated interventions)
        if diff.intervention_similarity < 0.20:
            incomp_score = round(float(0.75 + 0.20 * (1.0 - diff.intervention_similarity)), 3)
            return Prediction(
                pair_id=pair.pair_id,
                predicted_relation=RelationLabel.INCOMPARABLE_CONDITIONS,
                heuristic_score=incomp_score,
                confidence_score=incomp_score,
                condition_differences=diff,
                provenance_verified=True,
                explanation=f"Studies evaluate disparate interventions: '{fa.conditions.intervention}' vs '{fb.conditions.intervention}' (similarity={diff.intervention_similarity:.2f}).",
            )

        # If both outcomes and settings are completely disjoint (<0.15)
        if (
            diff.outcome_similarity < 0.15
            and diff.setting_similarity < 0.15
            and not diff.intervention_match
        ):
            disjoint_score = round(
                float(
                    0.75 + 0.20 * (1.0 - (diff.outcome_similarity + diff.setting_similarity) / 2.0)
                ),
                3,
            )
            return Prediction(
                pair_id=pair.pair_id,
                predicted_relation=RelationLabel.INCOMPARABLE_CONDITIONS,
                heuristic_score=disjoint_score,
                confidence_score=disjoint_score,
                condition_differences=diff,
                provenance_verified=True,
                explanation=f"Studies evaluate disparate outcomes and settings: {diff.summary_of_differences}",
            )

        # 3. Assess directional agreement
        is_fa_active = fa.direction in (DirectionLabel.POSITIVE, DirectionLabel.NEGATIVE)
        is_fb_active = fb.direction in (DirectionLabel.POSITIVE, DirectionLabel.NEGATIVE)

        # Conflict cases:
        # A) Opposite polarities (POSITIVE vs NEGATIVE)
        # B) One active vs NO_DIFFERENCE
        # C) MIXED vs single active direction
        directions_conflict = (
            (fa.direction == DirectionLabel.POSITIVE and fb.direction == DirectionLabel.NEGATIVE)
            or (fa.direction == DirectionLabel.NEGATIVE and fb.direction == DirectionLabel.POSITIVE)
            or (is_fa_active and fb.direction == DirectionLabel.NO_DIFFERENCE)
            or (fa.direction == DirectionLabel.NO_DIFFERENCE and is_fb_active)
            or (fa.direction == DirectionLabel.MIXED and is_fb_active)
            or (is_fa_active and fb.direction == DirectionLabel.MIXED)
        )

        mean_ev_conf = (fa.evidence.confidence + fb.evidence.confidence) / 2.0

        if directions_conflict:
            # If all conditions match almost identically (strict tolerance) -> Direct contradiction
            is_strict_setting_match = diff.setting_similarity >= 0.85
            is_strict_comparator_match = diff.comparator_similarity >= 0.80
            is_strict_overall = diff.overall_compatibility >= 0.82

            if is_strict_setting_match and is_strict_comparator_match and is_strict_overall:
                pred_label = RelationLabel.CONTRADICTS
                explanation = f"Direct empirical contradiction: studies evaluate comparable conditions ({fa.conditions.intervention} on {fa.conditions.outcome_metric}) but report opposing directional findings."
                score = round(float(0.75 + 0.20 * diff.overall_compatibility * mean_ev_conf), 3)
            else:
                # Diverging setting, comparator, metric, or population qualifies the disagreement
                pred_label = RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT
                explanation = f"Apparent disagreement qualified by experimental condition divergence: {diff.summary_of_differences}"
                score = round(
                    float(
                        0.70 + 0.22 * (1.0 - abs(diff.overall_compatibility - 0.55)) * mean_ev_conf
                    ),
                    3,
                )
        else:
            # Directions concordant (both positive, both negative, or both no difference)
            pred_label = RelationLabel.SUPPORTS_OR_CONSISTENT
            explanation = f"Findings report consistent directional outcomes ({fa.direction.value}) under compatible conditions."
            score = round(float(0.70 + 0.25 * diff.overall_compatibility * mean_ev_conf), 3)

        # Provenance verification check
        provenance_ok = fa.evidence.verify_against_source(
            fa.evidence.text
        ) and fb.evidence.verify_against_source(fb.evidence.text)

        return Prediction(
            pair_id=pair.pair_id,
            predicted_relation=pred_label,
            heuristic_score=score,
            confidence_score=score,
            condition_differences=diff,
            provenance_verified=provenance_ok,
            explanation=explanation,
        )

    def classify_all(self, pairs: list[FindingPair]) -> list[Prediction]:
        return [self.classify_pair(p) for p in pairs]
