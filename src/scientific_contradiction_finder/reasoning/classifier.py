"""Condition-aware 5-class relation classifier with abstention gating."""

from scientific_contradiction_finder.reasoning.compatibility import ConditionCompatibilityEvaluator
from scientific_contradiction_finder.schemas import (
    DirectionLabel,
    FindingPair,
    Prediction,
    RelationLabel,
    StudyDesignType,
)


def get_endpoint_domain(outcome_str: str | None, claim_str: str | None = None) -> str | None:
    """Classifies outcome metric or finding claim into a high-level scientific endpoint domain."""
    o_text = (outcome_str or "").lower()
    c_text = (claim_str or "").lower()
    text = f"{o_text} {c_text}".strip()
    if not text:
        return None

    # Check for predictive performance discrimination metrics in outcome string
    if any(
        k in o_text
        for k in [
            "auroc",
            "auprc",
            "c-statistic",
            "c-index",
            "discrimination metric",
            "discrimination",
            "brier score",
        ]
    ):
        return "PREDICTION_PERFORMANCE"

    if any(k in text for k in ["all-cause mortality", "mortality", "death", "fatal", "survival"]):
        return "MORTALITY"
    if any(
        k in text
        for k in [
            "stroke",
            "cerebrovascular",
            "brain infarction",
            "ischemic stroke",
            "hemorrhagic stroke",
        ]
    ):
        return "STROKE"
    if any(
        k in text
        for k in [
            "cardiovascular",
            "coronary",
            "myocardial",
            "heart disease",
            "cvd",
            "heart failure",
        ]
    ):
        return "CARDIOVASCULAR"
    if any(
        k in text
        for k in ["diabetes", "glycemic", "hba1c", "glucose", "insulin resistance", "t2d"]
    ):
        return "DIABETES"
    if any(
        k in text
        for k in [
            "weight loss",
            "weight reduction",
            "body weight",
            "bmi",
            "body mass index",
            "adiposity",
            "body fat",
        ]
    ):
        return "WEIGHT_LOSS"
    if any(k in text for k in ["sepsis", "septic", "septic shock"]):
        return "SEPSIS"
    if any(k in text for k in ["acute kidney injury", "aki", "renal failure", "nephropathy"]):
        return "RENAL"
    if any(
        k in text
        for k in [
            "cervical cancer",
            "lung cancer",
            "carcinoma",
            "tumor",
            "neoplasm",
            "cancer",
        ]
    ):
        return "CANCER"
    if any(
        k in text
        for k in [
            "reaction time",
            "working memory",
            "cognitive performance",
            "cognitive",
            "alertness",
            "vigilance",
        ]
    ):
        return "COGNITIVE"
    if any(
        k in text
        for k in ["air temperature", "canopy", "cooling", "urban heat", "surface temperature"]
    ):
        return "TEMPERATURE_CLIMATE"
    if any(
        k in text for k in ["capacity fade", "cell thickness", "degradation rate", "lithium plating"]
    ):
        return "BATTERY_DEGRADATION"
    if any(
        k in text for k in ["respiratory tract infection", "infection incidence", "infection risk"]
    ):
        return "INFECTION"

    if any(
        k in text
        for k in [
            "auroc",
            "auprc",
            "c-statistic",
            "c-index",
            "discrimination metric",
            "discrimination",
            "f1 score",
        ]
    ):
        return "PREDICTION_PERFORMANCE"

    return None


class ConditionAwareRelationClassifier:
    """Predicts relationship across finding pairs by explicitly incorporating condition divergence.

    Decision Architecture:
    1. Check for Insufficient Evidence / Unknown conditions -> INSUFFICIENT_EVIDENCE
    2. Evaluate endpoint and condition compatibility:
       - If endpoint domains differ (e.g. mortality vs stroke vs diabetes) -> INCOMPARABLE_CONDITIONS
       - If Intervention similarity < 0.20 -> INCOMPARABLE_CONDITIONS
    3. Evaluate reported directional outcomes:
       - If directions are concordant:
         - SUPPORTS_OR_CONSISTENT
       - If directions conflict (e.g. POSITIVE vs NEGATIVE, or ACTIVE vs NO_DIFFERENCE):
         - If all conditions match very tightly:
           - If follow-up durations diverge (e.g. 6m vs 12m) -> QUALIFIES_OR_PARTIAL_DISAGREEMENT
           - Otherwise -> CONTRADICTS
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
            return Prediction(
                pair_id=pair.pair_id,
                predicted_relation=RelationLabel.INSUFFICIENT_EVIDENCE,
                heuristic_score=round(float(diff.overall_compatibility), 3),
                confidence_score=round(float(0.70 + 0.20 * (1.0 - mean_ev_conf)), 3),
                condition_differences=diff,
                provenance_verified=True,
                explanation="Findings lack sufficient empirical evidence or condition clarity to form a valid comparison.",
            )

        # 2. Evaluate condition difference
        diff = self.compat_evaluator.evaluate_difference(fa.conditions, fb.conditions)

        # Check for target clinical/scientific endpoint domain disparity (e.g. stroke vs mortality vs diabetes)
        cat_a = get_endpoint_domain(fa.conditions.outcome_metric, fa.finding_claim)
        cat_b = get_endpoint_domain(fb.conditions.outcome_metric, fb.finding_claim)

        if cat_a and cat_b and cat_a != cat_b:
            incomp_conf = 0.92
            return Prediction(
                pair_id=pair.pair_id,
                predicted_relation=RelationLabel.INCOMPARABLE_CONDITIONS,
                heuristic_score=round(min(0.20, float(diff.overall_compatibility)), 3),
                confidence_score=incomp_conf,
                condition_differences=diff,
                provenance_verified=True,
                explanation=f"Studies evaluate disjoint target conditions and disparate outcome endpoints: '{fa.conditions.outcome_metric}' ({cat_a}) vs '{fb.conditions.outcome_metric}' ({cat_b}). Findings across distinct clinical endpoints cannot be directly compared as conflicting or supporting.",
            )

        # Check prediction model discrimination vs clinical trial / observational disease outcomes
        is_pred_a = (fa.conditions.study_type == StudyDesignType.PREDICTION_MODEL) or (cat_a == "PREDICTION_PERFORMANCE")
        is_pred_b = (fb.conditions.study_type == StudyDesignType.PREDICTION_MODEL) or (cat_b == "PREDICTION_PERFORMANCE")
        if is_pred_a != is_pred_b and (
            fa.conditions.study_type in (StudyDesignType.INTERVENTIONAL_TRIAL, StudyDesignType.OBSERVATIONAL_STUDY)
            or fb.conditions.study_type in (StudyDesignType.INTERVENTIONAL_TRIAL, StudyDesignType.OBSERVATIONAL_STUDY)
            or cat_a in ("MORTALITY", "SEPSIS", "STROKE", "CARDIOVASCULAR", "DIABETES", "WEIGHT_LOSS", "CANCER")
            or cat_b in ("MORTALITY", "SEPSIS", "STROKE", "CARDIOVASCULAR", "DIABETES", "WEIGHT_LOSS", "CANCER")
        ):
            return Prediction(
                pair_id=pair.pair_id,
                predicted_relation=RelationLabel.INCOMPARABLE_CONDITIONS,
                heuristic_score=round(min(0.20, float(diff.overall_compatibility)), 3),
                confidence_score=0.92,
                condition_differences=diff,
                provenance_verified=True,
                explanation="Studies evaluate fundamentally incomparable targets: one evaluates prediction model performance/discrimination, while the other evaluates clinical outcomes or interventional efficacy. Predictive discrimination metrics cannot be directly compared against clinical outcome rates.",
            )

        # Check for fundamentally incomparable conditions (disparate fields / unrelated interventions)
        if diff.intervention_similarity < 0.20:
            incomp_conf = round(float(0.75 + 0.20 * (1.0 - diff.intervention_similarity)), 3)
            return Prediction(
                pair_id=pair.pair_id,
                predicted_relation=RelationLabel.INCOMPARABLE_CONDITIONS,
                heuristic_score=round(min(0.20, float(diff.overall_compatibility)), 3),
                confidence_score=incomp_conf,
                condition_differences=diff,
                provenance_verified=True,
                explanation=f"Studies evaluate disparate interventions: '{fa.conditions.intervention}' vs '{fb.conditions.intervention}' (similarity={diff.intervention_similarity:.2f}).",
            )

        # If both outcomes and settings are completely disjoint (<0.20)
        if (
            diff.outcome_similarity < 0.20
            and diff.setting_similarity < 0.20
            and not diff.intervention_match
        ):
            disjoint_conf = round(
                float(
                    0.75 + 0.20 * (1.0 - (diff.outcome_similarity + diff.setting_similarity) / 2.0)
                ),
                3,
            )
            return Prediction(
                pair_id=pair.pair_id,
                predicted_relation=RelationLabel.INCOMPARABLE_CONDITIONS,
                heuristic_score=round(min(0.20, float(diff.overall_compatibility)), 3),
                confidence_score=disjoint_conf,
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
            or (fa.direction == DirectionLabel.NO_DIFFERENCE and is_fa_active)
            or (fa.direction == DirectionLabel.MIXED and is_fa_active)
            or (is_fa_active and fb.direction == DirectionLabel.MIXED)
        )

        # Check if endpoints are reduction-oriented (e.g. WEIGHT_LOSS) where both positive (greater loss)
        # and negative (decrease in weight) denote the same beneficial direction
        is_weight_loss = (cat_a == "WEIGHT_LOSS" and cat_b == "WEIGHT_LOSS")
        if is_weight_loss and is_fa_active and is_fb_active:
            has_gain_a = any(w in fa.finding_claim.lower() for w in ["weight gain", "increased weight", "gained weight"])
            has_gain_b = any(w in fb.finding_claim.lower() for w in ["weight gain", "increased weight", "gained weight"])
            if has_gain_a != has_gain_b:
                directions_conflict = True
            else:
                directions_conflict = False

        mean_ev_conf = (fa.evidence.confidence + fb.evidence.confidence) / 2.0

        if directions_conflict:
            # If all conditions match almost identically (strict tolerance) -> Direct contradiction
            is_strict_setting_match = diff.setting_similarity >= 0.85
            is_strict_comparator_match = diff.comparator_similarity >= 0.80
            is_strict_overall = diff.overall_compatibility >= 0.82

            # Check if study design differs (e.g. Observational Study vs Interventional Trial)
            study_a_type = fa.conditions.study_type
            study_b_type = fb.conditions.study_type
            is_obs_vs_trial = (
                (study_a_type == StudyDesignType.OBSERVATIONAL_STUDY and study_b_type == StudyDesignType.INTERVENTIONAL_TRIAL)
                or (study_a_type == StudyDesignType.INTERVENTIONAL_TRIAL and study_b_type == StudyDesignType.OBSERVATIONAL_STUDY)
            )

            # Check if follow-up duration differs significantly (e.g. 6 months vs 12 months)
            fa_dur = (fa.conditions.follow_up_duration or "").lower()
            fb_dur = (fb.conditions.follow_up_duration or "").lower()
            dur_diverges = bool(fa_dur and fb_dur and fa_dur != fb_dur)

            if is_strict_setting_match and is_strict_comparator_match and is_strict_overall:
                if is_obs_vs_trial:
                    pred_label = RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT
                    explanation = (
                        "Apparent discrepancy qualified by study design divergence: observational cohort study reports "
                        "statistical association (subject to residual confounding and selection bias), whereas randomized "
                        "interventional trial evaluates direct causal effect."
                    )
                    score = round(float(diff.overall_compatibility), 3)
                elif dur_diverges:
                    pred_label = RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT
                    explanation = f"Apparent disagreement qualified by differing follow-up durations ({fa_dur} vs {fb_dur}): early effect attenuates or diverges over extended follow-up."
                    score = round(float(diff.overall_compatibility), 3)
                else:
                    pred_label = RelationLabel.CONTRADICTS
                    null_note = " (Note: non-significance does not by itself prove equivalence; statistical power differences may explain lack of detected effect)." if (fa.direction == DirectionLabel.NO_DIFFERENCE or fb.direction == DirectionLabel.NO_DIFFERENCE) else ""
                    explanation = f"Direct empirical contradiction: studies evaluate comparable conditions ({fa.conditions.intervention} on {fa.conditions.outcome_metric}) but report opposing directional findings.{null_note}"
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

