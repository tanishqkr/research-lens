"""Regression tests for Milestone 2: Scientific Reasoning and Evidence Integrity."""


from scientific_contradiction_finder.extraction.condition_tagger import (
    extract_numerical_results,
    infer_study_type,
)
from scientific_contradiction_finder.reasoning.classifier import (
    ConditionAwareRelationClassifier,
)
from scientific_contradiction_finder.schemas import (
    ConditionSlots,
    DirectionLabel,
    EvidenceSpan,
    Finding,
    FindingPair,
    RelationLabel,
    StudyDesignType,
)


def test_1_multi_arm_slot_attribution():
    """1. Multi-arm trial slot attribution: each estimate attributed to correct arm, comparator, endpoint, horizon, uncertainty."""
    sent = (
        "At 6 months, patients randomized to low-carbohydrate diets lost 5.8 kg (p = 0.002) "
        "compared with 1.9 kg in the low-fat diet group (n = 60 per arm)."
    )
    results = extract_numerical_results(sent)
    assert len(results) >= 2

    # Check low-carb arm
    low_carb_res = next(
        (r for r in results if r.arm_name and "low-carbohydrate" in r.arm_name.lower() and r.unit == "kg"),
        None,
    )
    assert low_carb_res is not None
    assert low_carb_res.point_estimate == 5.8
    assert low_carb_res.unit == "kg"
    assert low_carb_res.comparator_arm is not None
    assert "low-fat" in low_carb_res.comparator_arm.lower()
    assert low_carb_res.follow_up_horizon == "6 months"
    assert low_carb_res.p_value == 0.002
    assert "p = 0.002" in (low_carb_res.statistical_uncertainty or "")

    # Check low-fat arm
    low_fat_res = next(
        (r for r in results if r.arm_name and "low-fat" in r.arm_name.lower() and r.unit == "kg"),
        None,
    )
    assert low_fat_res is not None
    assert low_fat_res.point_estimate == 1.9
    assert low_fat_res.unit == "kg"
    assert low_fat_res.comparator_arm is not None
    assert "low-carbohydrate" in low_fat_res.comparator_arm.lower()
    assert low_fat_res.follow_up_horizon == "6 months"
    assert low_fat_res.sample_size == 60


def test_2_odds_ratio_with_ci_and_arm_attribution():
    """2. Odds ratio extraction with CI, arm, comparator, and statistical uncertainty."""
    sent = "Compared with standard care (OR 1.00), arm A achieved OR 1.45 (95% CI 1.12 to 1.88, p = 0.002) at 12 months."
    results = extract_numerical_results(sent)
    or_res = next((r for r in results if r.metric_name == "OR" and r.point_estimate == 1.45), None)
    assert or_res is not None
    assert or_res.ci_lower == 1.12
    assert or_res.ci_upper == 1.88
    assert or_res.p_value == 0.002
    assert "95% CI 1.12 to 1.88" in (or_res.statistical_uncertainty or "")
    assert or_res.follow_up_horizon == "12 months"


def test_3_robust_study_design_classification():
    """3. Study design classification: INTERVENTIONAL_TRIAL vs OBSERVATIONAL_STUDY vs SYSTEMATIC_REVIEW vs PREDICTION_MODEL vs UNKNOWN."""
    # Systematic review
    t_rev, _ = infer_study_type("A systematic review and meta-analysis of randomized trials evaluated dietary patterns.")
    assert t_rev == StudyDesignType.SYSTEMATIC_REVIEW_META_ANALYSIS

    # Interventional trial
    t_rct, _ = infer_study_type("In this randomized, double-blind, parallel-group trial of 240 patients...")
    assert t_rct == StudyDesignType.INTERVENTIONAL_TRIAL

    # Observational study
    t_obs, _ = infer_study_type("A prospective cohort study followed 15,000 adults over a 10-year period.")
    assert t_obs == StudyDesignType.OBSERVATIONAL_STUDY

    # Prediction model
    t_pred, _ = infer_study_type("We developed a machine learning prediction model using XGBoost to predict sepsis onset (AUROC 0.88).")
    assert t_pred == StudyDesignType.PREDICTION_MODEL

    # Ambiguous / unstated -> UNKNOWN
    t_unk, _ = infer_study_type("We evaluated the chemical properties and binding affinity of compound X.")
    assert t_unk == StudyDesignType.UNKNOWN


def test_4_prediction_performance_vs_clinical_outcome_incomparability():
    """4. Predictive performance metrics (AUROC) vs clinical incidence endpoints are marked INCOMPARABLE_CONDITIONS."""
    classifier = ConditionAwareRelationClassifier()

    fa_model = Finding(
        finding_id="f_pred",
        paper_id="p1",
        paper_title="Sepsis Prediction ML",
        evidence=EvidenceSpan(text="XGBoost model achieved AUROC 0.89 for predicting pediatric sepsis onset."),
        finding_claim="XGBoost predicts pediatric sepsis onset with high discrimination.",
        direction=DirectionLabel.POSITIVE,
        conditions=ConditionSlots(
            intervention="XGBoost Model",
            outcome_metric="AUROC Discrimination",
            study_type=StudyDesignType.PREDICTION_MODEL,
        ),
    )

    fb_trial = Finding(
        finding_id="f_trial",
        paper_id="p2",
        paper_title="Antibiotic Therapy for Sepsis",
        evidence=EvidenceSpan(text="Early antibiotic administration reduced sepsis mortality in children (RR 0.72)."),
        finding_claim="Early antibiotics reduce pediatric sepsis mortality.",
        direction=DirectionLabel.NEGATIVE,
        conditions=ConditionSlots(
            intervention="Early Antibiotics",
            outcome_metric="Sepsis Mortality",
            study_type=StudyDesignType.INTERVENTIONAL_TRIAL,
        ),
    )

    pair = FindingPair(pair_id="pair_pred_trial", finding_a=fa_model, finding_b=fb_trial)
    pred = classifier.classify_pair(pair)

    assert pred.predicted_relation == RelationLabel.INCOMPARABLE_CONDITIONS
    assert pred.heuristic_score <= 0.50
    assert "incomparable targets" in pred.explanation.lower() or "disparate" in pred.explanation.lower()


def test_5_observational_vs_interventional_trial_divergence():
    """5. Observational association vs randomized trial discrepancy is classified as QUALIFIES_OR_PARTIAL_DISAGREEMENT, not CONTRADICTS."""
    classifier = ConditionAwareRelationClassifier()

    fa_obs = Finding(
        finding_id="f_obs",
        paper_id="p1",
        paper_title="Alcohol Cohort Study",
        evidence=EvidenceSpan(text="Moderate alcohol was associated with 18% lower all-cause mortality (HR 0.82)."),
        finding_claim="Moderate alcohol is associated with lower mortality.",
        direction=DirectionLabel.NEGATIVE,
        conditions=ConditionSlots(
            intervention="Moderate Alcohol Consumption",
            comparator="Non-Drinkers",
            outcome_metric="All-Cause Mortality",
            dataset_setting="General Adult Population",
            study_type=StudyDesignType.OBSERVATIONAL_STUDY,
        ),
    )

    fb_trial = Finding(
        finding_id="f_trial",
        paper_id="p2",
        paper_title="Alcohol Cessation RCT",
        evidence=EvidenceSpan(text="Randomized trial showed no reduction in all-cause mortality with moderate alcohol."),
        finding_claim="Moderate alcohol showed no effect on mortality in randomized trial.",
        direction=DirectionLabel.NO_DIFFERENCE,
        conditions=ConditionSlots(
            intervention="Moderate Alcohol Consumption",
            comparator="Non-Drinkers",
            outcome_metric="All-Cause Mortality",
            dataset_setting="General Adult Population",
            study_type=StudyDesignType.INTERVENTIONAL_TRIAL,
        ),
    )

    pair = FindingPair(pair_id="pair_obs_rct", finding_a=fa_obs, finding_b=fb_trial)
    pred = classifier.classify_pair(pair)

    # Must be qualified by study design divergence, NOT direct contradiction
    assert pred.predicted_relation == RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT
    assert "study design divergence" in pred.explanation.lower()
    assert "observational" in pred.explanation.lower()
    assert "confounding" in pred.explanation.lower()


def test_6_nonsignificance_note_in_direct_contradiction():
    """6. When two identical RCTs report opposing significance, the explanation explicitly notes that non-significance does not prove equivalence."""
    classifier = ConditionAwareRelationClassifier()

    fa_rct1 = Finding(
        finding_id="f1",
        paper_id="p1",
        paper_title="Trial 1",
        evidence=EvidenceSpan(text="Intervention significantly reduced mortality (p = 0.01)."),
        finding_claim="Intervention reduces mortality.",
        direction=DirectionLabel.NEGATIVE,
        conditions=ConditionSlots(
            intervention="Drug X",
            comparator="Placebo",
            outcome_metric="All-Cause Mortality",
            dataset_setting="ICU Patients",
            study_type=StudyDesignType.INTERVENTIONAL_TRIAL,
        ),
    )

    fb_rct2 = Finding(
        finding_id="f2",
        paper_id="p2",
        paper_title="Trial 2",
        evidence=EvidenceSpan(text="Intervention showed no significant difference in mortality (p = 0.45)."),
        finding_claim="Intervention showed no difference in mortality.",
        direction=DirectionLabel.NO_DIFFERENCE,
        conditions=ConditionSlots(
            intervention="Drug X",
            comparator="Placebo",
            outcome_metric="All-Cause Mortality",
            dataset_setting="ICU Patients",
            study_type=StudyDesignType.INTERVENTIONAL_TRIAL,
        ),
    )

    pair = FindingPair(pair_id="pair_rct_rct", finding_a=fa_rct1, finding_b=fb_rct2)
    pred = classifier.classify_pair(pair)

    assert pred.predicted_relation == RelationLabel.CONTRADICTS
    assert "non-significance does not by itself prove equivalence" in pred.explanation.lower()


def test_7_negative_point_estimates_and_negative_ci_bounds():
    """7. Extraction handles negative estimates, negative CI bounds, unicode minus, and avoids CI bound re-extraction."""
    sent = (
        "In the intervention arm, weight changed by -1.33 kg (95% CI -1.79 to -0.87 kg, p < 0.001) "
        "compared with −0.21 kg in controls."
    )
    results = extract_numerical_results(sent)
    assert len(results) == 2

    # Intervention arm
    int_res = results[0]
    assert int_res.point_estimate == -1.33
    assert int_res.ci_lower == -1.79
    assert int_res.ci_upper == -0.87
    assert int_res.p_value == 0.001
    assert int_res.unit == "kg"

    # Control arm
    ctl_res = results[1]
    assert ctl_res.point_estimate == -0.21
    assert ctl_res.unit == "kg"
    # Ensure no bleeding of CI from arm 1 to control arm
    assert ctl_res.ci_lower is None
    assert ctl_res.ci_upper is None
    assert ctl_res.p_value is None


def test_8_study_type_background_filtering():
    """8. infer_study_type prioritizes METHODS/DESIGN sections and ignores prior trial mentions in BACKGROUND."""
    text = (
        "BACKGROUND: Prior randomized controlled trials demonstrated acute efficacy of the drug. "
        "METHODS: In this prospective cohort study of 4,200 women followed over 8 years, "
        "we assessed long-term cardiovascular incidence."
    )
    stype, details = infer_study_type(text)
    assert stype == StudyDesignType.OBSERVATIONAL_STUDY
    assert "observational" in details.lower() or "cohort" in details.lower()


def test_9_canonical_intervention_acronym_matching():
    """9. Condition similarity recognizes canonical acronyms (e.g. LCD vs Low-Carbohydrate Diet)."""
    classifier = ConditionAwareRelationClassifier()

    fa = Finding(
        finding_id="f_lcd",
        paper_id="p1",
        paper_title="Trial LCD",
        evidence=EvidenceSpan(text="LCD significantly reduced HbA1c levels by 0.6% compared to baseline."),
        finding_claim="LCD reduces HbA1c.",
        direction=DirectionLabel.NEGATIVE,
        conditions=ConditionSlots(
            intervention="LCD",
            comparator="Standard Care",
            outcome_metric="HbA1c",
            dataset_setting="Type 2 Diabetes",
            study_type=StudyDesignType.INTERVENTIONAL_TRIAL,
        ),
    )

    fb = Finding(
        finding_id="f_lowcarb",
        paper_id="p2",
        paper_title="Trial Low-Carb",
        evidence=EvidenceSpan(text="Low-carbohydrate diet reduced HbA1c by 0.5% compared to controls."),
        finding_claim="Low-carbohydrate diet reduces HbA1c.",
        direction=DirectionLabel.NEGATIVE,
        conditions=ConditionSlots(
            intervention="Low-Carbohydrate Diet",
            comparator="Control",
            outcome_metric="HbA1c",
            dataset_setting="Type 2 Diabetes",
            study_type=StudyDesignType.INTERVENTIONAL_TRIAL,
        ),
    )

    pair = FindingPair(pair_id="pair_lcd_lowcarb", finding_a=fa, finding_b=fb)
    pred = classifier.classify_pair(pair)

    # Must recognize intervention similarity and classify as SUPPORTS_OR_CONSISTENT, not INCOMPARABLE_CONDITIONS
    assert pred.predicted_relation == RelationLabel.SUPPORTS_OR_CONSISTENT
    assert pred.heuristic_score > 0.50


def test_10_weight_reduction_directional_concordance():
    """10. Opposite syntactical directions on weight reduction (lost weight vs reduced body weight) do not trigger false contradiction."""
    classifier = ConditionAwareRelationClassifier()

    fa_loss = Finding(
        finding_id="f_loss",
        paper_id="p1",
        paper_title="Diet Weight Loss Trial",
        evidence=EvidenceSpan(text="Ketogenic diet produced greater weight loss (lost 6.1 kg) compared with low-fat diet."),
        finding_claim="Ketogenic diet increases weight loss.",
        direction=DirectionLabel.POSITIVE,  # "loss" framed as positive outcome
        conditions=ConditionSlots(
            intervention="Ketogenic Diet",
            comparator="Low-Fat Diet",
            outcome_metric="Weight Loss",
            study_type=StudyDesignType.INTERVENTIONAL_TRIAL,
        ),
    )

    fb_reduction = Finding(
        finding_id="f_reduc",
        paper_id="p2",
        paper_title="Diet Body weight Reduction Trial",
        evidence=EvidenceSpan(text="Ketogenic diet led to marked reduction in body weight (-5.9 kg) relative to low-fat diet."),
        finding_claim="Ketogenic diet reduces body weight.",
        direction=DirectionLabel.NEGATIVE,  # "reduction" framed as negative metric change
        conditions=ConditionSlots(
            intervention="Ketogenic Diet",
            comparator="Low-Fat Diet",
            outcome_metric="Body Weight",
            study_type=StudyDesignType.INTERVENTIONAL_TRIAL,
        ),
    )

    pair = FindingPair(pair_id="pair_loss_reduction", finding_a=fa_loss, finding_b=fb_reduction)
    pred = classifier.classify_pair(pair)

    # Both findings agree that ketogenic diet leads to greater weight loss/reduction.
    # Should NOT be classified as CONTRADICTS.
    assert pred.predicted_relation in (RelationLabel.SUPPORTS_OR_CONSISTENT, RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT)
    assert pred.predicted_relation != RelationLabel.CONTRADICTS

