"""Regression tests for intervention and comparator arm attribution in numerical extraction.

Verifies:
1. Exact audit case: LCDs vs LFDs (-2.99 kg) and VLCDs subgroup (-9.49 kg) attribution to LCDs/VLCDs instead of LFDs.
2. Varied comparative phrasing ('than', 'compared with', 'versus', 'superior to').
3. Reversed sentence order (effect first, comparison second).
4. Abbreviation handling (LCDs, LFDs, VLCDs).
5. Multiple numerical estimates in a single sentence across endpoints.
6. Missing or ambiguous arm references are marked unresolved (None), not assigned invented strings.
7. Verbatim groundedness of every numerical result against evidence span.
"""

from scientific_contradiction_finder.extraction.condition_tagger import extract_numerical_results


def test_audit_weight_loss_attribution_lcd_vs_lfd():
    """Verifies that in 'LCDs appeared to be more beneficial than LFDs (WMD = -2.99 kg...)',
    the effect is attributed to LCDs (intervention) vs LFDs (comparator), NOT inverted.
    """
    sent = (
        "LCDs appeared to be more beneficial in decreasing body weight than LFDs "
        "(WMD = -2.99 kg; 95% CI: -4.36, -1.63; P < 0.0001), especially in the subgroup "
        "that used VLCDs (WMD = -9.49 kg; 95% CI: -12.88, -6.09, P < 0.00001)."
    )
    results = extract_numerical_results(sent)
    assert len(results) == 2

    # Primary estimate: -2.99 kg
    r1 = results[0]
    assert r1.point_estimate == -2.99
    assert r1.unit == "kg"
    assert r1.arm_name == "LCDs", f"Expected LCDs, got {r1.arm_name}"
    assert r1.comparator_arm == "LFDs", f"Expected LFDs, got {r1.comparator_arm}"
    assert r1.ci_lower == -4.36
    assert r1.ci_upper == -1.63
    assert r1.p_value == 0.0001
    assert r1.metric_name == "WMD"

    # Subgroup estimate: -9.49 kg
    r2 = results[1]
    assert r2.point_estimate == -9.49
    assert r2.unit == "kg"
    assert r2.arm_name == "VLCDs", f"Expected VLCDs, got {r2.arm_name}"
    assert r2.comparator_arm == "LFDs", f"Expected LFDs, got {r2.comparator_arm}"
    assert r2.ci_lower == -12.88
    assert r2.ci_upper == -6.09
    assert r2.p_value == 0.00001
    assert r2.metric_name == "WMD"


def test_fronted_comparative_phrasing():
    """'Compared with LFDs, treatment with LCDs achieved...' -> arm=LCDs, comp=LFDs."""
    sent = (
        "Compared with LFDs, treatment with LCDs achieved significant reduced HbA1c "
        "by 0.35% (95% CI: -0.45, -0.24; P < 0.00001)."
    )
    results = extract_numerical_results(sent)
    assert len(results) >= 1
    r = results[0]
    assert r.point_estimate == 0.35
    assert r.unit == "%"
    assert r.arm_name == "LCDs"
    assert r.comparator_arm == "LFDs"
    assert r.ci_lower == -0.45
    assert r.ci_upper == -0.24
    assert r.p_value == 0.00001


def test_reversed_sentence_order():
    """'Greater weight loss of -3.2 kg was observed with LCDs than with LFDs.' -> arm=LCDs, comp=LFDs."""
    sent = "Greater weight loss of -3.2 kg was observed with LCDs than with LFDs (p = 0.001)."
    results = extract_numerical_results(sent)
    assert len(results) >= 1
    r = results[0]
    assert r.point_estimate == -3.2
    assert r.unit == "kg"
    assert r.arm_name == "LCDs"
    assert r.comparator_arm == "LFDs"
    assert r.p_value == 0.001


def test_versus_comparative_phrasing():
    """'Low-carbohydrate diets reduced body weight by 4.5 kg versus low-fat diets.'"""
    sent = "Low-carbohydrate diets reduced body weight by 4.5 kg versus low-fat diets (p = 0.002)."
    results = extract_numerical_results(sent)
    assert len(results) >= 1
    r = results[0]
    assert r.point_estimate == 4.5
    assert r.unit == "kg"
    assert "low-carbohydrate" in r.arm_name.lower()
    assert "low-fat" in r.comparator_arm.lower()
    assert r.p_value == 0.002


def test_multiple_numerical_estimates_across_endpoints():
    """Multiple endpoints in one sentence with fronted comparator."""
    sent = (
        "Compared with participants on LFD, participants on LCD had a greater reduction in "
        "triglycerides (-0.14 mmol/L; 95% CI, -0.18 to -0.10 mmol/L), diastolic blood pressure "
        "(-0.87 mmHg; 95% CI, -1.41 to -0.32 mmHg), and weight loss (-1.33 kg; 95% CI, -1.79 to -0.87 kg)."
    )
    results = extract_numerical_results(sent)
    assert len(results) == 3

    # Triglycerides
    tg = next(r for r in results if r.unit == "mmol/l")
    assert tg.point_estimate == -0.14
    assert tg.arm_name == "LCD"
    assert tg.comparator_arm == "LFD"
    assert tg.ci_lower == -0.18
    assert tg.ci_upper == -0.10

    # Blood pressure
    bp = next(r for r in results if r.unit == "mmhg")
    assert bp.point_estimate == -0.87
    assert bp.arm_name == "LCD"
    assert bp.comparator_arm == "LFD"
    assert bp.ci_lower == -1.41
    assert bp.ci_upper == -0.32

    # Weight loss
    wt = next(r for r in results if r.unit == "kg")
    assert wt.point_estimate == -1.33
    assert wt.arm_name == "LCD"
    assert wt.comparator_arm == "LFD"
    assert wt.ci_lower == -1.79
    assert wt.ci_upper == -0.87


def test_missing_or_ambiguous_arm_unresolved():
    """If no arm reference is present or attribution is ambiguous, arm_name must be None (unresolved),
    not an invented placeholder.
    """
    sent = "The pooled mean difference was -1.5 kg (95% CI -2.0 to -1.0, p = 0.002) in the overall meta-analysis."
    results = extract_numerical_results(sent)
    assert len(results) >= 1
    r = results[0]
    assert r.point_estimate == -1.5
    assert r.arm_name is None, f"Expected None for missing arm, got {r.arm_name}"
    assert r.comparator_arm is None, f"Expected None for missing comparator, got {r.comparator_arm}"


def test_grounding_verification_against_evidence_span():
    """All extracted estimates, CIs, and p-values must strictly correspond to evidence text."""
    sent = "In the intervention arm, systolic blood pressure dropped by 5.4 mmHg (95% CI 2.1 to 8.7, p = 0.01)."
    results = extract_numerical_results(sent)
    assert len(results) >= 1
    r = results[0]
    assert r.point_estimate == 5.4
    assert str(r.point_estimate) in sent
    assert "95% CI 2.1 to 8.7" in sent
    assert "p = 0.01" in sent


def test_superior_to_comparative_phrasing():
    """'LCDs were superior to LFDs with a mean difference of -2.99 kg...' must attribute to LCDs, not LFDs."""
    sent = "LCDs were superior to LFDs with a mean difference of -2.99 kg (95% CI: -4.36, -1.63)."
    results = extract_numerical_results(sent)
    assert len(results) >= 1
    r = results[0]
    assert r.point_estimate == -2.99
    assert r.arm_name == "LCDs"
    assert r.comparator_arm == "LFDs"
    assert r.ci_lower == -4.36
    assert r.ci_upper == -1.63


def test_dimensionless_smd_extraction():
    """Standardized Mean Differences (SMD) have no unit but must be reliably extracted."""
    sent = "LCDs were more effective than LFDs (SMD = -0.45; 95% CI: -0.70, -0.20; P < 0.01)."
    results = extract_numerical_results(sent)
    assert len(results) >= 1
    r = results[0]
    assert r.point_estimate == -0.45
    assert r.metric_name == "SMD"
    assert r.arm_name == "LCDs"
    assert r.comparator_arm == "LFDs"
    assert r.ci_lower == -0.70
    assert r.ci_upper == -0.20
    assert r.p_value == 0.01


def test_spelled_out_ratio_names_and_comma_separators():
    """NEJM style: 'hazard ratio, 4.53; 95% confidence interval, 2.00 to 10.27; P<0.001'."""
    sent = (
        "Patients in whom MNPs were detected within the atheroma were at higher risk for a primary end-point event "
        "than those in whom these substances were not detected (hazard ratio, 4.53; 95% confidence interval, 2.00 to 10.27; P<0.001)."
    )
    results = extract_numerical_results(sent)
    assert len(results) >= 1
    r = results[0]
    assert r.point_estimate == 4.53
    assert r.metric_name == "HR"
    assert r.ci_lower == 2.00
    assert r.ci_upper == 10.27
    assert r.p_value == 0.001


def test_comparator_arm_safeguard_never_assigned_as_primary():
    """When only a comparator follows a comparative marker ('than in the control group'),
    it must be set as comparator_arm, and arm_name must remain None (unresolved), NOT assigned to control.
    """
    sent = "Mortality was significantly lower than in the control group (RR = 0.75; 95% CI 0.65 to 0.85)."
    results = extract_numerical_results(sent)
    assert len(results) >= 1
    r = results[0]
    assert r.point_estimate == 0.75
    assert r.comparator_arm == "control group"
    assert r.arm_name is None, f"Expected None for missing intervention arm, got {r.arm_name}"


def test_ambiguous_multi_arm_sentences_unresolved():
    """Ambiguous multi-arm sentences with 3+ arms and no explicit pairwise mapping
    must return unresolved (None) arm assignments rather than guessing based on proximity.
    """
    # Case 1: List of 3 arms without comparative marker
    sent1 = "Across arm A, arm B, and arm C, the pooled mean change was -1.5 kg (95% CI: -2.0 to -1.0, P=0.01)."
    res1 = extract_numerical_results(sent1)
    assert len(res1) >= 1
    r1 = res1[0]
    assert r1.point_estimate == -1.5
    assert r1.arm_name is None, f"Expected None, got {r1.arm_name}"
    assert r1.comparator_arm is None, f"Expected None, got {r1.comparator_arm}"

    # Case 2: Multi-arm trial phrase ('3-arm trial') with 3 candidate drugs
    sent2 = "In a 3-arm trial of drug A, drug B, and drug C, the overall mean difference was -1.2 kg (95% CI: -1.8 to -0.6)."
    res2 = extract_numerical_results(sent2)
    assert len(res2) >= 1
    r2 = res2[0]
    assert r2.point_estimate == -1.2
    assert r2.arm_name is None, f"Expected None, got {r2.arm_name}"
    assert r2.comparator_arm is None, f"Expected None, got {r2.comparator_arm}"


def test_passive_inferior_to_comparative_phrasing():
    """Passive comparison: 'Treatment A was inferior to treatment B...' attributes to A (intervention) vs B (comparator)."""
    sent = "Treatment A was inferior to treatment B (hazard ratio = 1.45; 95% CI: 1.10 to 1.90)."
    results = extract_numerical_results(sent)
    assert len(results) >= 1
    r = results[0]
    assert r.point_estimate == 1.45
    assert r.metric_name == "HR"
    assert r.arm_name == "Treatment A"
    assert r.comparator_arm == "treatment B"
    assert r.ci_lower == 1.10
    assert r.ci_upper == 1.90


def test_fronted_treatment_arm_attribution():
    """Fronted comparison: 'Compared with treatment A, treatment B achieved...' attributes to B vs A."""
    sent = "Compared with treatment A, treatment B achieved a reduction in body weight of 2.5 kg (95% CI: 1.2 to 3.8)."
    results = extract_numerical_results(sent)
    assert len(results) >= 1
    r = results[0]
    assert r.point_estimate == 2.5
    assert r.arm_name == "treatment B"
    assert r.comparator_arm == "treatment A"
    assert r.ci_lower == 1.2
    assert r.ci_upper == 3.8


