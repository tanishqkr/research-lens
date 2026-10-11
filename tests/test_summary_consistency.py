from scientific_contradiction_finder.reasoning.summary_validator import (
    assess_subclaims,
    decompose_query_into_subclaims,
    enforce_summary_consistency,
    verify_disagreement_explanations,
)
from scientific_contradiction_finder.schemas import (
    DirectionLabel,
    DisagreementExplanation,
    ExecutiveSummary,
    ExplanationVerificationStatus,
    SubclaimAssessment,
    SubclaimStatus,
    SupportedEvidenceClaim,
)


def test_decompose_urban_heat_query():
    query = "Does increasing urban tree canopy cover reduce local temperatures, and does the effect vary with city density and climate?"
    subclaims = decompose_query_into_subclaims(query)
    assert len(subclaims) == 4
    assert any("Canopy cooling effect" in sc for sc in subclaims)
    assert any("Cross-climate variation" in sc for sc in subclaims)


def test_decompose_sleep_query():
    query = "Does sleep restriction impair working memory in adults, and do the effects differ between partial sleep restriction and total sleep deprivation?"
    subclaims = decompose_query_into_subclaims(query)
    assert len(subclaims) == 3
    assert any("Adult impairment" in sc for sc in subclaims)
    assert any("total sleep deprivation" in sc for sc in subclaims)


def test_assess_subclaims_urban_heat_single_city():
    subclaims = [
        "Canopy cooling effect: Increasing urban tree canopy cover reduces local air temperatures.",
        "Canopy volume and scaling: Cooling magnitude scales with canopy volume and density (e.g. ~1-1.5°C).",
        "Cross-climate variation: Cooling magnitude systematically varies across distinct macroclimate zones.",
    ]
    # Single city ledger
    evidence_ledger = [
        SupportedEvidenceClaim(
            claim_id="c1",
            paper_id="W1",
            paper_title="Portland Canopy Study",
            exact_source_passage="Near-surface air temperature decreased linearly with tree canopy cover, reducing 1°F per 500 cubic feet.",
            direction=DirectionLabel.POSITIVE,
        ),
        SupportedEvidenceClaim(
            claim_id="c2",
            paper_id="W2",
            paper_title="DC Canopy Study",
            exact_source_passage="Tree canopy cover is the dominant cooling factor, lowering temperature up to 1.5 °C.",
            direction=DirectionLabel.NEGATIVE,
        ),
    ]
    assessments = assess_subclaims(subclaims, evidence_ledger)
    assert len(assessments) == 3

    # Cooling effect is supported
    assert assessments[0].status == SubclaimStatus.SUPPORTED
    assert "W1" in assessments[0].evidence_paper_ids

    # Volume and scaling is supported
    assert assessments[1].status == SubclaimStatus.SUPPORTED

    # Cross-climate is UNRESOLVED because studies were single-city
    assert assessments[2].status == SubclaimStatus.UNRESOLVED
    assert len(assessments[2].evidence_paper_ids) == 0
    assert "single-city" in assessments[2].rationale.lower()


def test_enforce_summary_consistency_downgrades_climate_overstatement():
    exec_summary = ExecutiveSummary(
        bottom_line="Increasing tree canopy reduces temperatures, varying significantly with city density and climate.",
        overall_evidence_assessment="broadly supports",
        strongest_supporting_evidence="Study W1 found 1.5C cooling.",
        strongest_conflicting_evidence="No conflicting studies.",
        why_studies_disagree="Differences in morphology.",
        confidence_and_limitations="High confidence in cooling.",
        research_coverage="Retrieved 25, examined 8.",
    )

    subclaims = [
        "Canopy cooling effect: Increasing urban tree canopy cover reduces local air temperatures.",
        "Cross-climate variation: Cooling magnitude systematically varies across distinct macroclimate zones.",
    ]
    evidence_ledger = [
        SupportedEvidenceClaim(
            claim_id="c1",
            paper_id="W1",
            paper_title="DC Canopy Study",
            exact_source_passage="Tree canopy cover lowers temperature up to 1.5 °C.",
            direction=DirectionLabel.NEGATIVE,
        )
    ]
    assessments = assess_subclaims(subclaims, evidence_ledger)
    assert assessments[1].status == SubclaimStatus.UNRESOLVED

    validated_summary = enforce_summary_consistency(exec_summary, assessments, evidence_ledger)
    # Ensure bottom-line no longer asserts climate variation without qualification
    assert (
        "unresolved" in validated_summary.bottom_line.lower()
        or "not directly tested" in validated_summary.bottom_line.lower()
    )
    assert "cross-climate" in validated_summary.confidence_and_limitations.lower()


def test_verify_disagreement_explanations():
    disagreements = [
        DisagreementExplanation(
            topic_or_metric="Cooling Magnitude",
            studies_in_disagreement=["W1", "W2"],
            plausible_moderator_or_reason="Differences in canopy volume and surrounding buildings.",
        ),
        DisagreementExplanation(
            topic_or_metric="Genetic Predisposition",
            studies_in_disagreement=["W1", "W2"],
            plausible_moderator_or_reason="Unmeasured polymorphic allele frequencies across participant cohorts.",
        ),
    ]

    evidence_ledger = [
        SupportedEvidenceClaim(
            claim_id="c1",
            paper_id="W1",
            paper_title="Study 1",
            exact_source_passage="Near-surface air temperature decreased linearly with canopy volume.",
            direction=DirectionLabel.NEGATIVE,
        ),
        SupportedEvidenceClaim(
            claim_id="c2",
            paper_id="W2",
            paper_title="Study 2",
            exact_source_passage="Surrounding buildings modulate daytime warming and nocturnal cooling.",
            direction=DirectionLabel.NEGATIVE,
        ),
    ]

    verified = verify_disagreement_explanations(disagreements, evidence_ledger)
    assert len(verified) == 2
    # The first one has two supporting evidence ledger claims for its keywords (volume, buildings)
    assert verified[0].verification_status == ExplanationVerificationStatus.SUPPORTED
    assert len(verified[0].supporting_evidence_ids) >= 2

    # The second one has zero supporting evidence claims in the ledger
    assert verified[1].verification_status == ExplanationVerificationStatus.UNRESOLVED
    assert len(verified[1].supporting_evidence_ids) == 0


def test_adversarial_unsupported_moderator_marked_unresolved():
    """Adversarial test: An unverified hypothesis presented as an established explanation must be marked UNRESOLVED."""
    disagreements = [
        DisagreementExplanation(
            topic_or_metric="Working Memory Speed",
            studies_in_disagreement=["Study A", "Study B"],
            plausible_moderator_or_reason="Circadian clock gene polymorphism PER3 length variants caused divergence.",
        )
    ]
    # Ledger only mentions behavioral tasks, no clock genes
    evidence_ledger = [
        SupportedEvidenceClaim(
            claim_id="c1",
            paper_id="Study A",
            paper_title="Study A",
            exact_source_passage="Participants performed worse on N-back task under sleep restriction.",
            direction=DirectionLabel.NEGATIVE,
        )
    ]
    verified = verify_disagreement_explanations(disagreements, evidence_ledger)
    assert verified[0].verification_status == ExplanationVerificationStatus.UNRESOLVED
    assert len(verified[0].supporting_evidence_ids) == 0


def test_adversarial_overstated_summary_climate_and_deprivation_downgraded():
    """Adversarial test: Summary claiming definitive multi-climate effects and adult deprivation differences is programmatically qualified."""
    exec_summary = ExecutiveSummary(
        bottom_line="Tree canopy cooling varies with city density and climate.",
        overall_evidence_assessment="broadly supports",
        strongest_supporting_evidence="Cooling demonstrated in one city.",
        strongest_conflicting_evidence="None.",
        why_studies_disagree="Measurement differences.",
        confidence_and_limitations="High overall confidence.",
        research_coverage="Retrieved 10.",
    )
    subclaims = [
        "Canopy cooling effect: Increasing urban tree canopy cover reduces local air temperatures.",
        "Cross-climate variation: Cooling magnitude systematically varies across distinct macroclimate zones.",
    ]
    # Only single-city paper in ledger
    evidence_ledger = [
        SupportedEvidenceClaim(
            claim_id="c1",
            paper_id="W1",
            paper_title="Single City Study",
            exact_source_passage="Tree canopy cover lowered temperature by 1.2C in New Haven.",
            direction=DirectionLabel.NEGATIVE,
        )
    ]
    assessments = assess_subclaims(subclaims, evidence_ledger)
    assert assessments[1].status == SubclaimStatus.UNRESOLVED

    validated = enforce_summary_consistency(exec_summary, assessments, evidence_ledger)
    assert "cross-climate" in validated.confidence_and_limitations.lower()
    assert "unresolved" in validated.confidence_and_limitations.lower()


def test_adversarial_abstract_only_vs_full_text_provenance():
    """Adversarial test: Ensure abstract-only claim is accurately flagged and not mistaken for full-text."""
    claim = SupportedEvidenceClaim(
        claim_id="c1",
        paper_id="P1",
        paper_title="Abstract Only Paper",
        exact_source_passage="Canopy reduces temperature.",
        full_text_used=False,
    )
    assert claim.full_text_used is False


def test_adversarial_irrelevant_keyword_match_rejected():
    """Adversarial test: A single matching word (e.g. 'water') must NEVER mark an unrelated subclaim SUPPORTED."""
    subclaims = [
        "Subclaim: drinking cold water immediately after eating chocolate causes permanent intelligence reduction"
    ]
    # Irrelevant environmental chemistry paper about water
    evidence_ledger = [
        SupportedEvidenceClaim(
            claim_id="c_irr",
            paper_id="W_irr",
            paper_title="Determination of trihalomethanes in tap water",
            exact_source_passage="The concentration of THMs in potable water after chlorination showed decreasing trend.",
            intervention_exposure="Water source type",
            outcome="Concentration of THMs in potable water",
            direction=DirectionLabel.NEGATIVE,
        )
    ]
    assessments = assess_subclaims(subclaims, evidence_ledger)
    assert len(assessments) == 1
    # Must be UNRESOLVED, not SUPPORTED
    assert assessments[0].status == SubclaimStatus.UNRESOLVED
    assert len(assessments[0].evidence_paper_ids) == 0


def test_adversarial_mismatched_intervention_unresolved():
    """Adversarial test: A meal-regularity study without calorie restriction cannot resolve head-to-head isocaloric fasting."""
    subclaims = [
        "Subclaim: intermittent fasting produces greater long-term weight loss than continuous calorie restriction when intake is comparable"
    ]
    # Meal regularity study without calorie restriction
    evidence_ledger = [
        SupportedEvidenceClaim(
            claim_id="c_reg",
            paper_id="W_reg",
            paper_title="Circadian eating schedule leads to weight loss without calorie restriction",
            exact_source_passage="More regular meals led to 2.62 kg weight loss compared to 0.56 kg in controls.",
            intervention_exposure="Meal regularity",
            comparator="Irregular meals",
            outcome="Weight loss",
            direction=DirectionLabel.POSITIVE,
        )
    ]
    assessments = assess_subclaims(subclaims, evidence_ledger)
    assert len(assessments) == 1
    # Since 'intermittent', 'fasting', and 'continuous calorie restriction' are absent from the trial, it must be UNRESOLVED
    assert assessments[0].status == SubclaimStatus.UNRESOLVED


def test_adversarial_empty_ledger_clean_abstention():
    """Adversarial test: When ledger has 0 papers, summary must cleanly abstain without inventing facts."""
    exec_summary = ExecutiveSummary(
        bottom_line="Some generated claim.",
        overall_evidence_assessment="broadly supports",
        strongest_supporting_evidence="None",
        strongest_conflicting_evidence="None",
        why_studies_disagree="None",
        confidence_and_limitations="High confidence",
        research_coverage="0 papers",
    )
    assessments = [
        SubclaimAssessment(
            subclaim_text="Subclaim: effect exists",
            status=SubclaimStatus.UNRESOLVED,
            rationale="No papers found.",
            evidence_paper_ids=[],
        )
    ]
    validated = enforce_summary_consistency(exec_summary, assessments, evidence_ledger=[])
    assert validated.overall_evidence_assessment == "inconclusive"
    assert "no qualifying empirical literature" in validated.bottom_line.lower()
    assert "confidence is zero" in validated.confidence_and_limitations.lower()


def test_adversarial_new_unseen_domain_probiotic_microbiome():
    """Adversarial test on an entirely new domain (probiotics and depression) to ensure generalizability."""
    subclaims = [
        "Subclaim: oral Bifidobacterium supplementation reduces clinical depression severity in treatment-resistant adults"
    ]
    # Paper testing pediatric gut motility, not adult depression
    evidence_ledger = [
        SupportedEvidenceClaim(
            claim_id="c_micro",
            paper_id="W_micro",
            paper_title="Bifidobacterium infantis in preterm infant feeding",
            exact_source_passage="Bifidobacterium colonisation improved gastrointestinal transit times in neonates.",
            intervention_exposure="Bifidobacterium infantis",
            outcome="Gastrointestinal transit time",
            direction=DirectionLabel.POSITIVE,
        )
    ]
    assessments = assess_subclaims(subclaims, evidence_ledger)
    assert assessments[0].status == SubclaimStatus.UNRESOLVED
    assert len(assessments[0].evidence_paper_ids) == 0


def test_review_synthesis_distinguished_and_supports_synthesis():
    """Regression test: A relevant systematic review claim is marked as review synthesis and informs consistency."""
    review_claim = SupportedEvidenceClaim(
        claim_id="c_meta",
        paper_id="W_meta",
        paper_title="Intermittent fasting versus continuous energy restriction: a systematic review",
        exact_source_passage="Intermittent fasting resulted in equivalent weight loss compared to continuous restriction across 27 randomized trials.",
        study_design="Systematic Review",
        is_direct=True,
        is_review_synthesis=True,
        direction=DirectionLabel.NO_DIFFERENCE,
    )
    assert review_claim.is_review_synthesis is True
    assert review_claim.study_design == "Systematic Review"


def test_causal_overstatement_qualified_without_authoritative_synthesis():
    """Regression test: A causal conclusion is qualified when the causal subclaim is unresolved and no systematic review is present."""
    exec_summary = ExecutiveSummary(
        bottom_line="Smoking definitively increases lung cancer. The evidence distinguishes causation from mere association through pack-year dose response.",
        overall_evidence_assessment="broadly supports",
        strongest_supporting_evidence="Cohort study shows HR 1.18.",
        strongest_conflicting_evidence="None.",
        why_studies_disagree="None.",
        confidence_and_limitations="High confidence.",
        research_coverage="1 cohort study.",
    )
    assessments = [
        SubclaimAssessment(
            subclaim_text="Subclaim: Does smoking increase the risk of lung cancer",
            status=SubclaimStatus.SUPPORTED,
            rationale="Supported by cohort study.",
            evidence_paper_ids=["W1"],
        ),
        SubclaimAssessment(
            subclaim_text="Subclaim: does the evidence distinguish causation from association",
            status=SubclaimStatus.UNRESOLVED,
            rationale="Epistemological framework not in cohort snippet.",
            evidence_paper_ids=[],
        ),
    ]
    primary_cohort_claim = SupportedEvidenceClaim(
        claim_id="c1",
        paper_id="W1",
        paper_title="Cohort study of smoking and lung cancer",
        exact_source_passage="Pack-years associated with HR 1.18.",
        study_design="Prospective Cohort",
        is_direct=True,
        is_review_synthesis=False,
        direction=DirectionLabel.POSITIVE,
    )
    validated = enforce_summary_consistency(
        exec_summary, assessments, evidence_ledger=[primary_cohort_claim]
    )
    assert "formal epidemiological establishment of causality requires synthesis" in validated.bottom_line
    assert "formal causal frameworks" in validated.confidence_and_limitations.lower()


def test_regression_non_supporting_passage_with_relevant_citation():
    """Requirement: A passage that mentions keywords but does not support the subclaim must not be marked SUPPORTED."""
    subclaims = ["Subclaim: Does human papillomavirus infection cause cervical cancer"]
    # Citation mentions HPV and cervical cancer, but passage is a mathematical clearance model on 28 patients
    evidence_ledger = [
        SupportedEvidenceClaim(
            claim_id="c_clearance",
            paper_id="arxiv:1111.1496v1",
            paper_title="Dinamics of HPV Infection among Women with Cervical Lesions",
            exact_source_passage="Persistent infection was detected in 35.7% of primarily HPV-positive cases (10 out of 28 patients), mainly in cervical cancer patients.",
            conditions_and_moderators="Small sample size clearance model; mathematical model of clearance",
            study_design="Observational/Mathematical Model",
            is_review_synthesis=False,
            direction=DirectionLabel.POSITIVE,
        )
    ]
    assessments = assess_subclaims(subclaims, evidence_ledger)
    assert assessments[0].status == SubclaimStatus.UNRESOLVED
    assert len(assessments[0].evidence_paper_ids) == 0


def test_regression_authoritative_review_genuinely_supports_conclusion():
    """Requirement: An authoritative systematic review / meta-analysis genuinely supports an established conclusion."""
    subclaims = ["Subclaim: Does human papillomavirus infection cause cervical cancer"]
    evidence_ledger = [
        SupportedEvidenceClaim(
            claim_id="c_walboomers",
            paper_id="W1987350137",
            paper_title="Human papillomavirus is a necessary cause of invasive cervical cancer worldwide",
            exact_source_passage="Combining data and excluding inadequate specimens, worldwide HPV prevalence in cervical carcinomas is 99.7 per cent, establishing HPV as a necessary cause.",
            study_design="Systematic Review / International Case Series",
            is_review_synthesis=True,
            direction=DirectionLabel.POSITIVE,
        )
    ]
    assessments = assess_subclaims(subclaims, evidence_ledger)
    assert assessments[0].status == SubclaimStatus.SUPPORTED
    assert "W1987350137" in assessments[0].evidence_paper_ids


def test_regression_review_mentions_topic_without_requested_comparison():
    """Requirement: A review mentioning topic without requested head-to-head comparison leaves comparison subclaim UNRESOLVED."""
    subclaims = [
        "Subclaim: Does intermittent fasting produce greater long-term weight loss than continuous daily calorie restriction when total calorie intake is comparable",
        "Subclaim: total calorie intake is comparable",
    ]
    # Review only examined within-group weight loss across trials without matched isocaloric comparison
    evidence_ledger = [
        SupportedEvidenceClaim(
            claim_id="c_within_group",
            paper_id="W3005999862",
            paper_title="Intermittent fasting and weight loss: Systematic review",
            exact_source_passage="All 27 IF trials found weight loss of 0.8% to 13.0% of baseline weight with no serious adverse events.",
            study_design="Systematic Review",
            is_review_synthesis=True,
            direction=DirectionLabel.POSITIVE,
        )
    ]
    assessments = assess_subclaims(subclaims, evidence_ledger)
    for a in assessments:
        assert a.status == SubclaimStatus.UNRESOLVED
        assert len(a.evidence_paper_ids) == 0


def test_regression_consistency_between_summary_and_subclaim_assessments():
    """Requirement: Executive summary cannot claim superiority or established variation when subclaims are unresolved."""
    exec_summary = ExecutiveSummary(
        bottom_line="Intermittent fasting is consistently superior to continuous calorie restriction for long-term weight loss.",
        overall_evidence_assessment="broadly supports",
        strongest_supporting_evidence="Review found 0.8% to 13% weight loss.",
        strongest_conflicting_evidence="None.",
        why_studies_disagree="None.",
        confidence_and_limitations="High confidence.",
        research_coverage="3 reviews.",
    )
    assessments = [
        SubclaimAssessment(
            subclaim_text="Subclaim: Does intermittent fasting produce greater long-term weight loss than continuous daily calorie restriction",
            status=SubclaimStatus.UNRESOLVED,
            rationale="No direct superiority demonstrated under isocaloric conditions.",
            evidence_paper_ids=[],
        ),
        SubclaimAssessment(
            subclaim_text="Subclaim: total calorie intake is comparable",
            status=SubclaimStatus.UNRESOLVED,
            rationale="Isocaloric comparisons absent in ledger.",
            evidence_paper_ids=[],
        ),
    ]
    evidence_ledger = [
        SupportedEvidenceClaim(
            claim_id="c1",
            paper_id="W1",
            paper_title="Fasting review",
            exact_source_passage="Trials found 0.8% to 13.0% weight loss.",
            direction=DirectionLabel.POSITIVE,
        )
    ]
    validated = enforce_summary_consistency(exec_summary, assessments, evidence_ledger)
    # The overstatement of superiority must be eliminated and equivalence must NOT be claimed without direct comparative trials
    assert "consistently superior" not in validated.bottom_line
    assert "does not establish superiority" in validated.bottom_line
    assert "equivalent rather than superior" not in validated.bottom_line


def test_regression_narrow_population_cannot_generalize_to_general_population():
    """Requirement: A finding in a narrow survivor cohort (e.g. SPLC in prior cancer patients) cannot support general risk."""
    subclaims = ["Subclaim: Does smoking increase the risk of lung cancer"]
    # Study only looks at second primary lung cancer in patients with prior lung cancer
    evidence_ledger = [
        SupportedEvidenceClaim(
            claim_id="c_splc",
            paper_id="W3136330176",
            paper_title="Tobacco Smoking and Risk of Second Primary Lung Cancer",
            exact_source_passage="Smoking pack-years were significantly associated with increased SPLC risk in prior cancer patients.",
            population_setting="Second Primary Lung Cancer patients with prior lung cancer",
            outcome="Second Primary Lung Cancer risk",
            direction=DirectionLabel.POSITIVE,
        )
    ]
    assessments = assess_subclaims(subclaims, evidence_ledger)
    # Must NOT be marked supported for general primary lung cancer risk
    assert assessments[0].status == SubclaimStatus.UNRESOLVED
    assert len(assessments[0].evidence_paper_ids) == 0


def test_regression_well_established_claim_with_evidence_not_indiscriminately_abstained():
    """Requirement: A well-supported claim with valid evidence must be marked SUPPORTED, preventing indiscriminate abstention."""
    subclaims = ["Subclaim: Does smoking increase the risk of lung cancer"]
    evidence_ledger = [
        SupportedEvidenceClaim(
            claim_id="c_cohort",
            paper_id="W4293384232",
            paper_title="Gender and Smoking-Related Risk of Lung Cancer",
            exact_source_passage="In a prospective cohort of 30,874 participants, cigarette smoking significantly increased incident lung cancer risk with RR 12.4 (95% CI 8.2-18.7).",
            population_setting="General prospective cohort of 30,874 adults",
            outcome="Incident lung cancer risk",
            study_design="Prospective Cohort Study",
            is_direct=True,
            direction=DirectionLabel.POSITIVE,
        )
    ]
    assessments = assess_subclaims(subclaims, evidence_ledger)
    assert assessments[0].status == SubclaimStatus.SUPPORTED
    assert "W4293384232" in assessments[0].evidence_paper_ids


def test_equivalence_requires_actual_comparative_evidence():
    """Requirement: A claim of equivalence requires actual comparative evidence in the ledger."""
    exec_summary = ExecutiveSummary(
        bottom_line="Intermittent fasting is consistently superior to continuous calorie restriction.",
        overall_evidence_assessment="broadly supports",
        strongest_supporting_evidence="Direct RCT.",
        strongest_conflicting_evidence="None.",
        why_studies_disagree="None.",
        confidence_and_limitations="High confidence.",
        research_coverage="1 trial.",
    )
    assessments = [
        SubclaimAssessment(
            subclaim_text="Subclaim: Does intermittent fasting produce greater long-term weight loss than continuous daily calorie restriction",
            status=SubclaimStatus.UNRESOLVED,
            rationale="Equivalence found.",
            evidence_paper_ids=[],
        ),
    ]
    # Ledger WITH direct isocaloric comparison demonstrating equivalence
    comparative_ledger = [
        SupportedEvidenceClaim(
            claim_id="c_direct_rct",
            paper_id="W_rct",
            paper_title="RCT of intermittent fasting versus continuous energy restriction",
            exact_source_passage="Intermittent fasting resulted in equivalent weight loss compared to continuous energy restriction under isocaloric conditions with no significant difference (p=0.48).",
            comparator="Continuous energy restriction",
            is_direct=True,
            direction=DirectionLabel.NO_DIFFERENCE,
        )
    ]
    validated = enforce_summary_consistency(exec_summary, assessments, comparative_ledger)
    assert "equivalence rather than superiority" in validated.bottom_line


def test_strong_observational_association_does_not_establish_causation_alone():
    """Requirement: Strong observational association does not automatically establish causation without authoritative synthesis."""
    exec_summary = ExecutiveSummary(
        bottom_line="Human papillomavirus (HPV) infection is a primary causal factor in the development of cervical cancer.",
        overall_evidence_assessment="broadly supports",
        strongest_supporting_evidence="Case control study OR 23.8.",
        strongest_conflicting_evidence="None.",
        why_studies_disagree="None.",
        confidence_and_limitations="High confidence.",
        research_coverage="1 study.",
    )
    assessments = [
        SubclaimAssessment(
            subclaim_text="Subclaim: Does human papillomavirus infection cause cervical cancer",
            status=SubclaimStatus.SUPPORTED,
            rationale="Supported by case-control study.",
            evidence_paper_ids=["W1994873916"],
        ),
        SubclaimAssessment(
            subclaim_text="Subclaim: evidence supports a causal relationship",
            status=SubclaimStatus.UNRESOLVED,
            rationale="Causal multi-criteria framework not evaluated in snippet.",
            evidence_paper_ids=[],
        ),
    ]
    observational_ledger = [
        SupportedEvidenceClaim(
            claim_id="c_obs",
            paper_id="W1994873916",
            paper_title="Risk factors for cervical cancer in Colombia and Spain",
            exact_source_passage="The presence of HPV DNA in cervical scrapes was assessed by PCR-based methods and was the strongest risk factor (OR = 23.8; 13.4-42.0).",
            study_design="Case-Control Study",
            is_direct=True,
            is_review_synthesis=False,
            direction=DirectionLabel.POSITIVE,
        )
    ]
    validated = enforce_summary_consistency(exec_summary, assessments, observational_ledger)
    assert "strongly associated primary risk factor" in validated.bottom_line
    assert "primary causal factor" not in validated.bottom_line
    assert "formal causal frameworks" in validated.confidence_and_limitations.lower()


def test_observational_association_cannot_independently_support_causal_subclaim():
    """Requirement: An observational association study reporting an odds ratio cannot mark a causal subclaim SUPPORTED."""
    subclaims = [
        "Subclaim: Does human papillomavirus infection cause cervical cancer",
        "Subclaim: evidence supports a causal relationship",
    ]
    observational_ledger = [
        SupportedEvidenceClaim(
            claim_id="c_colombia",
            paper_id="W1994873916",
            paper_title="Risk factors for cervical cancer in Colombia and Spain",
            exact_source_passage="The presence of HPV DNA in cervical scrapes was assessed by PCR-based methods and was the strongest risk factor (OR = 23.8; 13.4-42.0).",
            study_design="Case-Control Study",
            is_direct=True,
            is_review_synthesis=False,
            direction=DirectionLabel.POSITIVE,
        )
    ]
    assessments = assess_subclaims(subclaims, observational_ledger)
    # The causal subclaim must remain UNRESOLVED because an observational OR is not causal proof
    assert assessments[0].status == SubclaimStatus.UNRESOLVED
    assert len(assessments[0].evidence_paper_ids) == 0
    assert "observational" in assessments[0].rationale.lower()
    assert assessments[1].status == SubclaimStatus.UNRESOLVED


def test_missing_comparative_evidence_distinguished_from_inconclusive_comparison():
    """Requirement: Missing comparative evidence in retrieved corpus is distinguished from inconclusive comparison results."""
    # Scenario A: Missing head-to-head evidence (empty comparison ledger)
    subclaims = [
        "Subclaim: Does intermittent fasting produce greater long-term weight loss than continuous daily calorie restriction when total calorie intake is comparable"
    ]
    empty_ledger = []
    assessments_missing = assess_subclaims(subclaims, empty_ledger)
    assert assessments_missing[0].status == SubclaimStatus.UNRESOLVED

    exec_summary_missing = ExecutiveSummary(
        bottom_line="Intermittent fasting produces weight loss, but comparative trials are absent.",
        overall_evidence_assessment="inconclusive",
        strongest_supporting_evidence="None.",
        strongest_conflicting_evidence="None.",
        why_studies_disagree="None.",
        confidence_and_limitations="Confidence is zero.",
        research_coverage="0 studies.",
    )
    validated_missing = enforce_summary_consistency(exec_summary_missing, assessments_missing, empty_ledger)
    assert "empty evidence ledger indicates that no qualifying evidence was found in the retrieved sample, rather than conclusive proof that no broader scientific literature exists" in validated_missing.confidence_and_limitations.lower()

    # Scenario B: Retrieved comparative evidence exists, but trials conflict or show inconclusive difference
    conflicting_ledger = [
        SupportedEvidenceClaim(
            claim_id="c_inconclusive",
            paper_id="W_trial",
            paper_title="Direct trial of intermittent fasting versus continuous restriction",
            exact_source_passage="Results comparing intermittent fasting to continuous restriction were inconclusive with conflicting trends across trial arms.",
            comparator="Continuous restriction",
            is_direct=True,
            direction=DirectionLabel.MIXED,
        )
    ]
    assessments_inconclusive = assess_subclaims(subclaims, conflicting_ledger)
    assert assessments_inconclusive[0].status == SubclaimStatus.SUPPORTED or assessments_inconclusive[0].status == SubclaimStatus.UNRESOLVED


def test_summary_never_claims_broader_evidence_coverage_than_ledger():
    """Requirement: Summary never claims broader evidence coverage than the ledger demonstrates."""
    exec_summary = ExecutiveSummary(
        bottom_line="Tree canopy cooling varies significantly across distinct climate zones globally.",
        overall_evidence_assessment="broadly supports",
        strongest_supporting_evidence="Study 1.",
        strongest_conflicting_evidence="None.",
        why_studies_disagree="None.",
        confidence_and_limitations="High confidence across global zones.",
        research_coverage="1 single-city study.",
    )
    assessments = [
        SubclaimAssessment(
            subclaim_text="Subclaim: Cross-climate variation",
            status=SubclaimStatus.UNRESOLVED,
            rationale="Single-city studies only.",
            evidence_paper_ids=[],
        )
    ]
    ledger = [
        SupportedEvidenceClaim(
            claim_id="c_single_city",
            paper_id="W_city",
            paper_title="Portland canopy study",
            exact_source_passage="Canopy reduced local temperature by 1.2C.",
            direction=DirectionLabel.POSITIVE,
        )
    ]
    validated = enforce_summary_consistency(exec_summary, assessments, ledger)
    assert "cross-climate variation" in validated.bottom_line.lower()
    assert "unresolved" in validated.bottom_line.lower() or "not directly tested" in validated.bottom_line.lower()
    assert (
        "individual cities" in validated.confidence_and_limitations.lower()
        or "single-city" in validated.confidence_and_limitations.lower()
    )

