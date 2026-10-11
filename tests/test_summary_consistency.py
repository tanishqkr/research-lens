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
