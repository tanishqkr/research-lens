"""Regression tests for extraction defects remediation."""


from scientific_contradiction_finder.extraction.condition_tagger import clean_slot_string
from scientific_contradiction_finder.extraction.finding_extractor import FindingExtractor
from scientific_contradiction_finder.schemas import DirectionLabel


def test_clean_slot_string_unclosed_parentheses():
    """Verify that broken comparator extraction with unclosed parentheses like 'Abstainers (P' is repaired."""
    assert clean_slot_string("Abstainers (P") == "Abstainers"
    assert clean_slot_string("Non-drinkers (") == "Non-drinkers"
    assert clean_slot_string("Placebo (p = 0.05)") == "Placebo"


def test_clean_slot_string_invalid_statistical_terms():
    """Verify that statistical metrics with leading determiners like 'The Relative Risk' or 'Our Aim' are rejected."""
    assert clean_slot_string("The Relative Risk") is None
    assert clean_slot_string("relative risk") is None
    assert clean_slot_string("The Odds Ratio") is None
    assert clean_slot_string("The Hazard Ratio") is None
    assert clean_slot_string("Our Findings") is None


def test_clean_slot_string_standalone_superlatives():
    """Verify that standalone superlatives like 'the lowest' or 'highest' are rejected as condition slots."""
    assert clean_slot_string("The Lowest") is None
    assert clean_slot_string("the lowest") is None
    assert clean_slot_string("The Highest") is None
    assert clean_slot_string("Highest") is None


def test_infer_direction_relative_risk():
    """Verify that passages with numerical relative risk ratios get proper directional inference."""
    extractor = FindingExtractor()

    # Relative risk > 1.0 indicates positive/elevated risk
    sent_pos = "The relative risk was 3.29 (95% confidence interval, 1.60-6.78) for the highest arsenic level."
    assert extractor.infer_direction(sent_pos) == DirectionLabel.POSITIVE

    # Hazard ratio < 1.0 indicates negative/reduced risk
    sent_neg = "The hazard ratio was 0.68 (95% CI, 0.52-0.89) for cardiovascular death."
    assert extractor.infer_direction(sent_neg) == DirectionLabel.NEGATIVE

    # Risk ratio ~ 1.0 indicates no difference
    sent_nodiff = "The odds ratio was 1.01 (95% CI, 0.88-1.15) showing no clear effect."
    assert extractor.infer_direction(sent_nodiff) == DirectionLabel.NO_DIFFERENCE


def test_rejection_of_participant_recruitment_methods():
    """Verify that sentences solely describing participant recruitment and allocation are rejected."""
    extractor = FindingExtractor()
    sent_recruit = "A total of 450 participants were enrolled and randomized to intervention arms."
    assert len(extractor.extract_evidence_spans(sent_recruit)) == 0


def test_conjunction_or_not_confused_with_odds_ratio():
    """Verify that lowercase English conjunction 'or' in comparison phrases is not misclassified as Odds Ratio (OR)."""
    extractor = FindingExtractor()

    # Phrase containing conjunction 'or = 3' should not be detected as empirical finding span or positive direction
    text_conj = "In patients followed for (< or = 3 months), coffee intake was evaluated."
    assert len(extractor.extract_evidence_spans(text_conj)) == 0
    assert extractor.infer_direction(text_conj) == DirectionLabel.UNCLEAR

    # Valid uppercase 'OR = ' and explicit 'odds ratio' should still be recognized properly
    text_or_upper = "Coffee consumption was associated with reduced risk (OR = 0.75, 95% CI 0.60-0.92)."
    assert len(extractor.extract_evidence_spans(text_or_upper)) == 1
    assert extractor.infer_direction(text_or_upper) == DirectionLabel.NEGATIVE

    text_or_pos = "Heavy intake was associated with elevated risk (OR = 2.40, 95% CI 1.30-4.10)."
    assert extractor.infer_direction(text_or_pos) == DirectionLabel.POSITIVE

    text_or_explicit = "The odds ratio was 0.82 for cardiovascular events."
    assert extractor.infer_direction(text_or_explicit) == DirectionLabel.NEGATIVE

    # Condition tagger should not extract a spurious OR numerical result
    from scientific_contradiction_finder.extraction.condition_tagger import ConditionTagger

    tagger = ConditionTagger()
    res = tagger.extract_conditions("The duration was (< or = 3 months) in coffee drinkers.")
    assert len(res.numerical_results) == 0


def test_rejection_of_meta_research_terms_and_introductory_clauses():
    """Verify that meta-terms like 'The Literature', 'Information', and 'Incidence' are rejected as condition slots."""
    from scientific_contradiction_finder.extraction.condition_tagger import (
        ConditionTagger,
        clean_slot_string,
    )

    # Unit checks on clean_slot_string
    for term in [
        "information",
        "Information",
        "the information",
        "literature",
        "The Literature",
        "most studies",
        "incidence",
        "The Incidence",
        "Prospectively ascertained information",
        "Based on the literature reviewed",
    ]:
        assert clean_slot_string(term) is None, f"Expected None for {term}, got {clean_slot_string(term)}"

    tagger = ConditionTagger()

    # Introductory review clause should not capture 'The Literature' as outcome
    res1 = tagger.extract_conditions(
        "Based on the literature reviewed, coffee intake reduced cardiovascular outcome."
    )
    assert res1.outcome_metric == "Cardiovascular Outcome"
    assert res1.intervention == "Coffee Intake"

    # Introductory methodology clause should not capture 'Prospectively Ascertained Information' as intervention or outcome
    res2 = tagger.extract_conditions(
        "Prospectively ascertained information indicated that coffee intake reduced cardiovascular outcome."
    )
    assert res2.outcome_metric == "Cardiovascular Outcome"
    assert res2.intervention == "Coffee Intake"

