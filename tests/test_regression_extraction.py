from scientific_contradiction_finder.extraction.condition_tagger import ConditionTagger
from scientific_contradiction_finder.extraction.finding_extractor import FindingExtractor

FAILED_PASSAGES = [
    # 1. Non-empirical / background claim that must be rejected
    (
        "Vision Transformer models have demonstrated great promise across computer vision tasks.",
        False,
    ),
    # 2. Battery sentence where thickness increase shouldn't be tagged as intervention='significant increases'
    (
        "The results indicate that significant increases in cell thickness occur during high-temperature cycling due to active material degradation.",
        True,
    ),
    # 3. Follow-up study sentence where intervention shouldn't be 'In the 14-week follow-up'
    (
        "In the 14-week follow-up study, baseline 25(OH)D levels below 25 nmol/L were associated with a 42% increased risk of acute respiratory infection compared to adequate levels.",
        True,
    ),
]


def test_regression_passages():
    extractor = FindingExtractor()
    tagger = ConditionTagger()

    # Case 1: Vision transformer promise must be rejected
    spans_1 = extractor.extract_evidence_spans(FAILED_PASSAGES[0][0])
    assert len(spans_1) == 0, "Non-empirical promise sentence should have been rejected!"

    # Case 2: Battery degradation
    spans_2 = extractor.extract_evidence_spans(FAILED_PASSAGES[1][0])
    assert len(spans_2) > 0
    cond_2 = tagger.extract_conditions(spans_2[0].text)
    assert cond_2.intervention != "significant increases"
    assert "thickness" in cond_2.outcome_metric.lower()

    # Case 3: Vitamin D follow-up study
    spans_3 = extractor.extract_evidence_spans(FAILED_PASSAGES[2][0])
    assert len(spans_3) > 0
    cond_3 = tagger.extract_conditions(spans_3[0].text)
    assert cond_3.intervention != "In the 14-week follow-up"
    assert "25(oh)d" in cond_3.intervention.lower()
    assert "adequate levels" in cond_3.comparator.lower()


if __name__ == "__main__":
    test_regression_passages()
    print("Regression test passed successfully!")
