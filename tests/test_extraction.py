from scientific_contradiction_finder.extraction.condition_tagger import ConditionTagger
from scientific_contradiction_finder.extraction.finding_extractor import FindingExtractor
from scientific_contradiction_finder.extraction.sentence_splitter import SentenceSplitter
from scientific_contradiction_finder.schemas import DirectionLabel


def test_sentence_splitter_exact_offsets():
    abstract = "Metformin is a biguanide. In our randomized trial, metformin reduced HbA1c by 1.1%. Placebo showed no effect."
    sentences = SentenceSplitter.split_with_offsets(abstract)
    assert len(sentences) == 3

    for sent_text, start, end in sentences:
        assert abstract[start:end] == sent_text


def test_finding_extractor_provenance():
    abstract = (
        "Background: Type 2 diabetes is prevalent worldwide. "
        "Methods: We conducted a double-blind trial. "
        "Results: Metformin demonstrated significant HbA1c reduction of 1.2% versus placebo. "
        "Conclusion: Metformin is effective."
    )
    extractor = FindingExtractor()
    spans = extractor.extract_evidence_spans(abstract)
    assert len(spans) > 0

    # Ensure every extracted span strictly satisfies substring provenance
    for span in spans:
        assert span.verify_against_source(abstract)
        assert abstract[span.start_char : span.end_char] == span.text


def test_direction_inference():
    extractor = FindingExtractor()
    assert (
        extractor.infer_direction("RoBERTa significantly outperformed BERT on SQuAD.")
        == DirectionLabel.POSITIVE
    )
    assert (
        extractor.infer_direction("Statins reduced major cardiovascular events.")
        == DirectionLabel.NEGATIVE
    )
    assert (
        extractor.infer_direction(
            "Hydroxychloroquine showed no significant difference in 28-day mortality."
        )
        == DirectionLabel.NO_DIFFERENCE
    )


def test_condition_tagger():
    tagger = ConditionTagger()
    sent = "Low-dose aspirin reduced recurrent myocardial infarction compared to placebo in adults with coronary disease."
    conditions = tagger.extract_conditions(sent)

    assert "aspirin" in conditions.intervention.lower()
    assert "placebo" in conditions.comparator.lower()
    assert conditions.is_unknown is False
