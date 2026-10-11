from scientific_contradiction_finder.pipeline import ContradictionFinderPipeline
from scientific_contradiction_finder.schemas import Paper


def test_pipeline_extract_and_compare():
    pipeline = ContradictionFinderPipeline()

    p1 = Paper(
        paper_id="paper_1",
        title="Impact of Metformin in Severe Diabetes",
        abstract="In this double-blind RCT, metformin reduced HbA1c significantly by 1.2% versus placebo in severe T2D patients.",
        source_provider="openalex",
    )
    p2 = Paper(
        paper_id="paper_2",
        title="Evaluation of Metformin in Mild Prediabetes",
        abstract="In our community trial, metformin did not reduce HbA1c significantly compared to placebo in mild prediabetic subjects.",
        source_provider="openalex",
    )

    findings = pipeline.extract_findings_from_papers([p1, p2])
    assert len(findings) == 2
    assert findings[0].evidence.verify_against_source(p1.abstract)
    assert findings[1].evidence.verify_against_source(p2.abstract)

    comparisons = pipeline.compare_findings(findings)
    assert len(comparisons) == 1
    pair, pred = comparisons[0]
    assert pred.provenance_verified is True
    print(f"Predicted relationship between p1 and p2: {pred.predicted_relation.value}")
