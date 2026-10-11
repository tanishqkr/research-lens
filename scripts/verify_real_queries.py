"""Execute and trace all 3 real scientific queries end-to-end through the pipeline."""

import asyncio
import json

from scientific_contradiction_finder.pipeline import ContradictionFinderPipeline

QUERIES = [
    (
        "Query A (Battery science)",
        "Do higher temperatures accelerate lithium-ion battery degradation?",
    ),
    (
        "Query B (Machine learning)",
        "Does data augmentation improve the generalization of vision transformers?",
    ),
    (
        "Query C (Medical research)",
        "Does vitamin D supplementation reduce respiratory infections in adults?",
    ),
]


async def run_diagnostics():
    pipeline = ContradictionFinderPipeline()
    results = []

    print("\n=======================================================")
    print("MANDATORY END-TO-END RUNTIME VERIFICATION (3 QUERIES)")
    print("=======================================================\n")

    for label, q in QUERIES:
        print(f"--- Running {label} ---")
        print(f'Query: "{q}"')
        res = await pipeline.run(q, limit=10)

        diag = res["diagnostics"]
        oa_count = diag["openalex"]["results_count"]
        oa_status = diag["openalex"]["status"]
        ax_count = diag["arxiv"]["results_count"]
        ax_status = diag["arxiv"]["status"]
        usable = diag["usable_abstracts"]
        findings_count = res["findings_extracted"]
        comparisons_count = len(res["comparisons"])

        print(
            f"Diagnostics: OpenAlex={oa_count} (status {oa_status}), arXiv={ax_count} (status {ax_status})"
        )
        print(
            f"Usable Abstracts: {usable}, Extracted Findings: {findings_count}, Candidate Pairs: {comparisons_count}"
        )

        sample_papers = []
        for p in res["papers"][:3]:
            sample_papers.append(
                {
                    "id": p.paper_id,
                    "title": p.title,
                    "provider": p.source_provider,
                    "doi_or_url": p.doi or p.url,
                }
            )
            print(f"   [Paper] ({p.source_provider}) {p.title}")

        sample_findings = []
        for f in res["findings"][:3]:
            sample_findings.append(
                {
                    "claim": f.finding_claim,
                    "direction": f.direction.value,
                    "conditions": f.conditions.model_dump(),
                }
            )
            print(f'   [Evidence] "{f.finding_claim[:120]}..."')
            print(
                f"      Direction: {f.direction.value} | I: {f.conditions.intervention} | O: {f.conditions.outcome_metric}"
            )

        sample_verdicts = []
        for pair, pred in res["comparisons"][:3]:
            sample_verdicts.append(
                {
                    "pair_id": pair.pair_id,
                    "verdict": pred.predicted_relation.value,
                    "confidence": pred.confidence_score,
                    "explanation": pred.explanation,
                }
            )
            print(
                f"   [Verdict] {pred.predicted_relation.value} (conf={pred.confidence_score:.2f})"
            )
            print(f"      Explanation: {pred.explanation}")

        results.append(
            {
                "label": label,
                "query": q,
                "openalex_count": oa_count,
                "openalex_status": oa_status,
                "arxiv_count": ax_count,
                "arxiv_status": ax_status,
                "usable_abstracts": usable,
                "extracted_findings": findings_count,
                "comparisons_count": comparisons_count,
                "sample_papers": sample_papers,
                "sample_findings": sample_findings,
                "sample_verdicts": sample_verdicts,
            }
        )
        print()

    with open("data/runtime_verification_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Completed verification. Saved to data/runtime_verification_results.json")


if __name__ == "__main__":
    asyncio.run(run_diagnostics())
