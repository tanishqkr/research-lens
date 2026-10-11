"""Comparative evaluation of Local vs Groq-Assisted query structuring and extraction across the 3 user queries."""

import asyncio
import json

from scientific_contradiction_finder.pipeline import ContradictionFinderPipeline

QUERIES = [
    (
        "Query 1 (Battery degradation)",
        "Do lithium-ion batteries degrade faster when charged at high temperatures compared with moderate temperatures?",
    ),
    (
        "Query 2 (Vision transformers)",
        "Does image augmentation improve image classification accuracy in vision transformer models compared with training without augmentation?",
    ),
    (
        "Query 3 (Vitamin D & respiratory)",
        "Does vitamin D supplementation reduce the risk of acute respiratory infections, and does the effect depend on baseline vitamin D levels?",
    ),
]


async def evaluate_comparative():
    pipeline_local = ContradictionFinderPipeline(use_groq=False)
    pipeline_groq = ContradictionFinderPipeline(use_groq=True)

    results = []

    print("\n=======================================================")
    print("COMPARATIVE EVALUATION: LOCAL VS GROQ-ASSISTED")
    print("=======================================================\n")

    for label, q in QUERIES:
        print(f"--- Evaluating {label} ---")
        print(f'Query: "{q}"')

        # 1. Local only
        res_local = await pipeline_local.run(q, limit=10)
        # 2. Groq assisted
        res_groq = await pipeline_groq.run(q, limit=10)

        local_diag = res_local["diagnostics"]
        groq_diag = res_groq["diagnostics"]

        print(
            f"Local Retrieval: OpenAlex={local_diag['openalex']['results_count']}, arXiv={local_diag['arxiv']['results_count']}, Usable={local_diag['usable_abstracts']}, Findings={res_local['findings_extracted']}, Pairs={len(res_local['comparisons'])}"
        )
        print(
            f"Groq Retrieval:  OpenAlex={groq_diag['openalex']['results_count']}, arXiv={groq_diag['arxiv']['results_count']}, Usable={groq_diag['usable_abstracts']}, Findings={res_groq['findings_extracted']}, Pairs={len(res_groq['comparisons'])}"
        )
        if groq_diag.get("groq_structured"):
            print(f"  Groq Keywords: {groq_diag['groq_structured']['search_keywords']}")

        results.append(
            {
                "label": label,
                "query": q,
                "local": {
                    "openalex_count": local_diag["openalex"]["results_count"],
                    "arxiv_count": local_diag["arxiv"]["results_count"],
                    "usable_abstracts": local_diag["usable_abstracts"],
                    "findings_extracted": res_local["findings_extracted"],
                    "pairs_count": len(res_local["comparisons"]),
                    "sample_papers": [p.title for p in res_local["papers"][:2]],
                    "sample_findings": [
                        {
                            "claim": f.finding_claim[:100],
                            "I": f.conditions.intervention,
                            "O": f.conditions.outcome_metric,
                        }
                        for f in res_local["findings"][:2]
                    ],
                    "sample_verdicts": [
                        {
                            "verdict": pred.predicted_relation.value,
                            "conf": pred.confidence_score,
                            "expl": pred.explanation,
                        }
                        for _, pred in res_local["comparisons"][:2]
                    ],
                },
                "groq": {
                    "openalex_count": groq_diag["openalex"]["results_count"],
                    "arxiv_count": groq_diag["arxiv"]["results_count"],
                    "usable_abstracts": groq_diag["usable_abstracts"],
                    "findings_extracted": res_groq["findings_extracted"],
                    "pairs_count": len(res_groq["comparisons"]),
                    "groq_keywords": groq_diag["groq_structured"]["search_keywords"]
                    if groq_diag.get("groq_structured")
                    else [],
                    "sample_papers": [p.title for p in res_groq["papers"][:2]],
                    "sample_findings": [
                        {
                            "claim": f.finding_claim[:100],
                            "I": f.conditions.intervention,
                            "O": f.conditions.outcome_metric,
                        }
                        for f in res_groq["findings"][:2]
                    ],
                    "sample_verdicts": [
                        {
                            "verdict": pred.predicted_relation.value,
                            "conf": pred.confidence_score,
                            "expl": pred.explanation,
                        }
                        for _, pred in res_groq["comparisons"][:2]
                    ],
                },
            }
        )
        print()

    with open("data/groq_evaluation_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Comparative evaluation finished. Results saved to data/groq_evaluation_comparison.json")


if __name__ == "__main__":
    asyncio.run(evaluate_comparative())
