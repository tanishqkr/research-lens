"""Run the 6 Fixed Acceptance Queries (A through F) End-to-End.

Records complete outputs, verifies exact source passages, compares against
the unconditioned baseline where pairs exist, and exports structured results.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from scientific_contradiction_finder.ingestion.dataset_loader import BenchmarkLoader
from scientific_contradiction_finder.pipeline import ContradictionFinderPipeline
from scientific_contradiction_finder.reasoning.baseline import LexicalUnconditionedBaseline

ACCEPTANCE_QUERIES = [
    {
        "id": "Query_A",
        "name": "Diet Durability (Low-Carb vs Low-Fat)",
        "query": "Do randomized controlled trials show that low-carbohydrate diets produce greater weight loss than low-fat diets, and does the difference persist over time?",
        "focus": "Multi-arm attribution, short vs long-term persistence, null findings",
    },
    {
        "id": "Query_B",
        "name": "Alcohol & Mortality Confounding",
        "query": "Does moderate alcohol consumption reduce all-cause mortality, or can the apparent protective association be explained by study design and confounding?",
        "focus": "Observational vs trial classification, association vs causation, confounding",
    },
    {
        "id": "Query_C",
        "name": "Intermittent Fasting vs Continuous Restriction",
        "query": "Does intermittent fasting produce greater long-term weight loss than continuous calorie restriction when energy intake is comparable?",
        "focus": "Isocaloric comparison alignment, adherence, context-dependent vs contradiction",
    },
    {
        "id": "Query_D",
        "name": "Vitamin D Trial vs Observational CVD",
        "query": "Does vitamin D supplementation prevent cardiovascular events in generally healthy adults, and how do randomized trials compare with observational findings?",
        "focus": "RCT null results vs observational inverse correlation, CVD endpoint matching",
    },
    {
        "id": "Query_E",
        "name": "Pediatric Sepsis Machine Learning Prediction",
        "query": "Which machine-learning approaches have been evaluated for early prediction of pediatric sepsis, and how reliable is the evidence that one approach outperforms another?",
        "focus": "AUROC discrimination vs clinical outcomes, comparative model claims, prediction horizon",
    },
    {
        "id": "Query_F",
        "name": "Deliberately Sparse / Insufficient Evidence",
        "query": "Does high-dose intravenous curcumin improve survival in severe acute pancreatitis?",
        "focus": "Sparse evidence handling, safe abstention, zero hallucination of papers",
    },
]


async def run_suite():
    print("=" * 70)
    print("STARTING SCIENTIFIC CONTRADICTION FINDER ACCEPTANCE SUITE (QUERIES A–F)")
    print("=" * 70)

    # Initialize and fit unconditioned baseline on benchmark dataset
    print("\n[Init] Training Lexical Unconditioned Baseline on benchmark pairs...")
    loader = BenchmarkLoader()
    all_pairs = loader.load_cas_con150()
    train_pairs, _ = loader.get_train_test_split(all_pairs, test_ratio=0.3, random_seed=42)
    baseline = LexicalUnconditionedBaseline()
    baseline.fit(train_pairs)
    print(f"[Init] Baseline trained on {len(train_pairs)} pairs.")

    pipeline = ContradictionFinderPipeline(use_groq=False)  # Local deterministic mode for reproducible offline verification

    suite_results = []
    output_dir = Path("data/acceptance_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in ACCEPTANCE_QUERIES:
        qid = item["id"]
        qname = item["name"]
        query_text = item["query"]
        print(f"\n>>> Running {qid}: {qname}")
        print(f"    Query: '{query_text}'")

        start_time = time.time()
        try:
            res = await pipeline.run(query_text, limit=10, use_groq_interpretation=False)
            elapsed = time.time() - start_time
        except Exception as e:
            print(f"    ERROR running query: {e}")
            suite_results.append({
                "id": qid,
                "name": qname,
                "query": query_text,
                "status": "ERROR",
                "error": str(e),
            })
            continue

        report = res.get("research_report")
        diag = res.get("diagnostics", {})
        oa_diag = diag.get("openalex", {})
        ax_diag = diag.get("arxiv", {})

        print(f"    Elapsed: {elapsed:.2f}s | Papers: {res['papers_found']} | Usable: {diag.get('usable_abstracts', 0)} | Ledger Claims: {len(report.evidence_ledger) if report else 0}")
        if report and report.executive_summary:
            print(f"    Bottom Line: {report.executive_summary.bottom_line[:120]}...")
            print(f"    Evidence Assessment: {report.executive_summary.overall_evidence_assessment}")

        # Baseline comparison on extracted pairs
        baseline_comparisons = []
        if res["comparisons"]:
            pairs_to_test = [pair for pair, _ in res["comparisons"]]
            baseline_preds = baseline.predict(pairs_to_test)
            for (pair, ca_pred), base_pred in zip(res["comparisons"], baseline_preds):
                baseline_comparisons.append({
                    "pair_id": pair.pair_id,
                    "study_a": pair.finding_a.paper_title[:45],
                    "study_b": pair.finding_b.paper_title[:45],
                    "condition_aware_verdict": ca_pred.predicted_relation.value,
                    "condition_aware_confidence": ca_pred.confidence_score,
                    "condition_differences": ca_pred.condition_differences.summary_of_differences,
                    "unconditioned_baseline_verdict": base_pred.predicted_relation.value,
                    "unconditioned_baseline_confidence": base_pred.confidence_score,
                })

        record = {
            "id": qid,
            "name": qname,
            "query": query_text,
            "focus": item["focus"],
            "elapsed_seconds": round(elapsed, 2),
            "diagnostics": {
                "openalex_count": oa_diag.get("results_count", 0),
                "openalex_status": oa_diag.get("status"),
                "arxiv_count": ax_diag.get("results_count", 0),
                "arxiv_status": ax_diag.get("status"),
                "usable_abstracts": diag.get("usable_abstracts", 0),
                "findings_extracted": res.get("findings_extracted", 0),
                "candidate_pairs_compared": len(res.get("comparisons", [])),
            },
            "executive_summary": (
                {
                    "bottom_line": report.executive_summary.bottom_line,
                    "overall_evidence_assessment": report.executive_summary.overall_evidence_assessment,
                    "strongest_supporting_evidence": report.executive_summary.strongest_supporting_evidence,
                    "strongest_conflicting_evidence": report.executive_summary.strongest_conflicting_evidence,
                    "why_studies_disagree": report.executive_summary.why_studies_disagree,
                    "confidence_and_limitations": report.executive_summary.confidence_and_limitations,
                    "research_coverage": report.executive_summary.research_coverage,
                }
                if report and report.executive_summary
                else None
            ),
            "subclaim_assessments": (
                [
                    {
                        "subclaim": sc.subclaim_text,
                        "status": sc.status.value,
                        "rationale": sc.rationale,
                        "evidence_paper_ids": sc.evidence_paper_ids,
                    }
                    for sc in (report.subclaim_assessments or (report.executive_summary.subclaim_assessments if report.executive_summary else []))
                ]
                if report
                else []
            ),
            "evidence_ledger": (
                [
                    {
                        "paper_id": c.paper_id,
                        "paper_title": c.paper_title,
                        "doi_or_url": c.url or c.doi,
                        "study_type": c.study_type.value,
                        "exact_source_passage": c.exact_source_passage,
                        "intervention_exposure": c.intervention_exposure,
                        "comparator": c.comparator,
                        "outcome": c.outcome,
                        "direction": c.direction.value,
                        "population_setting": c.population_setting,
                        "follow_up_duration": c.follow_up_duration,
                        "numerical_results": [nr.model_dump() for nr in c.numerical_results],
                    }
                    for c in report.evidence_ledger
                ]
                if report
                else []
            ),
            "comparisons": baseline_comparisons,
        }
        suite_results.append(record)

    # Save complete JSON
    out_file = output_dir / "acceptance_run.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"suite": suite_results}, f, indent=2)

    print("\n" + "=" * 70)
    print(f"ACCEPTANCE SUITE COMPLETE. Saved to {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_suite())
