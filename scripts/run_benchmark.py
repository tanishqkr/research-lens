#!/usr/bin/env python3
"""Run the 30-query evaluation benchmark against the live Scientific Contradiction Finder pipeline.

Saves raw execution outputs to data/eval_benchmark/runs/<run_id>/ and computes deterministic
benchmark metrics across retrieval, claim grounding, relationship classification, abstention,
and summary consistency.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure package is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from scientific_contradiction_finder.pipeline import ContradictionFinderPipeline  # noqa: E402
from scientific_contradiction_finder.schemas import ResearchReport  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_benchmark")


def load_benchmark(benchmark_path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_single_run(benchmark_item: Dict[str, Any], report_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministically evaluates a single report output against benchmark expectations."""
    category = benchmark_item["category"]
    expected_rel = benchmark_item["expected_relationship"]

    exec_summary = report_dict.get("executive_summary", {})
    subclaims = report_dict.get("subclaim_assessments", [])
    ledger = report_dict.get("evidence_ledger", [])
    overall_assessment = exec_summary.get("overall_evidence_assessment", "").lower()

    # 1. Retrieval Quality
    retrieved_count = report_dict.get("candidate_papers_retrieved", 0)
    deep_dive_count = report_dict.get("papers_deep_dived", 0)
    ledger_count = len(ledger)

    if category == "irrelevant_implausible":
        # For implausible queries, success is 0 supported findings or clear abstention
        retrieval_pass = (ledger_count == 0) or all(s.get("status") == "UNRESOLVED" for s in subclaims)
    else:
        # Standard queries should successfully retrieve literature
        retrieval_pass = retrieved_count > 0 and (deep_dive_count > 0 or ledger_count > 0)

    # 2. Claim-Level Grounding
    grounded_claims = 0
    total_ledger_claims = len(ledger)
    for c in ledger:
        passage = c.get("exact_source_passage", "").strip()
        if passage and len(passage) > 20:
            grounded_claims += 1

    claim_grounding_score = 1.0 if total_ledger_claims == 0 else (grounded_claims / total_ledger_claims)

    # 3. Relationship Classification Alignment
    has_pos = any(c.get("direction") == "positive" for c in ledger)
    has_neg = any(c.get("direction") == "negative" for c in ledger)
    has_no_diff = any(c.get("direction") == "no_difference" for c in ledger)
    has_supported = any(s.get("status") == "SUPPORTED" for s in subclaims)
    all_unresolved = all(s.get("status") == "UNRESOLVED" for s in subclaims) if subclaims else True

    rel_score = 0.0
    if category == "irrelevant_implausible":
        if total_ledger_claims == 0 or all_unresolved or "inconclusive" in overall_assessment:
            rel_score = 1.0
        else:
            rel_score = 0.0

    elif category == "unresolved_insufficient":
        if all_unresolved or "inconclusive" in overall_assessment or "unresolved" in overall_assessment:
            rel_score = 1.0
        elif has_no_diff and not has_supported:
            rel_score = 1.0
        else:
            rel_score = 0.5 if "context-dependent" in overall_assessment else 0.0

    elif category == "consistent_finding":
        if "broadly supports" in overall_assessment and has_supported:
            rel_score = 1.0
        elif has_supported:
            rel_score = 0.75
        elif all_unresolved:
            rel_score = 0.5
        else:
            rel_score = 0.0

    elif category == "genuine_contradiction":
        if (
            "divergence" in overall_assessment
            or "disagreement" in overall_assessment
            or "context-dependent" in overall_assessment
        ):
            rel_score = 1.0
        elif (has_pos and (has_neg or has_no_diff)) or (has_neg and has_no_diff):
            rel_score = 1.0
        elif any(s.get("status") == "QUALIFIED" for s in subclaims):
            rel_score = 0.75
        elif all_unresolved:
            rel_score = 0.5
        else:
            rel_score = 0.25

    elif category in ("context_dependent", "conditional_comparative"):
        if "context-dependent" in overall_assessment or "qualified" in overall_assessment:
            rel_score = 1.0
        elif any(s.get("status") == "QUALIFIED" for s in subclaims):
            rel_score = 1.0
        elif (has_pos and has_neg) or (has_pos and has_no_diff):
            rel_score = 1.0
        elif all_unresolved:
            rel_score = 0.5
        else:
            rel_score = 0.25

    # 4. Abstention & False Confidence
    abstention_score = 1.0
    if category == "irrelevant_implausible" and has_supported:
        abstention_score = 0.0
    elif category == "unresolved_insufficient" and (has_supported and not has_no_diff):
        abstention_score = 0.0

    # 5. Summary Consistency
    summary_consistency_score = 1.0
    if all_unresolved and "broadly supports" in overall_assessment and has_supported is False:
        summary_consistency_score = 0.0

    # Composite Score
    composite_score = (
        (1.0 if retrieval_pass else 0.0) * 0.20
        + claim_grounding_score * 0.25
        + rel_score * 0.25
        + abstention_score * 0.15
        + summary_consistency_score * 0.15
    )

    return {
        "id": benchmark_item["id"],
        "category": category,
        "query": benchmark_item["query"],
        "expected_relationship": expected_rel,
        "overall_evidence_assessment": overall_assessment,
        "retrieved_count": retrieved_count,
        "deep_dive_count": deep_dive_count,
        "ledger_count": ledger_count,
        "subclaim_statuses": [s.get("status") for s in subclaims],
        "retrieval_pass": bool(retrieval_pass),
        "claim_grounding_score": round(claim_grounding_score, 3),
        "relationship_score": round(rel_score, 3),
        "abstention_score": round(abstention_score, 3),
        "summary_consistency_score": round(summary_consistency_score, 3),
        "composite_score": round(composite_score, 3),
    }


async def run_benchmark_suite(
    benchmark_file: Path,
    output_dir: Path,
    limit_queries: int | None = None,
    delay_between_runs: float = 2.0,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    items = load_benchmark(benchmark_file)
    if limit_queries:
        items = items[:limit_queries]

    logger.info(f"Loaded {len(items)} queries from {benchmark_file}")
    pipeline = ContradictionFinderPipeline(use_groq=True)

    run_evaluations = []
    run_start_time = time.time()

    for idx, item in enumerate(items, 1):
        q_id = item["id"]
        query = item["query"]
        category = item["category"]
        logger.info(f"[{idx}/{len(items)}] Running {q_id} ({category}): '{query[:65]}...'")

        t0 = time.time()
        error = None
        report_dict = {}

        try:
            report: ResearchReport = await pipeline.analyze_as_research_analyst(
                query=query, limit=10, max_deep_dives=5
            )
            report_dict = report.model_dump()
        except Exception as e:
            logger.error(f"Error executing {q_id}: {e}", exc_info=True)
            error = str(e)
            report_dict = {
                "query": query,
                "error": str(e),
                "candidate_papers_retrieved": 0,
                "papers_deep_dived": 0,
                "evidence_ledger": [],
                "subclaim_assessments": [],
                "executive_summary": {
                    "overall_evidence_assessment": "error",
                    "bottom_line": f"Execution failed: {e}",
                },
            }

        elapsed = round(time.time() - t0, 2)
        logger.info(f"[{idx}/{len(items)}] Finished {q_id} in {elapsed}s (error={error is not None})")

        # Save individual run file
        report_file = output_dir / f"{q_id}_{category}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        # Deterministic evaluation
        eval_result = evaluate_single_run(item, report_dict)
        eval_result["latency_seconds"] = elapsed
        eval_result["error"] = error
        run_evaluations.append(eval_result)

        if idx < len(items) and delay_between_runs > 0:
            await asyncio.sleep(delay_between_runs)

    total_duration = round(time.time() - run_start_time, 2)

    # Compute Aggregate Metrics
    total_q = len(run_evaluations)
    avg_composite = sum(e["composite_score"] for e in run_evaluations) / total_q if total_q else 0.0
    avg_retrieval_pass = sum(1 for e in run_evaluations if e["retrieval_pass"]) / total_q if total_q else 0.0
    avg_grounding = sum(e["claim_grounding_score"] for e in run_evaluations) / total_q if total_q else 0.0
    avg_relationship = sum(e["relationship_score"] for e in run_evaluations) / total_q if total_q else 0.0
    avg_abstention = sum(e["abstention_score"] for e in run_evaluations) / total_q if total_q else 0.0
    avg_consistency = sum(e["summary_consistency_score"] for e in run_evaluations) / total_q if total_q else 0.0

    # Per-category scores
    categories = sorted(list(set(e["category"] for e in run_evaluations)))
    category_metrics = {}
    for cat in categories:
        cat_evals = [e for e in run_evaluations if e["category"] == cat]
        c_count = len(cat_evals)
        category_metrics[cat] = {
            "count": c_count,
            "avg_composite": round(sum(e["composite_score"] for e in cat_evals) / c_count, 3),
            "retrieval_pass_rate": round(sum(1 for e in cat_evals if e["retrieval_pass"]) / c_count, 3),
            "avg_relationship_score": round(sum(e["relationship_score"] for e in cat_evals) / c_count, 3),
            "avg_abstention_score": round(sum(e["abstention_score"] for e in cat_evals) / c_count, 3),
        }

    summary_metrics = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_queries": total_q,
        "total_duration_seconds": total_duration,
        "overall": {
            "mean_composite_score": round(avg_composite, 3),
            "retrieval_pass_rate": round(avg_retrieval_pass, 3),
            "claim_grounding_rate": round(avg_grounding, 3),
            "relationship_classification_score": round(avg_relationship, 3),
            "abstention_accuracy": round(avg_abstention, 3),
            "summary_consistency_rate": round(avg_consistency, 3),
        },
        "by_category": category_metrics,
        "evaluations": run_evaluations,
    }

    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    logger.info(f"Evaluation complete. Metrics saved to {metrics_file}")
    logger.info(f"Mean Composite Score: {avg_composite:.3f}")
    return summary_metrics


def main():
    parser = argparse.ArgumentParser(description="Run 30-Query Benchmark on Scientific Contradiction Finder")
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=repo_root / "data" / "eval_benchmark" / "benchmark_v1.jsonl",
        help="Path to benchmark JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for run results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries (for testing/smoke run)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between successive queries to respect API rate limits",
    )

    args = parser.parse_args()
    if args.output_dir is None:
        run_timestamp = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        output_dir = repo_root / "data" / "eval_benchmark" / "runs" / run_timestamp
    else:
        output_dir = args.output_dir

    asyncio.run(
        run_benchmark_suite(
            benchmark_file=args.benchmark_file,
            output_dir=output_dir,
            limit_queries=args.limit,
            delay_between_runs=args.delay,
        )
    )


if __name__ == "__main__":
    main()
