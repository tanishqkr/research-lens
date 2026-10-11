"""Full evaluation and benchmark comparison script generating docs/EVALUATION.md results."""

import json
from pathlib import Path

from scientific_contradiction_finder.evaluation.evaluator import PipelineEvaluator
from scientific_contradiction_finder.ingestion.dataset_loader import BenchmarkLoader
from scientific_contradiction_finder.reasoning.baseline import LexicalUnconditionedBaseline
from scientific_contradiction_finder.reasoning.classifier import ConditionAwareRelationClassifier


def main():
    loader = BenchmarkLoader()
    pairs = loader.load_cas_con150()
    train_pairs, test_pairs = loader.get_train_test_split(pairs, test_ratio=0.3, random_seed=42)

    evaluator = PipelineEvaluator()

    # 1. Unconditioned Baseline
    baseline = LexicalUnconditionedBaseline()
    baseline.fit(train_pairs)
    baseline_preds = baseline.predict(test_pairs)
    baseline_result = evaluator.evaluate_predictions(
        test_pairs, baseline_preds, dataset_name="Unconditioned TF-IDF Baseline"
    )

    # 2. Full Condition-Aware Model
    condition_clf = ConditionAwareRelationClassifier()
    ca_preds = condition_clf.classify_all(test_pairs)
    ca_result = evaluator.evaluate_predictions(
        test_pairs, ca_preds, dataset_name="Condition-Aware Pipeline (Proposed)"
    )

    # 3. Ablations
    ablations = evaluator.run_ablation_study(test_pairs)

    summary = {
        "dataset": "CAS-Con150",
        "n_train": len(train_pairs),
        "n_test": len(test_pairs),
        "unconditioned_baseline": baseline_result.model_dump(),
        "condition_aware_model": ca_result.model_dump(),
        "ablations": ablations,
    }

    out_file = Path("data/evaluation_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Test Pairs: {len(test_pairs)}")
    print(
        f"Unconditioned Baseline Accuracy: {baseline_result.accuracy:.4f} | Macro-F1: {baseline_result.macro_f1:.4f}"
    )
    print(
        f"Condition-Aware Model  Accuracy: {ca_result.accuracy:.4f} | Macro-F1: {ca_result.macro_f1:.4f}"
    )
    print(f"Ablation (No Setting)  Macro-F1: {ablations['no_setting_ablation']['macro_f1']:.4f}")
    print(
        f"Ablation (No Incomp)   Macro-F1: {ablations['no_incomparable_ablation']['macro_f1']:.4f}"
    )
    print("====================================================\n")


if __name__ == "__main__":
    main()
