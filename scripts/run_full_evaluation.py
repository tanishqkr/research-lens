"""Full evaluation and benchmark comparison script generating corrected results.

Evaluates:
1. CAS-Con150 with Leakage-Safe Claim Family Grouping (Sound Train/Test Split)
2. CAS-Con150 Historical Split (Demonstrating original shortcut/leakage artifact)
3. Independent Evaluation Benchmark (35 distinct primary literature pairs)
"""

import json
from pathlib import Path

from scientific_contradiction_finder.evaluation.evaluator import PipelineEvaluator
from scientific_contradiction_finder.ingestion.dataset_loader import BenchmarkLoader
from scientific_contradiction_finder.reasoning.baseline import LexicalUnconditionedBaseline
from scientific_contradiction_finder.reasoning.classifier import ConditionAwareRelationClassifier


def main():
    loader = BenchmarkLoader()
    evaluator = PipelineEvaluator()

    cas_pairs = loader.load_cas_con150()

    # =========================================================================
    # 1. CAS-Con150 Leakage-Safe Evaluation (Claim-Family Grouped)
    # =========================================================================
    print("\n--- 1. Evaluating CAS-Con150 (Leakage-Safe Split) ---")
    ls_train, ls_test = loader.get_train_test_split(
        cas_pairs, test_ratio=0.3, random_seed=42, leakage_safe=True
    )

    baseline_ls = LexicalUnconditionedBaseline()
    baseline_ls.fit(ls_train)
    baseline_ls_preds = baseline_ls.predict(ls_test)
    baseline_ls_res = evaluator.evaluate_predictions(
        ls_test, baseline_ls_preds, dataset_name="CAS-Con150 Leakage-Safe (TF-IDF Baseline)"
    )

    ca_clf = ConditionAwareRelationClassifier()
    ca_ls_preds = ca_clf.classify_all(ls_test)
    ca_ls_res = evaluator.evaluate_predictions(
        ls_test, ca_ls_preds, dataset_name="CAS-Con150 Leakage-Safe (Condition-Aware Pipeline)"
    )

    ablations_ls = evaluator.run_ablation_study(ls_test)

    # =========================================================================
    # 2. CAS-Con150 Historical Split (For direct historical comparison)
    # =========================================================================
    print("\n--- 2. Evaluating CAS-Con150 (Historical Leaky Split) ---")
    hist_train, hist_test = loader.get_train_test_split(
        cas_pairs, test_ratio=0.3, random_seed=42, leakage_safe=False
    )

    baseline_hist = LexicalUnconditionedBaseline()
    baseline_hist.fit(hist_train)
    baseline_hist_preds = baseline_hist.predict(hist_test)
    baseline_hist_res = evaluator.evaluate_predictions(
        hist_test, baseline_hist_preds, dataset_name="CAS-Con150 Historical (TF-IDF Baseline)"
    )

    ca_hist_preds = ca_clf.classify_all(hist_test)
    ca_hist_res = evaluator.evaluate_predictions(
        hist_test, ca_hist_preds, dataset_name="CAS-Con150 Historical (Condition-Aware Pipeline)"
    )

    # =========================================================================
    # 3. Independent Evaluation Benchmark (35 Genuinely Unique Pairs)
    # =========================================================================
    print("\n--- 3. Evaluating Independent Benchmark (35 Unique Real Literature Pairs) ---")
    indep_pairs = loader.load_independent_eval_set()

    # Evaluate TF-IDF trained on full CAS-Con150 on independent pairs
    baseline_indep = LexicalUnconditionedBaseline()
    baseline_indep.fit(cas_pairs)
    baseline_indep_preds = baseline_indep.predict(indep_pairs)
    baseline_indep_res = evaluator.evaluate_predictions(
        indep_pairs, baseline_indep_preds, dataset_name="Independent Benchmark (TF-IDF Baseline)"
    )

    ca_indep_preds = ca_clf.classify_all(indep_pairs)
    ca_indep_res = evaluator.evaluate_predictions(
        indep_pairs, ca_indep_preds, dataset_name="Independent Benchmark (Condition-Aware Pipeline)"
    )

    ablations_indep = evaluator.run_ablation_study(indep_pairs)

    # =========================================================================
    # Summary Assembly and Storage
    # =========================================================================
    summary = {
        "metadata": {
            "evaluation_date": "2026-10-12",
            "evaluator": "scientific_contradiction_finder.evaluation.evaluator.PipelineEvaluator",
            "description": "Corrected evaluation suite with template leakage elimination and independent validation set.",
        },
        "cas_con150_leakage_safe": {
            "n_train": len(ls_train),
            "n_test": len(ls_test),
            "unconditioned_baseline": baseline_ls_res.model_dump(),
            "condition_aware_model": ca_ls_res.model_dump(),
            "ablations": ablations_ls,
        },
        "cas_con150_historical": {
            "n_train": len(hist_train),
            "n_test": len(hist_test),
            "unconditioned_baseline": baseline_hist_res.model_dump(),
            "condition_aware_model": ca_hist_res.model_dump(),
        },
        "independent_benchmark": {
            "total_pairs": len(indep_pairs),
            "unconditioned_baseline": baseline_indep_res.model_dump(),
            "condition_aware_model": ca_indep_res.model_dump(),
            "ablations": ablations_indep,
        },
    }

    out_file = Path("data/evaluation_summary_corrected.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================ EVALUATION SUMMARY (CORRECTED) ================")
    print(f"1. CAS-Con150 Leakage-Safe Test Pairs: {len(ls_test)}")
    print(f"   TF-IDF Baseline Accuracy:       {baseline_ls_res.accuracy:.4f} | Macro-F1: {baseline_ls_res.macro_f1:.4f}")
    print(f"   Condition-Aware Model Accuracy: {ca_ls_res.accuracy:.4f} | Macro-F1: {ca_ls_res.macro_f1:.4f}")
    print(f"   Ablation (No Setting) Macro-F1: {ablations_ls['no_setting_ablation']['macro_f1']:.4f}")
    print(f"   Ablation (No Incomp)  Macro-F1: {ablations_ls['no_incomparable_ablation']['macro_f1']:.4f}")

    print(f"\n2. CAS-Con150 Historical (Leaky) Test Pairs: {len(hist_test)}")
    print(f"   TF-IDF Baseline Accuracy:       {baseline_hist_res.accuracy:.4f} | Macro-F1: {baseline_hist_res.macro_f1:.4f}")
    print(f"   Condition-Aware Model Accuracy: {ca_hist_res.accuracy:.4f} | Macro-F1: {ca_hist_res.macro_f1:.4f}")

    print(f"\n3. Independent Benchmark Test Pairs: {len(indep_pairs)}")
    print(f"   TF-IDF Baseline Accuracy:       {baseline_indep_res.accuracy:.4f} | Macro-F1: {baseline_indep_res.macro_f1:.4f}")
    print(f"   Condition-Aware Model Accuracy: {ca_indep_res.accuracy:.4f} | Macro-F1: {ca_indep_res.macro_f1:.4f}")
    print(f"   Ablation (No Setting) Macro-F1: {ablations_indep['no_setting_ablation']['macro_f1']:.4f}")
    print(f"   Ablation (No Incomp)  Macro-F1: {ablations_indep['no_incomparable_ablation']['macro_f1']:.4f}")
    print(f"\nSaved full results to {out_file}")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
