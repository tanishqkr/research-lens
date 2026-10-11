"""CLI Evaluation runner for benchmarks and baselines."""

import argparse
import json
from pathlib import Path
from scientific_contradiction_finder.ingestion.dataset_loader import BenchmarkLoader
from scientific_contradiction_finder.reasoning.baseline import LexicalUnconditionedBaseline


def run_evaluation(dataset_name: str = "cas_con150", output_file: str = "data/eval_results.json"):
    print(f"--- Running Evaluation on {dataset_name} ---")
    loader = BenchmarkLoader()
    if dataset_name == "cas_con150":
        pairs = loader.load_cas_con150()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_pairs, test_pairs = loader.get_train_test_split(pairs, test_ratio=0.3, random_seed=42)
    print(f"Dataset split: {len(train_pairs)} train pairs, {len(test_pairs)} test pairs.")

    baseline = LexicalUnconditionedBaseline()
    baseline.fit(train_pairs)
    eval_metrics = baseline.evaluate(test_pairs)

    print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"Macro-F1: {eval_metrics['macro_f1']:.4f}")

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"Saved evaluation metrics to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Scientific Contradiction Finder Pipeline")
    parser.add_argument("--dataset", type=str, default="cas_con150")
    parser.add_argument("--output", type=str, default="data/eval_results.json")
    args = parser.parse_args()
    run_evaluation(args.dataset, args.output)
