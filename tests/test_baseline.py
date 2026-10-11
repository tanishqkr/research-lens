import pytest
from scientific_contradiction_finder.ingestion.dataset_loader import BenchmarkLoader
from scientific_contradiction_finder.reasoning.baseline import LexicalUnconditionedBaseline


def test_lexical_baseline_train_eval():
    loader = BenchmarkLoader()
    pairs = loader.load_cas_con150()
    train_pairs, test_pairs = loader.get_train_test_split(pairs, test_ratio=0.3, random_seed=42)

    baseline = LexicalUnconditionedBaseline()
    baseline.fit(train_pairs)

    eval_results = baseline.evaluate(test_pairs)
    assert "accuracy" in eval_results
    assert "macro_f1" in eval_results
    assert eval_results["accuracy"] > 0.0
    print(f"Baseline accuracy: {eval_results['accuracy']:.4f}, Macro-F1: {eval_results['macro_f1']:.4f}")
