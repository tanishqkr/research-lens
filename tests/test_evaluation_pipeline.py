import pytest
from scientific_contradiction_finder.ingestion.dataset_loader import BenchmarkLoader
from scientific_contradiction_finder.reasoning.classifier import ConditionAwareRelationClassifier
from scientific_contradiction_finder.evaluation.evaluator import PipelineEvaluator


def test_full_condition_aware_evaluation():
    loader = BenchmarkLoader()
    pairs = loader.load_cas_con150()
    train_pairs, test_pairs = loader.get_train_test_split(pairs, test_ratio=0.3, random_seed=42)

    classifier = ConditionAwareRelationClassifier()
    predictions = classifier.classify_all(test_pairs)

    evaluator = PipelineEvaluator()
    result = evaluator.evaluate_predictions(test_pairs, predictions, dataset_name="CAS-Con150 Test")

    assert result.accuracy > 0.80
    assert result.macro_f1 > 0.80
    assert result.total_pairs == len(test_pairs)


def test_ablation_study():
    loader = BenchmarkLoader()
    pairs = loader.load_cas_con150()
    _, test_pairs = loader.get_train_test_split(pairs, test_ratio=0.3, random_seed=42)

    evaluator = PipelineEvaluator()
    ablations = evaluator.run_ablation_study(test_pairs)

    full_f1 = ablations["full_model"]["macro_f1"]
    no_setting_f1 = ablations["no_setting_ablation"]["macro_f1"]
    no_incomp_f1 = ablations["no_incomparable_ablation"]["macro_f1"]

    # Validating the core research hypothesis: removing condition parameters degrades Macro-F1
    assert full_f1 >= no_setting_f1
    assert full_f1 > no_incomp_f1
