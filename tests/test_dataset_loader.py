from scientific_contradiction_finder.ingestion.dataset_loader import BenchmarkLoader
from scientific_contradiction_finder.schemas import RelationLabel


def test_load_cas_con150():
    loader = BenchmarkLoader()
    pairs = loader.load_cas_con150()
    assert len(pairs) == 150

    # Test label distribution
    labels = [p.ground_truth_relation for p in pairs]
    assert labels.count(RelationLabel.SUPPORTS_OR_CONSISTENT) == 30
    assert labels.count(RelationLabel.CONTRADICTS) == 30
    assert labels.count(RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT) == 40
    assert labels.count(RelationLabel.INCOMPARABLE_CONDITIONS) == 30
    assert labels.count(RelationLabel.INSUFFICIENT_EVIDENCE) == 20


def test_stratified_split_integrity():
    loader = BenchmarkLoader()
    pairs = loader.load_cas_con150()
    train_pairs, test_pairs = loader.get_train_test_split(pairs, test_ratio=0.3, random_seed=42)

    assert len(train_pairs) + len(test_pairs) == 150
    # No leakage
    train_ids = {p.pair_id for p in train_pairs}
    test_ids = {p.pair_id for p in test_pairs}
    assert len(train_ids.intersection(test_ids)) == 0

    # Ensure all 5 classes are present in both splits
    train_labels = {p.ground_truth_relation for p in train_pairs}
    test_labels = {p.ground_truth_relation for p in test_pairs}
    assert len(train_labels) == 5
    assert len(test_labels) == 5


def test_leakage_safe_claim_family_isolation():
    loader = BenchmarkLoader()
    pairs = loader.load_cas_con150()

    train_pairs, test_pairs = loader.get_train_test_split(
        pairs, test_ratio=0.3, random_seed=42, leakage_safe=True
    )
    assert len(train_pairs) + len(test_pairs) == 150

    # Ensure zero claim family overlap
    train_claims = {(p.finding_a.finding_claim, p.finding_b.finding_claim) for p in train_pairs}
    test_claims = {(p.finding_a.finding_claim, p.finding_b.finding_claim) for p in test_pairs}
    assert len(train_claims.intersection(test_claims)) == 0


def test_load_independent_eval_set():
    loader = BenchmarkLoader()
    pairs = loader.load_independent_eval_set()
    assert len(pairs) == 35

    # Check balanced 5-class distribution (7 per class)
    labels = [p.ground_truth_relation for p in pairs]
    assert labels.count(RelationLabel.SUPPORTS_OR_CONSISTENT) == 7
    assert labels.count(RelationLabel.CONTRADICTS) == 7
    assert labels.count(RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT) == 7
    assert labels.count(RelationLabel.INCOMPARABLE_CONDITIONS) == 7
    assert labels.count(RelationLabel.INSUFFICIENT_EVIDENCE) == 7

    # Ensure all have valid evidence, conditions, and notes/provenance
    for p in pairs:
        assert len(p.finding_a.finding_claim) > 0
        assert len(p.finding_b.finding_claim) > 0
        assert p.finding_a.conditions is not None
        assert p.finding_b.conditions is not None
        assert p.notes is not None and len(p.notes) > 0
