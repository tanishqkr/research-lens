"""Comprehensive evaluation suite comparing Unconditioned vs Condition-Aware models and ablations."""

from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from scientific_contradiction_finder.schemas import FindingPair, Prediction, RelationLabel, EvaluationResult
from scientific_contradiction_finder.reasoning.classifier import ConditionAwareRelationClassifier
from scientific_contradiction_finder.reasoning.baseline import LexicalUnconditionedBaseline


class PipelineEvaluator:
    """Evaluates contradiction models, computes classwise metrics, and runs condition ablation studies."""

    def __init__(self):
        self.labels = [
            RelationLabel.SUPPORTS_OR_CONSISTENT.value,
            RelationLabel.CONTRADICTS.value,
            RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT.value,
            RelationLabel.INCOMPARABLE_CONDITIONS.value,
            RelationLabel.INSUFFICIENT_EVIDENCE.value
        ]

    def evaluate_predictions(
        self,
        pairs: List[FindingPair],
        predictions: List[Prediction],
        dataset_name: str = "CAS-Con150"
    ) -> EvaluationResult:
        y_true = [p.ground_truth_relation.value for p in pairs]
        y_pred = [p.predicted_relation.value for p in predictions]

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, labels=self.labels, average="macro", zero_division=0))

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=self.labels, zero_division=0
        )
        per_class = {label: float(f) for label, f in zip(self.labels, f1)}

        cm = confusion_matrix(y_true, y_pred, labels=self.labels)
        cm_dict = {
            true_label: {pred_label: int(cm[i][j]) for j, pred_label in enumerate(self.labels)}
            for i, true_label in enumerate(self.labels)
        }

        abstained = sum(1 for p in predictions if p.predicted_relation == RelationLabel.INSUFFICIENT_EVIDENCE)

        return EvaluationResult(
            dataset_name=dataset_name,
            total_pairs=len(pairs),
            evaluated_pairs=len(pairs),
            abstained_count=abstained,
            accuracy=acc,
            macro_f1=macro_f1,
            per_class_f1=per_class,
            confusion_matrix=cm_dict
        )

    def run_ablation_study(self, test_pairs: List[FindingPair]) -> Dict[str, Any]:
        """Ablation study comparing Condition-Aware full model against ablated variants."""
        clf = ConditionAwareRelationClassifier()
        full_preds = clf.classify_all(test_pairs)
        full_result = self.evaluate_predictions(test_pairs, full_preds, dataset_name="Full Condition-Aware")

        # Ablation 1: Ignore Setting Divergence (Force setting_match = True)
        preds_no_setting = []
        for p in test_pairs:
            pred = clf.classify_pair(p)
            # If it was qualified because of setting, it becomes contradiction
            if pred.predicted_relation == RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT:
                pred.predicted_relation = RelationLabel.CONTRADICTS
            preds_no_setting.append(pred)
        no_setting_res = self.evaluate_predictions(test_pairs, preds_no_setting, dataset_name="Ablation: No Setting")

        # Ablation 2: Ignore Incomparability (Force incomparable pairs into contradiction/neutral)
        preds_no_incomp = []
        for p in test_pairs:
            pred = clf.classify_pair(p)
            if pred.predicted_relation == RelationLabel.INCOMPARABLE_CONDITIONS:
                pred.predicted_relation = RelationLabel.CONTRADICTS
            preds_no_incomp.append(pred)
        no_incomp_res = self.evaluate_predictions(test_pairs, preds_no_incomp, dataset_name="Ablation: No Incomparable Check")

        return {
            "full_model": full_result.model_dump(),
            "no_setting_ablation": no_setting_res.model_dump(),
            "no_incomparable_ablation": no_incomp_res.model_dump()
        }
