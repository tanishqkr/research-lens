"""Comprehensive evaluation suite comparing Unconditioned vs Condition-Aware models and ablations."""

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from scientific_contradiction_finder.reasoning.classifier import ConditionAwareRelationClassifier
from scientific_contradiction_finder.schemas import (
    EvaluationResult,
    FindingPair,
    Prediction,
    RelationLabel,
)


class PipelineEvaluator:
    """Evaluates contradiction models, computes classwise metrics, and runs condition ablation studies."""

    def __init__(self):
        self.labels = [
            RelationLabel.SUPPORTS_OR_CONSISTENT.value,
            RelationLabel.CONTRADICTS.value,
            RelationLabel.QUALIFIES_OR_PARTIAL_DISAGREEMENT.value,
            RelationLabel.INCOMPARABLE_CONDITIONS.value,
            RelationLabel.INSUFFICIENT_EVIDENCE.value,
        ]

    def evaluate_predictions(
        self,
        pairs: list[FindingPair],
        predictions: list[Prediction],
        dataset_name: str = "CAS-Con150",
    ) -> EvaluationResult:
        y_true = [p.ground_truth_relation.value for p in pairs]
        y_pred = [p.predicted_relation.value for p in predictions]

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(
            f1_score(y_true, y_pred, labels=self.labels, average="macro", zero_division=0)
        )

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.labels, zero_division=0
        )
        per_class = {label: float(f) for label, f in zip(self.labels, f1)}
        per_class_prec = {label: float(p) for label, p in zip(self.labels, precision)}
        per_class_rec = {label: float(r) for label, r in zip(self.labels, recall)}
        per_class_supp = {label: int(s) for label, s in zip(self.labels, support)}

        cm = confusion_matrix(y_true, y_pred, labels=self.labels)
        cm_dict = {
            true_label: {pred_label: int(cm[i][j]) for j, pred_label in enumerate(self.labels)}
            for i, true_label in enumerate(self.labels)
        }

        abstained = sum(
            1 for p in predictions if p.predicted_relation == RelationLabel.INSUFFICIENT_EVIDENCE
        )

        return EvaluationResult(
            dataset_name=dataset_name,
            total_pairs=len(pairs),
            evaluated_pairs=len(pairs),
            abstained_count=abstained,
            accuracy=acc,
            macro_f1=macro_f1,
            per_class_f1=per_class,
            confusion_matrix=cm_dict,
            metadata={
                "per_class_precision": per_class_prec,
                "per_class_recall": per_class_rec,
                "per_class_support": per_class_supp,
            },
        )

    def run_ablation_study(self, test_pairs: list[FindingPair]) -> dict[str, Any]:
        """Ablation study comparing Condition-Aware full model against genuine feature-disabled variants."""
        clf = ConditionAwareRelationClassifier()
        full_preds = clf.classify_all(test_pairs)
        full_result = self.evaluate_predictions(
            test_pairs, full_preds, dataset_name="Full Condition-Aware"
        )

        # Genuine Ablation 1: Disable Setting Gating
        clf_no_setting = ConditionAwareRelationClassifier(enable_setting_gating=False)
        preds_no_setting = clf_no_setting.classify_all(test_pairs)
        no_setting_res = self.evaluate_predictions(
            test_pairs, preds_no_setting, dataset_name="Ablation: No Setting"
        )

        # Genuine Ablation 2: Disable Incomparability Check
        clf_no_incomp = ConditionAwareRelationClassifier(enable_incomparability_check=False)
        preds_no_incomp = clf_no_incomp.classify_all(test_pairs)
        no_incomp_res = self.evaluate_predictions(
            test_pairs, preds_no_incomp, dataset_name="Ablation: No Incomparable Check"
        )

        return {
            "full_model": full_result.model_dump(),
            "no_setting_ablation": no_setting_res.model_dump(),
            "no_incomparable_ablation": no_incomp_res.model_dump(),
        }
