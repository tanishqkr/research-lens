"""Simple TF-IDF & lexical similarity baselines for scientific relationship classification."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from scientific_contradiction_finder.schemas import (
    ConditionDifference,
    FindingPair,
    Prediction,
    RelationLabel,
)


class LexicalUnconditionedBaseline:
    """An unconditioned baseline that compares finding claim text alone using TF-IDF."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=500, stop_words="english", ngram_range=(1, 2)
        )
        self.clf = LogisticRegression(class_weight="balanced", max_iter=200, random_state=42)
        self.is_fitted = False

    def _extract_pair_features(self, pairs: list[FindingPair], fit: bool = False) -> np.ndarray:
        texts_a = [p.finding_a.finding_claim for p in pairs]
        texts_b = [p.finding_b.finding_claim for p in pairs]

        if fit:
            all_texts = texts_a + texts_b
            self.vectorizer.fit(all_texts)

        vecs_a = self.vectorizer.transform(texts_a).toarray()
        vecs_b = self.vectorizer.transform(texts_b).toarray()

        # Compute cosine similarity, elementwise absolute diff, and concatenation
        cos_sim = np.sum(vecs_a * vecs_b, axis=1, keepdims=True)
        # normalize
        norm_a = np.linalg.norm(vecs_a, axis=1, keepdims=True) + 1e-9
        norm_b = np.linalg.norm(vecs_b, axis=1, keepdims=True) + 1e-9
        cos_sim = cos_sim / (norm_a * norm_b)

        abs_diff = np.abs(vecs_a - vecs_b)
        features = np.hstack([vecs_a, vecs_b, abs_diff, cos_sim])
        return features

    def fit(self, train_pairs: list[FindingPair]):
        X_train = self._extract_pair_features(train_pairs, fit=True)
        y_train = [p.ground_truth_relation.value for p in train_pairs]
        self.clf.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, pairs: list[FindingPair]) -> list[Prediction]:
        if not self.is_fitted:
            raise RuntimeError("Baseline classifier must be fitted before predict.")

        X = self._extract_pair_features(pairs, fit=False)
        preds = self.clf.predict(X)
        probs = self.clf.predict_proba(X)
        max_probs = np.max(probs, axis=1)

        results = []
        for pair, pred_label, conf in zip(pairs, preds, max_probs):
            diff = ConditionDifference(
                intervention_match=False,
                intervention_similarity=0.5,
                comparator_match=False,
                comparator_similarity=0.5,
                outcome_match=False,
                outcome_similarity=0.5,
                setting_match=False,
                setting_similarity=0.5,
                overall_compatibility=0.5,
                summary_of_differences="Unconditioned lexical baseline (conditions not parsed).",
            )
            prediction = Prediction(
                pair_id=pair.pair_id,
                predicted_relation=RelationLabel(pred_label),
                confidence_score=float(conf),
                unconditioned_relation=RelationLabel(pred_label),
                condition_differences=diff,
                provenance_verified=True,
                explanation=f"Predicted by unconditioned TF-IDF baseline with confidence {conf:.2f}.",
            )
            results.append(prediction)
        return results

    def evaluate(self, test_pairs: list[FindingPair]) -> dict:
        predictions = self.predict(test_pairs)
        y_true = [p.ground_truth_relation.value for p in test_pairs]
        y_pred = [p.predicted_relation.value for p in predictions]

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        return {"accuracy": acc, "macro_f1": macro_f1, "classification_report": report}
