"""Condition compatibility and difference evaluator using semantic embeddings and lexical overlap."""

import numpy as np
from sentence_transformers import SentenceTransformer

from scientific_contradiction_finder.schemas import ConditionDifference, ConditionSlots


class ConditionCompatibilityEvaluator:
    """Evaluates divergence and compatibility across experimental condition slots (I, C, O, S)."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        match_threshold: float = 0.50,
    ):
        self.model_name = model_name
        self.device = device
        self._model: SentenceTransformer | None = None
        self.match_threshold = match_threshold
        self._cache: dict[str, np.ndarray] = {}

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _encode_text(self, text: str) -> np.ndarray:
        norm = text.strip().lower()
        if norm not in self._cache:
            emb = self.model.encode([norm], convert_to_numpy=True, normalize_embeddings=True)[0]
            self._cache[norm] = emb
        return self._cache[norm]

    def canonicalize(self, text: str) -> str:
        if not text:
            return ""
        t = text.strip()
        t_lower = t.lower()
        from scientific_contradiction_finder.extraction.condition_tagger import (
            CANONICAL_COMPARATORS,
            CANONICAL_INTERVENTIONS,
            CANONICAL_OUTCOMES,
        )

        if t_lower in CANONICAL_INTERVENTIONS:
            return CANONICAL_INTERVENTIONS[t_lower]
        if t_lower in CANONICAL_COMPARATORS:
            return CANONICAL_COMPARATORS[t_lower]
        if t_lower in CANONICAL_OUTCOMES:
            return CANONICAL_OUTCOMES[t_lower]
        return t

    def compute_similarity(self, s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        c1 = self.canonicalize(s1)
        c2 = self.canonicalize(s2)
        # Quick exact / normalized match check
        if c1.strip().lower() == c2.strip().lower():
            return 1.0

        v1 = self._encode_text(c1)
        v2 = self._encode_text(c2)
        cos_sim = float(np.dot(v1, v2))
        return max(0.0, min(1.0, cos_sim))

    def evaluate_difference(
        self, cond_a: ConditionSlots, cond_b: ConditionSlots
    ) -> ConditionDifference:
        if cond_a.is_unknown or cond_b.is_unknown:
            return ConditionDifference(
                intervention_match=False,
                intervention_similarity=0.0,
                comparator_match=False,
                comparator_similarity=0.0,
                outcome_match=False,
                outcome_similarity=0.0,
                setting_match=False,
                setting_similarity=0.0,
                overall_compatibility=0.0,
                summary_of_differences="One or both findings contain unspecified/unknown experimental conditions.",
            )

        sim_i = self.compute_similarity(cond_a.intervention, cond_b.intervention)
        sim_c = self.compute_similarity(cond_a.comparator, cond_b.comparator)
        sim_o = self.compute_similarity(cond_a.outcome_metric, cond_b.outcome_metric)
        sim_s = self.compute_similarity(cond_a.dataset_setting, cond_b.dataset_setting)

        match_i = sim_i >= self.match_threshold
        match_c = sim_c >= self.match_threshold
        match_o = sim_o >= self.match_threshold
        match_s = sim_s >= self.match_threshold

        # Weighted overall compatibility: Intervention and Outcome carry primary weight
        weights = [0.35, 0.20, 0.30, 0.15]
        overall = float(
            sim_i * weights[0] + sim_c * weights[1] + sim_o * weights[2] + sim_s * weights[3]
        )

        diff_notes = []
        if not match_i:
            diff_notes.append(
                f"Intervention differs: '{cond_a.intervention}' vs '{cond_b.intervention}' (sim={sim_i:.2f})"
            )
        if not match_c:
            diff_notes.append(
                f"Comparator differs: '{cond_a.comparator}' vs '{cond_b.comparator}' (sim={sim_c:.2f})"
            )
        if not match_o:
            diff_notes.append(
                f"Outcome metric differs: '{cond_a.outcome_metric}' vs '{cond_b.outcome_metric}' (sim={sim_o:.2f})"
            )
        if not match_s:
            diff_notes.append(
                f"Setting/Cohort differs: '{cond_a.dataset_setting}' vs '{cond_b.dataset_setting}' (sim={sim_s:.2f})"
            )

        summary = (
            "; ".join(diff_notes)
            if diff_notes
            else "Experimental conditions match closely across all dimensions."
        )

        return ConditionDifference(
            intervention_match=match_i,
            intervention_similarity=sim_i,
            comparator_match=match_c,
            comparator_similarity=sim_c,
            outcome_match=match_o,
            outcome_similarity=sim_o,
            setting_match=match_s,
            setting_similarity=sim_s,
            overall_compatibility=overall,
            summary_of_differences=summary,
        )
