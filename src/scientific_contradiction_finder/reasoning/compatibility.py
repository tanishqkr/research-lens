"""Condition compatibility and difference evaluator."""

import re
from typing import Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scientific_contradiction_finder.schemas import ConditionSlots, ConditionDifference


def compute_string_similarity(s1: str, s2: str) -> float:
    """Computes token Jaccard similarity and character overlap between two slot strings."""
    if not s1 or not s2:
        return 0.0
    t1 = set(re.findall(r"\b\w+\b", s1.lower()))
    t2 = set(re.findall(r"\b\w+\b", s2.lower()))
    if not t1 or not t2:
        return 0.0
    inter = len(t1.intersection(t2))
    union = len(t1.union(t2))
    jaccard = inter / union

    # Check substring containment bonus
    if s1.lower() in s2.lower() or s2.lower() in s1.lower():
        jaccard = max(jaccard, 0.8)

    return float(min(1.0, jaccard))


class ConditionCompatibilityEvaluator:
    """Evaluates divergence and compatibility across experimental condition slots (I, C, O, S)."""

    def __init__(self, match_threshold: float = 0.55):
        self.match_threshold = match_threshold

    def evaluate_difference(self, cond_a: ConditionSlots, cond_b: ConditionSlots) -> ConditionDifference:
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
                summary_of_differences="One or both findings contain unspecified/unknown experimental conditions."
            )

        sim_i = compute_string_similarity(cond_a.intervention, cond_b.intervention)
        sim_c = compute_string_similarity(cond_a.comparator, cond_b.comparator)
        sim_o = compute_string_similarity(cond_a.outcome_metric, cond_b.outcome_metric)
        sim_s = compute_string_similarity(cond_a.dataset_setting, cond_b.dataset_setting)

        match_i = sim_i >= self.match_threshold
        match_c = sim_c >= self.match_threshold
        match_o = sim_o >= self.match_threshold
        match_s = sim_s >= self.match_threshold

        # Weighted overall compatibility: Intervention and Outcome carry highest weight
        weights = [0.35, 0.20, 0.30, 0.15]
        overall = (sim_i * weights[0] + sim_c * weights[1] + sim_o * weights[2] + sim_s * weights[3])

        diff_notes = []
        if not match_i:
            diff_notes.append(f"Intervention differs: '{cond_a.intervention}' vs '{cond_b.intervention}'")
        if not match_c:
            diff_notes.append(f"Comparator differs: '{cond_a.comparator}' vs '{cond_b.comparator}'")
        if not match_o:
            diff_notes.append(f"Outcome metric differs: '{cond_a.outcome_metric}' vs '{cond_b.outcome_metric}'")
        if not match_s:
            diff_notes.append(f"Setting/Cohort differs: '{cond_a.dataset_setting}' vs '{cond_b.dataset_setting}'")

        summary = "; ".join(diff_notes) if diff_notes else "Experimental conditions match closely across all dimensions."

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
            summary_of_differences=summary
        )
