"""Condition slot tagger extracting Intervention (I), Comparator (C), Outcome (O), and Setting (S)."""

import re
from typing import Optional
import spacy
from scientific_contradiction_finder.schemas import ConditionSlots

# Predefined lexical triggers for biomedical and computational parameters
COMPARATOR_MARKERS = [r"\bversus\b", r"\bvs\.?\b", r"\bcompared with\b", r"\bcompared to\b", r"\bagainst\b"]
OUTCOME_MARKERS = [r"\bmortality\b", r"\bhba1c\b", r"\bexacerbation\b", r"\bf1 score\b", r"\baccuracy\b", r"\bperplexity\b", r"\befficacy\b", r"\bspeedup\b"]
SETTING_MARKERS = [r"\bpatients with\b", r"\bcohort\b", r"\bbenchmark\b", r"\bdataset\b", r"\badults\b", r"\btrials\b", r"\bsetting\b"]


class ConditionTagger:
    """Extracts 4-tuple experimental conditions: (Intervention, Comparator, Outcome, Setting)."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception:
            self.nlp = None

    def extract_conditions(self, sentence_text: str, abstract_text: Optional[str] = None) -> ConditionSlots:
        """Extracts PICO condition slots from finding sentence and optional abstract context."""
        text_lower = sentence_text.lower()

        # 1. Comparator extraction
        comparator = "Standard Baseline / Unspecified"
        for pattern in COMPARATOR_MARKERS:
            match = re.search(f"{pattern}\\s+([a-zA-Z0-9\\s\\-\\(\\)]{{3,35}}?)(?:\\.|,|\\bin\\b|\\bwith\\b|\\bfor\\b|$)", text_lower)
            if match:
                extracted_c = match.group(1).strip()
                if len(extracted_c) > 2:
                    comparator = extracted_c.title()
                break

        # 2. Intervention extraction (typically the grammatical subject or noun chunk before verb)
        intervention = "Unspecified Intervention"
        if self.nlp:
            doc = self.nlp(sentence_text)
            # Find subject noun phrase
            for chunk in doc.noun_chunks:
                # Ignore generic meta nouns
                if chunk.root.dep_ in ("nsubj", "nsubjpass") and chunk.text.lower() not in ("we", "results", "study", "trials", "data"):
                    intervention = chunk.text.strip()
                    break
        if intervention == "Unspecified Intervention":
            # Fallback regex for first substantial phrase
            first_words = " ".join(sentence_text.split()[:4]).rstrip(",.:;")
            intervention = first_words

        # 3. Outcome extraction
        outcome = "Primary Empirical Outcome"
        for marker in OUTCOME_MARKERS:
            if re.search(marker, text_lower):
                outcome = marker.replace("\\b", "").title()
                break
        if outcome == "Primary Empirical Outcome" and self.nlp:
            # Look for direct object or prepositional object
            doc = self.nlp(sentence_text)
            for chunk in doc.noun_chunks:
                if chunk.root.dep_ in ("dobj", "pobj") and len(chunk.text.split()) <= 4:
                    outcome = chunk.text.strip().title()
                    break

        # 4. Setting extraction
        setting = "General Scientific Setting"
        for pattern in SETTING_MARKERS:
            match = re.search(f"{pattern}\\s+([a-zA-Z0-9\\s\\-\\(\\)]{{3,35}}?)(?:\\.|,|\\band\\b|$)", text_lower)
            if match:
                cleaned_p = pattern.replace(r'\b', '').title()
                setting = f"{cleaned_p} {match.group(1).strip().title()}"
                break

        # Check if conditions are essentially unknown/sparse
        is_unknown = (intervention == "Unspecified Intervention" and comparator == "Standard Baseline / Unspecified")
        confidence = 0.5 if is_unknown else 0.85

        return ConditionSlots(
            intervention=intervention,
            comparator=comparator,
            outcome_metric=outcome,
            dataset_setting=setting,
            confidence=confidence,
            is_unknown=is_unknown
        )
