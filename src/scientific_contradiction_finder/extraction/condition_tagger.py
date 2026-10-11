"""Condition slot tagger extracting Intervention (I), Comparator (C), Outcome (O), Setting (S), Population, and Qualifiers."""

import re
from typing import Optional

import spacy

from scientific_contradiction_finder.schemas import ConditionSlots

# Predefined lexical triggers for comparators, outcomes, and settings
COMPARATOR_MARKERS = [
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bcompared with\b",
    r"\bcompared to\b",
    r"\bagainst\b",
    r"\brelative to\b",
    r"\bthan\b",
]

OUTCOME_MARKERS = [
    r"\bdaytime air temperature\b",
    r"\bair temperature\b",
    r"\bland surface temperatures?\b",
    r"\bmean radiant temperature\b",
    r"\blocal temperature\b",
    r"\breaction times?\b",
    r"\bworking memory (?:performance|accuracy|capacity)?\b",
    r"\bvisual working memory\b",
    r"\bcognitive performance\b",
    r"\bdegradation rate\b",
    r"\bcapacity fade\b",
    r"\bcell thickness\b",
    r"\bmortality\b",
    r"\bhba1c\b",
    r"\bexacerbation\b",
    r"\bf1 score\b",
    r"\baccuracy\b",
    r"\bgeneralization\b",
    r"\bperplexity\b",
    r"\befficacy\b",
    r"\bspeedup\b",
    r"\brespiratory (?:tract )?infections?\b",
    r"\bincidence\b",
    r"\brisk of (?:[a-zA-Z\s]+)\b",
    r"\blifetime\b",
    r"\baging effects?\b",
]

SETTING_MARKERS = [
    r"\bpatients with\b",
    r"\bcohort\b",
    r"\bbenchmark\b",
    r"\bdataset\b",
    r"\bhealthy adults\b",
    r"\byoung adults\b",
    r"\bolder adults\b",
    r"\badults\b",
    r"\badolescents\b",
    r"\btrials?\b",
    r"\blithium-ion cells?\b",
    r"\bvision transformers?\b",
    r"\burban areas?\b",
    r"\bhigh density\b",
    r"\blow density\b",
]

# Blacklisted strings that should NEVER be extracted as an intervention or outcome
INVALID_SLOT_TERMS = {
    "the results",
    "results",
    "the experimental results",
    "our results",
    "significant increases",
    "significant decrease",
    "significant difference",
    "addition",
    "findings",
    "our findings",
    "in addition",
    "furthermore",
    "it",
    "we",
    "study",
    "follow-up",
    "in the follow-up",
    "the 14-week follow-up",
    "the 14-week follow-up study",
    "the study",
    "authors",
    "the authors",
    "this paper",
    "conclusion",
    "conclusions",
    "the researcher",
    "the researchers",
    "the day",
    "a result",
    "the action",
    "the effect",
    "the magnitude",
    "these results",
    "many studies",
    "great promise",
    "a promising approach",
    "no negative consequences",
    "an impairment",
    "accuracy",
}


def clean_slot_string(s: str) -> Optional[str]:
    """Cleans punctuation and determines if string is a valid scientific condition."""
    if not s:
        return None
    cleaned = re.sub(r"^[,\.;:\s\(\)\[\]]+|[,\.;:\s\(\)\[\]]+$", "", s).strip()
    if len(cleaned) < 3:
        return None
    if cleaned.lower() in INVALID_SLOT_TERMS:
        return None
    # Filter strings starting with introductory time/study phrases
    if re.match(
        r"^(in the|during the|across the)\s+\d+-(?:week|month|year|day)", cleaned, re.IGNORECASE
    ):
        return None
    return cleaned


class ConditionTagger:
    """Extracts experimental condition slots (Intervention, Comparator, Outcome, Setting) with validation."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception:
            self.nlp = None

    def extract_conditions(
        self, sentence_text: str, abstract_text: Optional[str] = None
    ) -> ConditionSlots:
        """Extracts PICO condition slots from finding sentence and abstract context."""
        text_lower = sentence_text.lower()

        # 1. Comparator extraction
        comparator = None
        for pattern in COMPARATOR_MARKERS:
            match = re.search(
                f"{pattern}\\s+([a-zA-Z0-9\\s\\-\\(\\)]{{3,40}}?)(?:\\.|,|\\bin\\b|\\bwith\\b|\\bfor\\b|\\bduring\\b|$)",
                text_lower,
            )
            if match:
                extracted_c = clean_slot_string(match.group(1).strip())
                if extracted_c:
                    comparator = extracted_c.title()
                break

        # 2. Outcome extraction
        outcome = None
        for marker in OUTCOME_MARKERS:
            m = re.search(marker, text_lower)
            if m:
                outcome = clean_slot_string(m.group(0))
                if outcome:
                    outcome = outcome.title()
                    break

        if not outcome and self.nlp:
            doc = self.nlp(sentence_text)
            for chunk in doc.noun_chunks:
                if chunk.root.dep_ in ("dobj", "pobj") and len(chunk.text.split()) <= 4:
                    cleaned = clean_slot_string(chunk.text)
                    if cleaned:
                        outcome = cleaned.title()
                        break

        # 3. Intervention extraction
        intervention = None
        domain_interventions = [
            r"\burban tree canopy(?:\s+cover)?\b",
            r"\btree canopy(?:\s+cover)?\b",
            r"\bcanopy cover\b",
            r"\bincreasing impervious cover\b",
            r"\bimpervious cover\b",
            r"\bpartial sleep restriction\b",
            r"\btotal sleep deprivation\b",
            r"\bsleep restriction\b",
            r"\bsleep deprivation\b",
            r"\b(?:high|elevated|ambient|low)\s+temperatures?(?:\s+charging)?\b",
            r"\bdata augmentation(?:\s+techniques?)?\b",
            r"\bvitamin d(?:3)?(?:\s+supplementation)?\b",
            r"\bbaseline 25\(oh\)d levels?(?:\s+below\s+\d+\s+nmol/l)?\b",
            r"\b(?:higher|lower)\s+charging(?:\s*-\s*discharging)?\s+rates?\b",
            r"\blithium plating\b",
            r"\bcell thickness increase\b",
        ]
        for ip in domain_interventions:
            im = re.search(ip, text_lower)
            if im:
                intervention = clean_slot_string(im.group(0).title())
                break

        if not intervention and self.nlp:
            doc = self.nlp(sentence_text)
            for chunk in doc.noun_chunks:
                if chunk.root.dep_ in ("nsubj", "nsubjpass"):
                    cleaned = clean_slot_string(chunk.text)
                    if cleaned:
                        intervention = cleaned.title()
                        break

        if not intervention and self.nlp:
            # Fallback: scan for first valid noun chunk not in invalid terms
            doc = self.nlp(sentence_text)
            for chunk in doc.noun_chunks:
                cleaned = clean_slot_string(chunk.text)
                if cleaned and cleaned.title() != outcome:
                    intervention = cleaned.title()
                    break

        # 4. Setting extraction
        setting = None
        for pattern in SETTING_MARKERS:
            match = re.search(
                f"{pattern}\\s*([a-zA-Z0-9\\s\\-\\(\\)]{{0,35}}?)(?:\\.|,|\\band\\b|$)", text_lower
            )
            if match:
                cleaned_p = clean_slot_string(match.group(0))
                if cleaned_p:
                    setting = cleaned_p.title()
                break

        # 5. Extract population and limitations if present
        population = None
        pop_m = re.search(
            r"\b(adults|healthy adults|young adults|older adults|children|adolescents|college students|patients|mice|rats)\b",
            text_lower,
        )
        if pop_m:
            population = pop_m.group(0).title()

        qualifiers = None
        qual_m = re.search(
            r"\b(?:when|where|if|after|for)\s+(?:canopy cover exceeded \d+%|\d+\s+hours?(?: of sleep)?|temperatures? above \d+°?c)",
            text_lower,
        )
        if qual_m:
            qualifiers = qual_m.group(0)

        is_unknown = bool(not intervention or not outcome)
        confidence = 0.50 if is_unknown else 0.85

        return ConditionSlots(
            intervention=intervention,
            comparator=comparator,
            outcome_metric=outcome,
            dataset_setting=setting,
            population=population,
            limitations_or_qualifiers=qualifiers,
            confidence=confidence,
            is_unknown=is_unknown,
        )
