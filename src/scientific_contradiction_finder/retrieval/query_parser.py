"""Polarity query expansion and syntactic claim parsing using spaCy."""

import re
from typing import List

import spacy
from pydantic import BaseModel


class ExpandedQuery(BaseModel):
    original_query: str
    expanded_queries: List[str]
    keyword_queries: List[str]
    polarity_inversions: List[str]
    comparison_queries: List[str]


# Common conversational question stop-words to strip for clean scholarly search
QUESTION_STOP_WORDS = {
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "can",
    "could",
    "will",
    "would",
    "should",
    "may",
    "might",
    "has",
    "have",
    "had",
    "what",
    "which",
    "how",
    "why",
    "the",
    "a",
    "an",
    "of",
    "in",
    "to",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "about",
    "into",
}

# Curated scientific polarity and antonym mappings
POLARITY_ANTONYMS = {
    "accelerate": "decelerate",
    "accelerates": "decelerates",
    "accelerated": "decelerated",
    "degrade": "stabilize",
    "degrades": "stabilizes",
    "degradation": "stability",
    "increases": "decreases",
    "increase": "decrease",
    "increased": "decreased",
    "improves": "degrades",
    "improve": "degrade",
    "improved": "degraded",
    "reduces": "increases",
    "reduce": "increase",
    "reduced": "increased",
    "prevents": "promotes",
    "prevent": "promote",
    "prevented": "promoted",
    "lowers": "raises",
    "lower": "raise",
    "lowered": "raised",
    "enhances": "impairs",
    "enhance": "impair",
    "enhanced": "impaired",
    "effective": "ineffective",
    "superior": "inferior",
    "beneficial": "harmful",
    "stimulates": "inhibits",
    "stimulate": "inhibit",
    "stimulated": "inhibited",
}


def extract_keywords_from_query(query: str) -> str:
    """Strips auxiliary question frames and punctuation to produce high-precision scholarly keywords."""
    # Remove question mark and punctuation
    cleaned = re.sub(r"[?!.,;:\"'()\[\]]", " ", query.lower())
    tokens = cleaned.split()
    # Filter question stop words
    keywords = [t for t in tokens if t not in QUESTION_STOP_WORDS and len(t) > 1]
    if keywords:
        return " ".join(keywords)
    return query.rstrip("?!.,;:")


class PolarityQueryExpander:
    """Expands empirical queries into keyword-focused, opposing polarity, and comparative search variants."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception:
            self.nlp = None

    def expand(self, query: str) -> ExpandedQuery:
        tokens = query.split()
        polarity_variants: List[str] = []

        # 1. High-precision keyword representation for API search engines
        core_keywords = extract_keywords_from_query(query)

        # 2. Lexical Antonym Inversion
        inversion_tokens = []
        replaced = False
        for tok in tokens:
            lower = tok.lower().strip(".,:;!?")
            if lower in POLARITY_ANTONYMS:
                opp = POLARITY_ANTONYMS[lower]
                inversion_tokens.append(opp)
                replaced = True
            else:
                inversion_tokens.append(tok)

        if replaced:
            polarity_variants.append(" ".join(inversion_tokens))
            inv_keywords = extract_keywords_from_query(" ".join(inversion_tokens))
            if inv_keywords and inv_keywords != core_keywords:
                polarity_variants.append(inv_keywords)
        else:
            polarity_variants.append(f"no significant effect on {core_keywords}")

        # 3. Comparative / Benchmark Variant
        comparison_variants = [
            f"{core_keywords} comparative evaluation",
            f"{core_keywords} trial benchmark",
        ]

        # Prioritize core keyword query first for API search success, followed by variants
        all_expanded = []
        if core_keywords:
            all_expanded.append(core_keywords)
        if query not in all_expanded:
            all_expanded.append(query)
        for v in polarity_variants:
            if v not in all_expanded:
                all_expanded.append(v)
        for c in comparison_variants:
            if c not in all_expanded:
                all_expanded.append(c)

        return ExpandedQuery(
            original_query=query,
            expanded_queries=all_expanded,
            keyword_queries=[core_keywords],
            polarity_inversions=polarity_variants,
            comparison_queries=comparison_variants,
        )
