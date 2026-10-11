"""Polarity query expansion and syntactic claim parsing using spaCy."""

import spacy
from pydantic import BaseModel


class ExpandedQuery(BaseModel):
    original_query: str
    expanded_queries: list[str]
    polarity_inversions: list[str]
    comparison_queries: list[str]


# Curated scientific polarity and antonym mappings
POLARITY_ANTONYMS = {
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
    "accelerates": "decelerates",
    "stimulates": "inhibits",
    "stimulate": "inhibit",
    "stimulated": "inhibited",
}


class PolarityQueryExpander:
    """Expands empirical queries into opposing polarity and comparative search variants."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception:
            self.nlp = None

    def expand(self, query: str) -> ExpandedQuery:
        tokens = query.split()
        polarity_variants: list[str] = []

        # 1. Lexical Antonym Inversion
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

        # 2. Operator Inversion ("does not / no effect on")
        if not replaced:
            polarity_variants.append(f"no significant effect of {query}")
        else:
            polarity_variants.append(
                f"no effect on {' '.join(tokens[2:]) if len(tokens) > 2 else query}"
            )

        # 3. Comparative / Benchmark Variant
        comparison_variants = [
            f"{query} randomized controlled trial",
            f"{query} meta-analysis clinical trial",
            f"{query} versus baseline comparison",
        ]

        all_expanded = [query]
        for v in polarity_variants:
            if v not in all_expanded:
                all_expanded.append(v)
        for c in comparison_variants:
            if c not in all_expanded:
                all_expanded.append(c)

        return ExpandedQuery(
            original_query=query,
            expanded_queries=all_expanded,
            polarity_inversions=polarity_variants,
            comparison_queries=comparison_variants,
        )
