"""Finding rationale extractor identifying empirical finding sentences from scientific abstracts."""

import re

from scientific_contradiction_finder.extraction.sentence_splitter import SentenceSplitter
from scientific_contradiction_finder.schemas import (
    DirectionLabel,
    EvidenceSpan,
)

# Keywords signaling empirical results, findings, and conclusions
FINDING_INDICATORS = [
    r"\bdemonstrated\b",
    r"\bshowed\b",
    r"\breduced\b",
    r"\bincreased\b",
    r"\bimproved\b",
    r"\bfound that\b",
    r"\bconclude\b",
    r"\bassociated with\b",
    r"\bsignificantly\b",
    r"\boutperformed\b",
    r"\bsurpassed\b",
    r"\bresulted in\b",
    r"\bno significant difference\b",
    r"\bfailed to\b",
    r"\bled to\b",
    r"\beachieved\b",
]

# Patterns signaling directionality
POSITIVE_PATTERNS = [
    r"\boutperformed\b",
    r"\bsignificant improvement\b",
    r"\bincreased\b",
    r"\bsurpassed\b",
    r"\benhanced\b",
    r"\bhigher\b",
    r"\beffective\b",
]
NEGATIVE_PATTERNS = [
    r"\breduced\b",
    r"\bdecreased\b",
    r"\blowered\b",
    r"\bdegraded\b",
    r"\bdiminished\b",
    r"\binferior\b",
]
NO_DIFF_PATTERNS = [
    r"\bno significant difference\b",
    r"\bfailed to\b",
    r"\bdid not reduce\b",
    r"\bno effect\b",
    r"\bcomparable to\b",
    r"\bequivalent\b",
]


class FindingExtractor:
    """Extracts candidate finding rationale sentences and classifies their directional effect."""

    def __init__(self, min_length_chars: int = 25):
        self.min_length_chars = min_length_chars
        self.finding_regex = re.compile("|".join(FINDING_INDICATORS), re.IGNORECASE)

    def extract_evidence_spans(self, text: str) -> list[EvidenceSpan]:
        sentences = SentenceSplitter.split_with_offsets(text)
        candidates: list[EvidenceSpan] = []

        for idx, (sent, start, end) in enumerate(sentences):
            if len(sent) < self.min_length_chars:
                continue

            # Prioritize sentences with empirical result keywords or late in abstract (Results/Conclusion)
            match = self.finding_regex.search(sent)
            is_late_sentence = idx >= max(1, len(sentences) - 3)

            confidence = 0.5
            if match:
                confidence += 0.35
            if is_late_sentence:
                confidence += 0.15

            if match or is_late_sentence:
                # Substring provenance guarantee: sent is verified against text[start:end]
                assert text[start:end] == sent, "Evidence span character offset mismatch!"
                candidates.append(
                    EvidenceSpan(
                        text=sent,
                        start_char=start,
                        end_char=end,
                        sentence_idx=idx,
                        confidence=min(1.0, confidence),
                    )
                )

        candidates.sort(key=lambda s: s.confidence, reverse=True)
        return candidates

    def infer_direction(self, text: str) -> DirectionLabel:
        text_lower = text.lower()
        for p in NO_DIFF_PATTERNS:
            if re.search(p, text_lower):
                return DirectionLabel.NO_DIFFERENCE
        for p in POSITIVE_PATTERNS:
            if re.search(p, text_lower):
                return DirectionLabel.POSITIVE
        for p in NEGATIVE_PATTERNS:
            if re.search(p, text_lower):
                return DirectionLabel.NEGATIVE
        return DirectionLabel.UNCLEAR
