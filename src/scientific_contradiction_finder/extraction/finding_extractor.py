"""Finding rationale extractor identifying true empirical finding sentences from scientific abstracts."""

import re
from typing import List

from scientific_contradiction_finder.extraction.sentence_splitter import SentenceSplitter
from scientific_contradiction_finder.schemas import DirectionLabel, EvidenceSpan

# Keywords signaling observational or experimental measurements, statistical comparisons, and empirical findings
EMPIRICAL_MEASUREMENT_PATTERNS = [
    r"\bdemonstrated significant\b",
    r"\bshowed significant\b",
    r"\bshows significant\b",
    r"\breduced the risk\b",
    r"\breduced the incidence\b",
    r"\breduced by\s+\d+",
    r"\bincreased by\s+\d+",
    r"\baccelerated by\s+\d+",
    r"\bdegraded by\s+\d+",
    r"\bstatistically significant\b",
    r"\bp\s*[<=]\s*0\.\d+",
    r"\b95%\s*ci\b",
    r"\brisk ratio\b",
    r"\bhazard ratio\b",
    r"\bodds ratio\b",
    r"\brelative risk\b",
    r"\brr\s*=\s*\d+",
    r"\bhr\s*=\s*\d+",
    r"\bor\s*=\s*\d+",
    r"\boutperformed\b",
    r"\boutperforms\b",
    r"\bsurpassed\b",
    r"\baccuracy of\s+\d+",
    r"\bf1 (?:score|points)\b",
    r"\bperplexity of\s+\d+",
    r"\bfound that\b",
    r"\bresults (?:indicate|show|demonstrate|revealed) that\b",
    r"\bthe experimental results\b",
    r"\bcapacity fade\b",
    r"\bcell degradation\b",
    r"\bthickness increased\b",
    r"\bcycling at\b",
    r"\bfailure occurred\b",
    r"\bno significant difference\b",
    r"\bfailed to reduce\b",
    r"\bdid not reduce\b",
    r"\bhad no effect\b",
    r"\b(?:were |was )?associated with\b",
    r"\b(?:increased|reduced|higher|lower) risk\b",
    r"\b\d+%\s*(?:increased|reduced|higher|lower|decrease|reduction)?\b",
]

# Patterns representing non-empirical statements, background promises, objectives, or metadata
NON_EMPIRICAL_REJECTION_PATTERNS = [
    r"\bdemonstrated great promise\b",
    r"\bhave shown great promise\b",
    r"\bpromising approach\b",
    r"\bin this paper(?:,| we)\b",
    r"\bin this study(?:,| we)\b",
    r"\bwe propose\b",
    r"\bwe introduce\b",
    r"\bwe investigate\b",
    r"\bwe examine\b",
    r"\bto determine whether\b",
    r"\bthe aim of this\b",
    r"\bthe objective of\b",
    r"\bfurther research is needed\b",
    r"\bconflict of interest\b",
    r"\bauthors declare\b",
    r"\bfunding:\b",
    r"\backnowledgments\b",
]

# Patterns signaling directionality
POSITIVE_PATTERNS = [
    r"\boutperformed\b",
    r"\boutperforms\b",
    r"\bsignificant improvement\b",
    r"\bincreased\b",
    r"\bincreases\b",
    r"\bsurpassed\b",
    r"\benhanced\b",
    r"\benhances\b",
    r"\bhigher\b",
    r"\beffective\b",
    r"\baccelerated\b",
    r"\baccelerates\b",
    r"\bimproved\b",
    r"\bimproves\b",
]
NEGATIVE_PATTERNS = [
    r"\breduced\b",
    r"\breduces\b",
    r"\bdecreased\b",
    r"\bdecreases\b",
    r"\blowered\b",
    r"\blowers\b",
    r"\bdegraded\b",
    r"\bdegrades\b",
    r"\bdiminished\b",
    r"\bdiminishes\b",
    r"\binferior\b",
    r"\bharmful\b",
    r"\bdegradation\b",
]
NO_DIFF_PATTERNS = [
    r"\bno significant difference\b",
    r"\bfailed to\b",
    r"\bdid not reduce\b",
    r"\bdoes not reduce\b",
    r"\bdid not improve\b",
    r"\bdoes not improve\b",
    r"\bno effect\b",
    r"\bcomparable to\b",
    r"\bequivalent\b",
    r"\blittle to no difference\b",
]


class FindingExtractor:
    """Extracts candidate finding rationale sentences and classifies their directional effect with empirical filtering."""

    def __init__(self, min_length_chars: int = 30):
        self.min_length_chars = min_length_chars
        self.empirical_regex = re.compile("|".join(EMPIRICAL_MEASUREMENT_PATTERNS), re.IGNORECASE)
        self.rejection_regex = re.compile("|".join(NON_EMPIRICAL_REJECTION_PATTERNS), re.IGNORECASE)

    def extract_evidence_spans(self, text: str) -> List[EvidenceSpan]:
        sentences = SentenceSplitter.split_with_offsets(text)
        candidates: List[EvidenceSpan] = []

        for idx, (sent, start, end) in enumerate(sentences):
            if len(sent) < self.min_length_chars:
                continue

            sent_clean = sent.strip()

            # 1. Reject explicit non-empirical or promise phrases
            if self.rejection_regex.search(sent_clean):
                # Only salvage if it also has concrete statistics (e.g., p < 0.05 or % numbers)
                if not re.search(
                    r"\b(?:\d+%(?:\s+ci)?|p\s*[<=]\s*0\.\d+|risk ratio)\b",
                    sent_clean,
                    re.IGNORECASE,
                ):
                    continue

            # 2. Require empirical measurement, result indicator, or conclusion location
            is_empirical = bool(self.empirical_regex.search(sent_clean))
            is_late_sentence = idx >= max(1, len(sentences) - 2)

            if not is_empirical and not is_late_sentence:
                continue

            # Compute empirical confidence
            confidence = 0.50
            if is_empirical:
                confidence += 0.35
            if re.search(
                r"\b(?:\d+%(?:\s+ci)?|p\s*[<=]\s*0\.\d+|rr\s*=|ci\b)\b", sent_clean, re.IGNORECASE
            ):
                confidence += 0.10

            assert text[start:end] == sent, "Evidence span character offset mismatch!"
            candidates.append(
                EvidenceSpan(
                    text=sent_clean,
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
