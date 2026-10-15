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
    r"\breduced\s+[a-zA-Z\s]{0,20}\s+by\s+\d+",
    r"\bincreased\s+[a-zA-Z\s]{0,20}\s+by\s+\d+",
    r"\breduced weight\b",
    r"\breduced body weight\b",
    r"\bweight loss\b",
    r"\bweight reduction\b",
    r"\blost significantly more weight\b",
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
    r"\bodds ratio\s*=\s*\d+",
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
    r"\b\d+(?:\.\d+)?\s*per\s*cent\b",
    r"\bprevalence of\s+\d+",
    r"\bprevalence was\s+\d+",
    r"\battributable fraction\b",
    r"\bconditio sine qua non\b",
    r"\bnecessary cause\b",
]

# Patterns representing non-empirical statements, background promises, objectives, or metadata
NON_EMPIRICAL_REJECTION_PATTERNS = [
    r"^(?:background|introduction|objective|aim|aims|purpose|design|setting):",
    r"\b(?:this study|our study|the present study|we)\s+aimed\s+to\b",
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
    r"\bto evaluate whether\b",
    r"\bto investigate whether\b",
    r"\bto assess whether\b",
    r"\bthe aim of this\b",
    r"\bthe purpose of this\b",
    r"\bthe objective of\b",
    r"\bfurther research is needed\b",
    r"\bconflict of interest\b",
    r"\bauthors declare\b",
    r"\bfunding:\b",
    r"\backnowledgments\b",
    r"\bis common\b",
    r"\bare common\b",
    r"\bis increasingly recognized\b",
    r"\bhas received increasing attention\b",
    r"\bplays a crucial role\b",
    r"\bwere administered the structured clinical interview\b",
    r"\bwere recruited\b",
    r"\bwas conducted in accordance with\b",
    r"\bwritten informed consent was obtained\b",
    r"\bthis review summarizes\b",
    r"\bwe review\b",
    r"\ba systematic review of\b",
    r"\baccounts for \d+%\b",
    r"\bis the leading cause of\b",
    r"\bis a major public health\b",
    r"\bremains a leading cause\b",
    r"\bis an important risk factor\b",
    r"\bwere (?:enrolled|randomized|randomly assigned)\b",
    r"\bwere assigned to\b",
]

# Patterns signaling directionality
POSITIVE_PATTERNS = [
    r"\boutperformed\b",
    r"\boutperforms\b",
    r"\bsignificant improvement\b",
    r"\bincreased\b",
    r"\bincreases\b",
    r"\bincreasing\b",
    r"\bsurpassed\b",
    r"\benhanced\b",
    r"\benhances\b",
    r"\bhigher\b",
    r"\beffective\b",
    r"\befficacy\b",
    r"\baccelerated\b",
    r"\baccelerates\b",
    r"\bimproved\b",
    r"\bimproves\b",
    r"\bimprovement\b",
    r"\bbeneficial\b",
    r"\bgreater\s+(?:weight\s+loss|reduction|decrease|efficacy)\b",
    r"\bsuperior\s+(?:weight\s+loss|efficacy|performance)\b",
    r"\b(?:had\s+)?lost\s+(?:significantly\s+)?more\s+weight\b",
    r"\bsignificantly\s+greater\b",
    r"\bgreater\b",
]
NEGATIVE_PATTERNS = [
    r"\breduced\b",
    r"\breduces\b",
    r"\breducing\b",
    r"\breduction\b",
    r"\bdecreased\b",
    r"\bdecreases\b",
    r"\bdecreasing\b",
    r"\bdecrease\b",
    r"\blowered\b",
    r"\blowers\b",
    r"\blowering\b",
    r"\blower\b",
    r"\bdegraded\b",
    r"\bdegrades\b",
    r"\bdiminished\b",
    r"\bdiminishes\b",
    r"\binferior\b",
    r"\bharmful\b",
    r"\bdegradation\b",
    r"\bworse\b",
    r"\bworsened\b",
]
NO_DIFF_PATTERNS = [
    r"\bno significant difference\b",
    r"\bfailed to\b",
    r"\bdid not reduce\b",
    r"\bdoes not reduce\b",
    r"\bdid not improve\b",
    r"\bdoes not improve\b",
    r"\bdid not differ\b",
    r"\bdoes not differ\b",
    r"\bwas not statistically significant\b",
    r"\bwere not statistically significant\b",
    r"\bno effect\b",
    r"\bcomparable to\b",
    r"\bequivalent\b",
    r"\blittle to no difference\b",
    r"\bno clear difference\b",
]


UPPERCASE_EMPIRICAL_MEASUREMENT_PATTERNS = [
    r"\bOR\s*=\s*\d+",
]


class FindingExtractor:
    """Extracts candidate finding rationale sentences and classifies their directional effect with empirical filtering."""

    def __init__(self, min_length_chars: int = 30):
        self.min_length_chars = min_length_chars
        self.empirical_regex = re.compile("|".join(EMPIRICAL_MEASUREMENT_PATTERNS), re.IGNORECASE)
        self.uppercase_empirical_regex = re.compile("|".join(UPPERCASE_EMPIRICAL_MEASUREMENT_PATTERNS))
        self.rejection_regex = re.compile("|".join(NON_EMPIRICAL_REJECTION_PATTERNS), re.IGNORECASE)

    def extract_evidence_spans(self, text: str) -> List[EvidenceSpan]:
        sentences = SentenceSplitter.split_with_offsets(text)
        candidates: List[EvidenceSpan] = []

        for idx, (sent, start, end) in enumerate(sentences):
            if len(sent) < self.min_length_chars:
                continue

            sent_clean = sent.strip()

            # 1. Reject explicit non-empirical or background/promise phrases
            if self.rejection_regex.search(sent_clean):
                continue

            # 2. Require empirical measurement or finding indicator
            is_empirical = bool(
                self.empirical_regex.search(sent_clean)
                or self.uppercase_empirical_regex.search(sent_clean)
            )
            if not is_empirical:
                continue

            # Compute empirical confidence
            confidence = 0.50
            if is_empirical:
                confidence += 0.35
            if re.search(
                r"\b(?:\d+%(?:\s+ci)?|p\s*[<=]\s*0\.\d+|rr\s*=|ci\b)\b", sent_clean, re.IGNORECASE
            ) or self.uppercase_empirical_regex.search(sent_clean):
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

        # Check for mixed effects (opposing effects across measures or measures where one changes and other does not)
        if (
            ("reaction time" in text_lower or "reaction times" in text_lower)
            and "accuracy" in text_lower
            and (
                "without affecting" in text_lower
                or "intact" in text_lower
                or "unchanged" in text_lower
            )
        ):
            return DirectionLabel.MIXED

        if (
            "mixed results" in text_lower
            or "mixed effects" in text_lower
            or "both positive and negative" in text_lower
        ):
            return DirectionLabel.MIXED

        # Domain contextual cooling direction:
        # Increasing canopy cover decreasing temperature is POSITIVE (beneficial cooling / cooling effect)
        if ("canopy" in text_lower or "tree" in text_lower) and (
            "cooling" in text_lower or "decrease in" in text_lower or "reduce" in text_lower
        ):
            if "impervious" not in text_lower or "cooling" in text_lower:
                return DirectionLabel.NEGATIVE  # Decreases temperature

        # Sleep restriction increasing reaction time or causing impairment:
        if ("sleep restriction" in text_lower or "sleep deprivation" in text_lower) and (
            "impairment" in text_lower
            or "impaired" in text_lower
            or "longer reaction" in text_lower
            or "increased reaction" in text_lower
        ):
            return DirectionLabel.NEGATIVE  # Adverse cognitive effect / impairment

        # Check for explicit no difference before positive/negative
        for p in NO_DIFF_PATTERNS:
            if re.search(p, text_lower):
                return DirectionLabel.NO_DIFFERENCE

        # Check for comparative superiority or greater reduction/loss
        if re.search(
            r"\b(?:greater|superior|more|larger)\s+(?:weight\s+loss|reduction|decrease|efficacy)\b",
            text_lower,
        ):
            return DirectionLabel.POSITIVE
        if re.search(r"\b(?:had\s+)?lost\s+(?:significantly\s+)?(?:more\s+)?weight\b", text_lower):
            return DirectionLabel.POSITIVE
        if re.search(
            r"\b(?:achieved|produced|led to|resulted in|showed|demonstrated|significant)?\s*(?:weight\s+loss|weight\s+reduction|body\s+weight\s+reduction)\b",
            text_lower,
        ):
            return DirectionLabel.POSITIVE
        if re.search(r"\b(?:beneficial\s+in\s+decreasing|decreased|reduced)\s+body\s+weight\b", text_lower):
            return DirectionLabel.POSITIVE

        # Inverse associations with mortality or clinical events (protective lowering -> NEGATIVE)
        if re.search(
            r"\b(?:inversely\s+associated\s+with|protective\s+(?:against|association|effect)|lower\s+risk\s+of)\s+(?:all[\s-]cause\s+)?(?:mortality|death|cvd|stroke|cardiovascular|infection)\b",
            text_lower,
        ):
            return DirectionLabel.NEGATIVE

        # Relative risk / hazard ratio / odds ratio numerical direction
        # Note: Do not match lowercase conjunction "or" in text_lower; require explicit "odds ratio"
        # or uppercase "\bOR\s*=\s*" in the original unlowercased string.
        rr_m = re.search(
            r"\b(?:relative risk|risk ratio|hazard ratio|odds ratio|rr|hr)\s*(?:was|=|\:)\s*(\d+(?:\.\d+)?)\b",
            text_lower,
        ) or re.search(
            r"\bOR\s*(?:was|=|\:)\s*(\d+(?:\.\d+)?)\b",
            text,
        )
        if rr_m:
            ratio_val = float(rr_m.group(1))
            if ratio_val > 1.05:
                return DirectionLabel.POSITIVE
            elif ratio_val < 0.95:
                return DirectionLabel.NEGATIVE
            else:
                return DirectionLabel.NO_DIFFERENCE

        for p in POSITIVE_PATTERNS:
            if re.search(p, text_lower):
                return DirectionLabel.POSITIVE
        for p in NEGATIVE_PATTERNS:
            if re.search(p, text_lower):
                return DirectionLabel.NEGATIVE
        return DirectionLabel.UNCLEAR
