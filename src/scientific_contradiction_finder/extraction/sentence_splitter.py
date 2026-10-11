"""Sentence boundary detection and character offset tracking."""

import re
from typing import List, Tuple


class SentenceSplitter:
    """Splits abstracts into sentences while tracking exact character offsets in the source text."""

    @staticmethod
    def split_with_offsets(text: str) -> List[Tuple[str, int, int]]:
        """Returns list of (sentence_text, start_char, end_char)."""
        if not text:
            return []

        # Find potential sentence ending punctuation followed by whitespace and an uppercase letter/number
        matches = list(re.finditer(r'[.!?]\s+(?=[A-Z0-9])', text))
        splits = []
        start_idx = 0

        for m in matches:
            end_idx = m.start() + 1
            sent_text = text[start_idx:end_idx].strip()
            if sent_text:
                # Find start char in original text
                actual_start = text.find(sent_text, start_idx)
                actual_end = actual_start + len(sent_text)
                splits.append((sent_text, actual_start, actual_end))
            start_idx = m.end()

        # Remainder
        remainder = text[start_idx:].strip()
        if remainder:
            actual_start = text.find(remainder, start_idx)
            actual_end = actual_start + len(remainder)
            splits.append((remainder, actual_start, actual_end))

        return splits
