"""
Text cleaning and normalization for healthcare documents.
"""

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

# Common medical abbreviations to preserve (not stripped)
MEDICAL_ABBREVIATIONS = {
    "mg", "ml", "mcg", "kg", "lb", "mmhg", "bpm", "bmi",
    "iv", "im", "sc", "po", "bid", "tid", "qid", "prn",
    "q.d.", "b.i.d.", "t.i.d.", "q.i.d.",
    "ae", "sae", "adrs", "ecg", "ekg", "mri", "ct", "pet",
    "rct", "rr", "or", "hr", "ci", "sd", "se", "p-value",
}

# Patterns to remove
_PAGE_NUMBER_PATTERNS = [
    r"^\s*-?\s*\d+\s*-?\s*$",           # standalone page numbers like "- 12 -" or "12"
    r"page\s+\d+\s+of\s+\d+",           # "Page 1 of 10"
    r"^\s*page\s+\d+\s*$",              # "Page 5"
]

_HEADER_FOOTER_PATTERNS = [
    r"confidential.*?page\s+\d+",
    r"draft.*?version\s+[\d\.]+",
    r"_{3,}",                            # separator lines ___
    r"-{3,}",                            # separator lines ---
    r"={3,}",                            # separator lines ===
]


class TextCleaner:
    """Clean and normalize extracted document text."""

    def clean(self, text: str) -> str:
        """Full cleaning pipeline."""
        text = self._normalize_unicode(text)
        text = self._remove_page_numbers(text)
        text = self._remove_headers_footers(text)
        text = self._remove_special_symbols(text)
        text = self._normalize_whitespace(text)
        text = self._fix_hyphenation(text)
        return text.strip()

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters to ASCII-compatible form."""
        text = unicodedata.normalize("NFKD", text)
        # Replace common unicode punctuation with ASCII equivalents
        replacements = {
            "’": "'", "‘": "'",   # curly single quotes
            "“": '"', "”": '"',   # curly double quotes
            "–": "-", "—": "-",   # en/em dash
            "•": "*",                   # bullet
            "·": "*",                   # middle dot bullet
            "®": "(R)", "™": "(TM)",
            "°": " degrees ",
            "μ": "u",                   # mu (micrograms)
            "≥": ">=", "≤": "<=",
            "±": "+/-",
        }
        for src, tgt in replacements.items():
            text = text.replace(src, tgt)
        return text

    def _remove_page_numbers(self, text: str) -> str:
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            is_page_num = any(
                re.match(pat, stripped, re.IGNORECASE)
                for pat in _PAGE_NUMBER_PATTERNS
            )
            if not is_page_num:
                cleaned.append(line)
        return "\n".join(cleaned)

    def _remove_headers_footers(self, text: str) -> str:
        for pat in _HEADER_FOOTER_PATTERNS:
            text = re.sub(pat, " ", text, flags=re.IGNORECASE | re.MULTILINE)
        return text

    def _remove_special_symbols(self, text: str) -> str:
        # Remove non-printable characters except newlines and tabs
        text = re.sub(r"[^\x09\x0a\x20-\x7e]", " ", text)
        # Remove excessive punctuation repetitions
        text = re.sub(r"([!?.]){3,}", r"\1\1\1", text)
        # Remove standalone special chars that add no meaning
        text = re.sub(r"\s[|\\~`^]\s", " ", text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        # Collapse multiple spaces/tabs to single space
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse more than 2 consecutive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip trailing spaces from each line
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines)

    def _fix_hyphenation(self, text: str) -> str:
        """Re-join words broken by end-of-line hyphens (common in PDFs)."""
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    def get_statistics(self, original: str, cleaned: str) -> dict:
        return {
            "original_chars": len(original),
            "cleaned_chars": len(cleaned),
            "reduction_pct": round((1 - len(cleaned) / max(len(original), 1)) * 100, 1),
            "original_words": len(original.split()),
            "cleaned_words": len(cleaned.split()),
        }
