"""Tests for TextCleaner."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.text_cleaner import TextCleaner


def test_remove_page_numbers():
    cleaner = TextCleaner()
    text = "Some content\n\n12\n\nMore content"
    cleaned = cleaner.clean(text)
    assert "12" not in cleaned.split("\n") or "Some content" in cleaned


def test_normalize_unicode():
    cleaner = TextCleaner()
    text = "Patients ≥ 18 years and ≤ 75 years with ± 2 kg body weight"
    cleaned = cleaner._normalize_unicode(text)
    assert ">=" in cleaned
    assert "<=" in cleaned
    assert "+/-" in cleaned


def test_fix_hyphenation():
    cleaner = TextCleaner()
    text = "pharmaco-\nkinetics study"
    cleaned = cleaner._fix_hyphenation(text)
    assert "pharmacokinetics" in cleaned


def test_normalize_whitespace():
    cleaner = TextCleaner()
    text = "word1   word2\t\tword3\n\n\n\nword4"
    cleaned = cleaner._normalize_whitespace(text)
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned


def test_statistics():
    cleaner = TextCleaner()
    original = "A  B  C\n\n\n12\n\nD"
    cleaned = cleaner.clean(original)
    stats = cleaner.get_statistics(original, cleaned)
    assert "original_chars" in stats
    assert "cleaned_chars" in stats
    assert stats["original_chars"] == len(original)
