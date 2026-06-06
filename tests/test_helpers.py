"""Tests for utility helpers."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.helpers import (
    highlight_medical_terms,
    truncate_text,
    format_file_size,
    parse_publication_year,
)


def test_highlight_medical_terms():
    text = "The patient received Metformin for diabetes management."
    result = highlight_medical_terms(text)
    assert "**Metformin**" in result or "**metformin**" in result.lower()


def test_truncate_text():
    text = "This is a long piece of text that should be truncated at a word boundary."
    result = truncate_text(text, max_chars=30)
    assert len(result) <= 33  # 30 + "..."
    assert result.endswith("...")


def test_truncate_text_short():
    text = "Short text"
    result = truncate_text(text, max_chars=100)
    assert result == text


def test_format_file_size():
    assert "B" in format_file_size(500)
    assert "KB" in format_file_size(2048)
    assert "MB" in format_file_size(2 * 1024 ** 2)


def test_parse_publication_year():
    assert parse_publication_year("Published in 2023") == 2023
    assert parse_publication_year("From a 2019 study") == 2019
    assert parse_publication_year("No year here") is None
