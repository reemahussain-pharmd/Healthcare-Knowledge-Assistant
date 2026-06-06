"""Tests for DocumentLoader using the bundled sample TXT files."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.loaders.document_loader import DocumentLoader

SAMPLE_TXT = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "documents", "sample_clinical_trial.txt"
)


def test_load_txt():
    loader = DocumentLoader()
    result = loader.load(SAMPLE_TXT)
    assert "text" in result
    assert len(result["text"]) > 100
    assert result["file_type"] == "TXT"
    assert "source_file" in result


def test_unsupported_format():
    loader = DocumentLoader()
    with pytest.raises(ValueError, match="Unsupported"):
        loader.load("somefile.xlsx")


def test_file_not_found():
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent_file.txt")


def test_supported_formats():
    loader = DocumentLoader()
    formats = loader.get_supported_formats()
    assert ".pdf" in formats
    assert ".docx" in formats
    assert ".txt" in formats
