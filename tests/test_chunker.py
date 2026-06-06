"""Tests for RecursiveChunker."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.preprocessing.chunker import RecursiveChunker


def test_basic_split():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    text = "Hello world. " * 50
    chunks = chunker.split(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100 + 20 + 50  # some tolerance for overlap


def test_overlap_in_consecutive_chunks():
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
    text = ("This is a sentence about hypertension. " * 20)
    chunks = chunker.split(text)
    # Check that consecutive chunks share some text
    if len(chunks) >= 2:
        last_words_of_first = set(chunks[0][-50:].split())
        first_words_of_second = set(chunks[1][:50].split())
        # There should be overlap
        assert len(last_words_of_first & first_words_of_second) > 0


def test_invalid_overlap():
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)


def test_split_with_metadata():
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
    text = "Clinical trial information. " * 30
    base_meta = {"document_id": "test-123", "title": "Test Doc"}
    result = chunker.split_with_metadata(text, base_meta)
    assert len(result) > 0
    for item in result:
        assert "text" in item
        assert "chunk_index" in item
        assert "document_id" in item
        assert item["document_id"] == "test-123"


def test_short_text_single_chunk():
    chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
    text = "Short medical summary."
    chunks = chunker.split(text)
    assert len(chunks) == 1
    assert chunks[0] == text
