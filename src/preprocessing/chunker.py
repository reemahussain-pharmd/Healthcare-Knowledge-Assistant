"""
Recursive character-based text chunking with overlap.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Separator hierarchy: paragraph → sentence → clause → word → character
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


class RecursiveChunker:
    """
    Split text recursively on a hierarchy of separators.
    Falls back to the next separator when a chunk is still too large.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None,
        min_chunk_size: int = 100,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS
        self.min_chunk_size = min_chunk_size

    def split(self, text: str) -> List[str]:
        """Return a list of text chunks."""
        raw_chunks = self._split_recursive(text, self.separators)
        merged = self._merge_with_overlap(raw_chunks)
        # Drop tiny trailing fragments (but always keep at least one chunk)
        filtered = [c for c in merged if len(c.strip()) >= self.min_chunk_size]
        if not filtered and merged:
            return [merged[0]]
        return filtered

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text until all pieces fit within chunk_size."""
        if not separators:
            # Last resort: hard split by character count
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            splits = list(text)
        else:
            splits = text.split(sep)

        good_splits: List[str] = []
        current = ""

        for split in splits:
            candidate = (current + sep + split).lstrip(sep) if current else split
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    good_splits.append(current)
                # If the split itself is too big, recurse with finer separator
                if len(split) > self.chunk_size:
                    sub_chunks = self._split_recursive(split, remaining_seps)
                    good_splits.extend(sub_chunks)
                    current = ""
                else:
                    current = split

        if current:
            good_splits.append(current)

        return good_splits

    def _merge_with_overlap(self, chunks: List[str]) -> List[str]:
        """
        Re-merge small chunks up to chunk_size and add overlap between
        consecutive chunks so context is preserved across boundaries.
        """
        if not chunks:
            return []

        merged: List[str] = []
        buffer = ""

        for chunk in chunks:
            if not buffer:
                buffer = chunk
            elif len(buffer) + len(chunk) + 1 <= self.chunk_size:
                buffer = buffer + " " + chunk
            else:
                merged.append(buffer.strip())
                # Carry over the overlap window from the end of buffer
                overlap_text = buffer[-self.chunk_overlap :] if self.chunk_overlap > 0 else ""
                buffer = (overlap_text + " " + chunk).strip() if overlap_text else chunk

        if buffer.strip():
            merged.append(buffer.strip())

        return merged

    def split_with_metadata(self, text: str, base_metadata: dict) -> List[dict]:
        """Return chunks as dicts that include position metadata."""
        chunks = self.split(text)
        result = []
        for idx, chunk_text in enumerate(chunks):
            result.append({
                "text": chunk_text,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk_text),
                **base_metadata,
            })
        return result
