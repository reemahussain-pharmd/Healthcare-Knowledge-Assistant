"""
High-level retriever: embeds a query then fetches relevant chunks.
"""

import logging
from typing import List, Dict, Optional, Any

from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Combines embedding + vector store for end-to-end semantic retrieval."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        default_top_k: int = 5,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        Embed *query* and return the top_k most relevant chunks.

        filters keys (all optional):
            disease_area, drug_name, document_type, publication_year
        """
        k = top_k or self.default_top_k
        query_embedding = self.embedding_model.embed_text(query)
        chunks = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=k,
            filters=filters,
        )
        logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:60]}...'")
        return chunks

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[Dict]:
        """Like retrieve() but filters out chunks below min_score."""
        chunks = self.retrieve(query, top_k=top_k, filters=filters)
        return [c for c in chunks if c.get("score", 0) >= min_score]

    def format_context(self, chunks: List[Dict]) -> str:
        """Build a formatted context string from retrieved chunks."""
        if not chunks:
            return "No relevant context found."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            title = meta.get("title", "Unknown Document")
            doc_type = meta.get("document_type", "")
            source = meta.get("source_file", "")
            score = chunk.get("score", 0)

            header = f"[Source {i}] {title}"
            if doc_type:
                header += f" ({doc_type})"
            if source:
                header += f" | File: {source}"
            header += f" | Relevance: {score:.2%}"

            parts.append(f"{header}\n{chunk['text']}")

        return "\n\n---\n\n".join(parts)

    def get_source_citations(self, chunks: List[Dict]) -> List[Dict]:
        """Extract citation info from retrieved chunks."""
        seen = set()
        citations = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            doc_id = meta.get("document_id", "")
            if doc_id not in seen:
                seen.add(doc_id)
                citations.append({
                    "document_id": doc_id,
                    "title": meta.get("title", "Unknown"),
                    "document_type": meta.get("document_type", ""),
                    "source_file": meta.get("source_file", ""),
                    "publication_year": meta.get("publication_year", ""),
                    "disease_area": meta.get("disease_area", ""),
                    "drug_name": meta.get("drug_name", ""),
                    "score": chunk.get("score", 0),
                })
        return citations
