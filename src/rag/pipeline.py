"""
End-to-end RAG pipeline: ingest documents and answer questions.
"""

import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

from src.loaders.document_loader import DocumentLoader
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.chunker import RecursiveChunker
from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.llm.llm_client import LLMClient

logger = logging.getLogger(__name__)


class HealthcareRAGPipeline:
    """
    Orchestrates document ingestion and question answering.

    Ingestion flow:
        file → loader → cleaner → chunker → embedder → vector store

    Query flow:
        question → embedder → vector store → retriever → LLM → answer
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel = None,
        vector_store: VectorStore = None,
        llm_client: LLMClient = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
    ):
        self.loader = DocumentLoader()
        self.cleaner = TextCleaner()
        self.chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or VectorStore()
        self.llm_client = llm_client
        self.retriever = Retriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            default_top_k=top_k,
        )
        self.question_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        file_path: str,
        metadata: Dict[str, Any] = None,
    ) -> Dict:
        """
        Full ingestion pipeline for a single document.

        metadata can include: title, document_type, disease_area,
                              drug_name, publication_year
        Returns a summary dict.
        """
        meta = metadata or {}
        document_id = str(uuid.uuid4())

        # 1. Load
        loaded = self.loader.load(file_path)
        raw_text = loaded["text"]
        if not raw_text.strip():
            raise ValueError(f"No text could be extracted from: {file_path}")

        # 2. Clean
        cleaned_text = self.cleaner.clean(raw_text)
        clean_stats = self.cleaner.get_statistics(raw_text, cleaned_text)

        # 3. Chunk
        base_meta = {
            "document_id": document_id,
            "title": meta.get("title", loaded["source_file"]),
            "document_type": meta.get("document_type", "Unknown"),
            "disease_area": meta.get("disease_area", "General"),
            "drug_name": meta.get("drug_name", "N/A"),
            "publication_year": str(meta.get("publication_year", "")),
            "source_file": loaded["source_file"],
            "upload_date": datetime.now().isoformat(),
            "file_type": loaded["file_type"],
        }
        chunks = self.chunker.split_with_metadata(cleaned_text, base_meta)

        if not chunks:
            raise ValueError("Chunking produced no usable chunks.")

        # 4. Embed
        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.embed_batch(texts, show_progress=True)

        # 5. Store
        ids = self.vector_store.add_chunks(chunks, embeddings)

        summary = {
            "document_id": document_id,
            "title": base_meta["title"],
            "source_file": loaded["source_file"],
            "total_chunks": len(chunks),
            "chunk_ids": ids,
            "raw_chars": clean_stats["original_chars"],
            "cleaned_chars": clean_stats["cleaned_chars"],
            "reduction_pct": clean_stats["reduction_pct"],
            "file_size_kb": loaded["file_size_kb"],
            "upload_date": base_meta["upload_date"],
        }
        logger.info(f"Ingested '{base_meta['title']}': {len(chunks)} chunks stored.")
        return summary

    # ------------------------------------------------------------------
    # Question Answering
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> Dict:
        """
        Answer a question using the RAG pipeline.

        Returns a dict with: answer, citations, retrieved_chunks,
                             confidence_score, response_time_s, provider, model
        """
        if self.llm_client is None:
            raise RuntimeError("No LLM client configured. Set LLM API key in .env.")

        # Retrieve
        chunks = self.retriever.retrieve_with_scores(
            question, top_k=top_k, filters=filters, min_score=min_score
        )

        if not chunks:
            return self._no_context_response(question)

        # Build context
        context = self.retriever.format_context(chunks)
        citations = self.retriever.get_source_citations(chunks)

        # Generate answer
        result = self.llm_client.generate(
            user_prompt=question,
            context=context,
        )

        # Confidence score: average similarity of retrieved chunks
        scores = [c.get("score", 0) for c in chunks]
        confidence = round(sum(scores) / len(scores), 4) if scores else 0.0

        response = {
            "question": question,
            "answer": result["answer"],
            "citations": citations,
            "retrieved_chunks": chunks,
            "confidence_score": confidence,
            "num_chunks_retrieved": len(chunks),
            "response_time_s": result["response_time_s"],
            "tokens_used": result.get("tokens_used", 0),
            "provider": result["provider"],
            "model": result["model"],
            "timestamp": datetime.now().isoformat(),
            "filters_applied": filters or {},
        }

        self.question_history.append({
            "question": question,
            "answer": result["answer"][:200] + "..." if len(result["answer"]) > 200 else result["answer"],
            "confidence_score": confidence,
            "num_chunks": len(chunks),
            "timestamp": response["timestamp"],
        })

        return response

    def _no_context_response(self, question: str) -> Dict:
        return {
            "question": question,
            "answer": "The provided documents do not contain sufficient information to answer this question. "
                      "Please upload relevant healthcare documents first.",
            "citations": [],
            "retrieved_chunks": [],
            "confidence_score": 0.0,
            "num_chunks_retrieved": 0,
            "response_time_s": 0.0,
            "tokens_used": 0,
            "provider": "N/A",
            "model": "N/A",
            "timestamp": datetime.now().isoformat(),
            "filters_applied": {},
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_system_stats(self) -> Dict:
        store_stats = self.vector_store.collection_stats()
        return {
            **store_stats,
            "embedding_model": self.embedding_model.model_name,
            "embedding_dimension": self.embedding_model.dimension,
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
            "questions_asked": len(self.question_history),
            "llm_provider": self.llm_client.provider if self.llm_client else "Not configured",
            "llm_model": self.llm_client.model if self.llm_client else "N/A",
        }
