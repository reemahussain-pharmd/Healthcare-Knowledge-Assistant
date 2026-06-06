"""
ChromaDB vector store wrapper for healthcare document chunks.
"""

import os
import uuid
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

COLLECTION_NAME = "healthcare_documents"


class VectorStore:
    """
    Manages a persistent ChromaDB collection for healthcare RAG.
    """

    def __init__(self, persist_dir: str = "data/chroma_db"):
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None

    def _init(self):
        if self._client is not None:
            return
        try:
            import chromadb
            os.makedirs(self.persist_dir, exist_ok=True)

            # ChromaDB 1.x uses a different API — handle both versions
            try:
                # ChromaDB >= 1.0
                self._client = chromadb.PersistentClient(path=self.persist_dir)
            except TypeError:
                # Older ChromaDB
                from chromadb.config import Settings
                self._client = chromadb.PersistentClient(
                    path=self.persist_dir,
                    settings=Settings(anonymized_telemetry=False),
                )

            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB initialised. Collection: {COLLECTION_NAME}")
        except ImportError:
            raise ImportError("Install chromadb: pip install chromadb")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> List[str]:
        """
        Persist a batch of chunks with their embeddings.

        Each chunk dict must have at least 'text' plus any metadata fields.
        Returns the list of generated IDs.
        """
        self._init()

        ids, docs, metas, embeds = [], [], [], []
        for chunk, emb in zip(chunks, embeddings):
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            docs.append(chunk["text"])
            embeds.append(emb)

            meta = {k: v for k, v in chunk.items() if k != "text"}
            # ChromaDB metadata values must be str/int/float/bool
            sanitised = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    sanitised[k] = v
                else:
                    sanitised[k] = str(v)
            metas.append(sanitised)

        self._collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeds,
            metadatas=metas,
        )
        logger.info(f"Added {len(ids)} chunks to ChromaDB.")
        return ids

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar chunks.

        filters: dict of metadata field → value for pre-filtering.
        Returns list of dicts with keys: id, text, metadata, distance, score.
        """
        self._init()

        where_clause = self._build_where(filters)
        kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        if where_clause:
            kwargs["where"] = where_clause

        if kwargs["n_results"] == 0:
            return []

        results = self._collection.query(**kwargs)

        output = []
        for idx in range(len(results["ids"][0])):
            distance = results["distances"][0][idx]
            # cosine distance → similarity score (0–1)
            score = round(1 - distance, 4)
            output.append({
                "id": results["ids"][0][idx],
                "text": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "distance": round(distance, 4),
                "score": score,
            })

        return output

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def delete_by_document_id(self, document_id: str) -> int:
        self._init()
        results = self._collection.get(where={"document_id": document_id})
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} chunks for document_id={document_id}")
        return len(ids)

    def count(self) -> int:
        self._init()
        return self._collection.count()

    def get_all_metadata(self) -> List[Dict]:
        """Return metadata for every stored chunk (without embeddings)."""
        self._init()
        if self.count() == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        return result.get("metadatas", [])

    def get_unique_documents(self) -> List[Dict]:
        """Return one entry per unique document_id."""
        metas = self.get_all_metadata()
        seen = {}
        for m in metas:
            doc_id = m.get("document_id", "unknown")
            if doc_id not in seen:
                seen[doc_id] = m
        return list(seen.values())

    def reset(self):
        """Delete and recreate the collection (destructive)."""
        self._init()
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("ChromaDB collection reset.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_where(self, filters: Optional[Dict]) -> Optional[Dict]:
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            if value and value not in ("All", "all", ""):
                conditions.append({key: {"$eq": value}})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def collection_stats(self) -> Dict:
        metas = self.get_all_metadata()
        doc_types = {}
        disease_areas = {}
        for m in metas:
            dt = m.get("document_type", "Unknown")
            doc_types[dt] = doc_types.get(dt, 0) + 1
            da = m.get("disease_area", "Unknown")
            disease_areas[da] = disease_areas.get(da, 0) + 1
        return {
            "total_chunks": self.count(),
            "total_documents": len(self.get_unique_documents()),
            "document_types": doc_types,
            "disease_areas": disease_areas,
        }
